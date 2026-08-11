//! Low-level operations of Ext4 filesystem.
//!
//! These interfaces are designed and arranged coresponding to FUSE low-level ops.
//! Ref: https://libfuse.github.io/doxygen/structfuse__lowlevel__ops.html

use super::{Ext4, WriteLogicalRange};
use crate::constants::*;
use crate::ext4_defs::*;
use crate::format_error;
use crate::prelude::*;
use crate::return_error;
use core::cmp::min;

#[macro_export]
macro_rules! println {
    ($($arg:tt)*) => {
        log::error!($($arg)*);
    };
}

// A page-cache writeback may contain only one dirty page.  Keep it in the
// transactional append path so consecutive pages extend the same deferred
// journal batch instead of entering the direct path and forcing a commit.
const DIRECT_RANGE_MIN_BLOCKS: usize = 1;
const DIRECT_RANGE_MAX_BLOCKS: usize = 256;
const DIRECT_RANGE_ZERO_CHUNK_BLOCKS: usize = 8;
/// Bound the metadata footprint of a writeback batch. Journal-ring pressure
/// supplies an additional dynamic limit in `defer_or_commit`.
const MAX_DEFERRED_JOURNAL_BLOCKS: usize = 256;

enum DirectRangePrepare {
    Initialized,
    DataWritten,
    Unsupported,
}

struct DirectRangePlan {
    start_lblock: LBlockId,
    count: u32,
    preferred_first: Option<PBlockId>,
    requires_merge: bool,
}

/// Attributes that can be set on an inode via `setattr`.
#[derive(Default)]
pub struct SetAttr {
    /// File mode and permissions
    pub mode: Option<InodeMode>,
    /// 32-bit user id
    pub uid: Option<u32>,
    /// 32-bit group id
    pub gid: Option<u32>,
    /// 64-bit file size
    pub size: Option<u64>,
    /// 32-bit access time in seconds
    pub atime: Option<u32>,
    /// 32-bit modify time in seconds
    pub mtime: Option<u32>,
    /// 32-bit change time in seconds
    pub ctime: Option<u32>,
    /// 32-bit create time in seconds
    pub crtime: Option<u32>,
}

#[derive(Clone, Copy)]
pub struct InodeOwner {
    pub uid: u32,
    pub gid: u32,
}

impl SetAttr {
    /// Create a new SetAttr struct with all fields set to None.
    pub fn new() -> Self {
        Self::default()
    }

    fn is_writeback_metadata_only(&self) -> bool {
        self.mode.is_none()
            && self.uid.is_none()
            && self.gid.is_none()
            && (self.size.is_some()
                || self.atime.is_some()
                || self.mtime.is_some()
                || self.ctime.is_some()
                || self.crtime.is_some())
    }

    fn apply_to(&self, inode: &mut Inode) {
        if let Some(mode) = self.mode {
            inode.set_mode(mode);
        }
        if let Some(uid) = self.uid {
            inode.set_uid(uid);
        }
        if let Some(gid) = self.gid {
            inode.set_gid(gid);
        }
        if let Some(size) = self.size {
            inode.set_size(size);
        }
        if let Some(atime) = self.atime {
            inode.set_atime(atime);
        }
        if let Some(mtime) = self.mtime {
            inode.set_mtime(mtime);
        }
        if let Some(ctime) = self.ctime {
            inode.set_ctime(ctime);
        }
        if let Some(crtime) = self.crtime {
            inode.set_crtime(crtime);
        }
    }
}

impl Ext4 {
    fn initialize_direct_range(&self, first: PBlockId, count: usize, zeros: &[u8]) -> Result<()> {
        let mut done = 0usize;
        while done < count {
            let blocks = min(DIRECT_RANGE_ZERO_CHUNK_BLOCKS, count - done);
            let pblock = first
                .checked_add(done as PBlockId)
                .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            self.block_device
                .write_blocks(pblock, &zeros[..blocks * BLOCK_SIZE])?;
            done += blocks;
        }
        Ok(())
    }

    fn write_direct_range_data(&self, first: PBlockId, count: usize, data: &[u8]) -> Result<()> {
        let total = count
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
        if data.len() < total {
            return Err(Ext4Error::new(ErrCode::EINVAL));
        }

        self.block_device.write_blocks(first, &data[..total])?;
        Ok(())
    }

    fn direct_range_plan(
        &self,
        inode: &InodeRef,
        range: &WriteLogicalRange,
    ) -> Result<Option<DirectRangePlan>> {
        let count = u32::try_from(range.block_count).map_err(|_| Ext4Error::new(ErrCode::EFBIG))?;
        if range.first_lblock.checked_add(count).is_none()
            || range.block_count < DIRECT_RANGE_MIN_BLOCKS
            || range.block_count > DIRECT_RANGE_MAX_BLOCKS
        {
            crate::println!(
                "[ext4_diag] plan_reject:out_of_bounds lblock={} count={} min={} max={}",
                range.first_lblock,
                range.block_count,
                DIRECT_RANGE_MIN_BLOCKS,
                DIRECT_RANGE_MAX_BLOCKS
            );
            return Ok(None);
        }
        let persistent_blocks = inode
            .inode
            .size()
            .checked_add(BLOCK_SIZE as u64 - 1)
            .map(|size| size / BLOCK_SIZE as u64)
            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
        if (range.first_lblock as u64) < persistent_blocks {
            crate::println!(
                "[ext4_diag] plan_reject:before_eof lblock={} persistent={} count={}",
                range.first_lblock,
                persistent_blocks,
                range.block_count
            );
            return Ok(None);
        }
        let Some(shape) = self.direct_append_shape(inode, range.first_lblock, count)? else {
            crate::println!(
                "[ext4_diag] plan_reject:append_shape lblock={} count={} size={}",
                range.first_lblock,
                count,
                inode.inode.size()
            );
            return Ok(None);
        };
        // [ext4_diag] plan_accept — commented out to reduce serial noise
        Ok(Some(DirectRangePlan {
            start_lblock: range.first_lblock,
            count,
            preferred_first: shape.preferred_first,
            requires_merge: shape.requires_merge,
        }))
    }

    fn try_prepare_direct_range(
        &self,
        inode: &mut InodeRef,
        range: &WriteLogicalRange,
        real_data: Option<&[u8]>,
    ) -> Result<DirectRangePrepare> {
        let Some(plan) = self.direct_range_plan(inode, range)? else {
            crate::println!(
                "[ext4_diag] direct_reject:plan_unsupported lblock={} count={}",
                range.first_lblock,
                range.block_count
            );
            return Ok(DirectRangePrepare::Unsupported);
        };

        let total = (plan.count as usize)
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
        let real_data = real_data.and_then(|data| data.get(..total));
        let zeros = if real_data.is_none() {
            let zero_bytes = DIRECT_RANGE_ZERO_CHUNK_BLOCKS * BLOCK_SIZE;
            let mut zeros = Vec::new();
            if zeros.try_reserve_exact(zero_bytes).is_err() {
                return Ok(DirectRangePrepare::Unsupported);
            }
            zeros.resize(zero_bytes, 0);
            Some(zeros)
        } else {
            None
        };

        let mut transaction = self.transaction_start_direct_range(4)?;
        let allocation = match self.transaction_alloc_direct_range(
            &mut transaction,
            inode.id,
            plan.preferred_first,
            plan.requires_merge,
            plan.count,
        ) {
            Ok(allocation) => allocation,
            Err(error) if error.code() == ErrCode::ENOSPC => {
                transaction.abort();
                crate::println!(
                    "[ext4_diag] direct_reject:enospc lblock={} count={}",
                    plan.start_lblock,
                    plan.count
                );
                return Ok(DirectRangePrepare::Unsupported);
            }
            Err(error) => return Err(error),
        };

        self.prepare_stats.record_call();
        self.prepare_stats.record_requested(plan.count as usize);
        self.prepare_stats
            .record_missing_blocks(plan.count as usize);
        let initialized = match real_data {
            Some(data) => self.write_direct_range_data(allocation.first, plan.count as usize, data),
            None => self.initialize_direct_range(
                allocation.first,
                plan.count as usize,
                zeros
                    .as_deref()
                    .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?,
            ),
        };
        if let Err(error) = initialized {
            self.prepare_stats.record_failure();
            transaction.abort();
            if error.code() == ErrCode::ENOMEM {
                return Ok(DirectRangePrepare::Unsupported);
            }
            return Err(error);
        }

        if let Err(error) =
            self.stage_direct_append_extent(inode, plan.start_lblock, allocation.first, plan.count)
        {
            self.prepare_stats.record_failure();
            transaction.abort();
            return Err(error);
        }
        let (inode_home, _) = self.inode_disk_pos(inode.id)?;
        if let Err(error) = self.transaction_stage_inode_with_csum(&mut transaction, inode) {
            self.prepare_stats.record_failure();
            transaction.abort();
            return Err(error);
        }
        self.prepare_stats.record_inode_io();
        if let Err(error) = transaction.commit_direct_range(
            self.block_device.as_ref(),
            self,
            &allocation.allocation_homes,
            inode_home,
        ) {
            self.prepare_stats.record_failure();
            if error.failure != super::journal_transaction::CommitFailure::BeforeCommit {
                self.poison(ErrCode::EIO);
            }
            return Err(error.error);
        }
        self.prepare_stats.record_bitmap_io();
        self.prepare_stats.record_gdt_io();
        self.prepare_stats.record_superblock_io();
        self.prepare_stats.record_inode_io();
        match real_data {
            Some(_) => {
                // [ext4_diag] direct_ok:data_written — commented out
                Ok(DirectRangePrepare::DataWritten)
            }
            None => {
                // [ext4_diag] direct_ok:initialized — commented out
                Ok(DirectRangePrepare::Initialized)
            }
        }
    }

    /// Allocate an append range through one JBD2 transaction.
    ///
    /// The data blocks are initialized before their extent is journaled.  The
    /// journal commit flushes that data before publishing the metadata commit,
    /// so a crash can expose neither uninitialized data nor a live extent with
    /// free allocation metadata.
    fn try_prepare_journal_range(
        &self,
        inode_id: InodeId,
        range: &WriteLogicalRange,
        real_data: Option<&[u8]>,
    ) -> Result<DirectRangePrepare> {
        let diagnostic = self.block_device.diagnostic_enabled();
        let prepare_start = if diagnostic {
            self.block_device.diagnostic_cycles()
        } else {
            0
        };
        // Read through the active transaction so consecutive deferred batches
        // extend the latest staged inode image instead of overwriting it.
        let mut transaction = self.transaction_start(4)?;
        let mut inode = self.transaction_read_inode(&transaction, inode_id)?;
        if inode.inode.mode().bits() == 0 {
            transaction.abort();
            return_error!(ErrCode::EINVAL, "Invalid inode {}", inode_id);
        }
        let plan = match self.direct_range_plan(&inode, range) {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                transaction.abort();
                return Ok(DirectRangePrepare::Unsupported);
            }
            Err(error) => {
                transaction.abort();
                return Err(error);
            }
        };

        let total = match (plan.count as usize).checked_mul(BLOCK_SIZE) {
            Some(total) => total,
            None => {
                transaction.abort();
                return Err(Ext4Error::new(ErrCode::EFBIG));
            }
        };
        let real_data = real_data.and_then(|data| data.get(..total));
        let zeros = if real_data.is_none() {
            let zero_bytes = DIRECT_RANGE_ZERO_CHUNK_BLOCKS * BLOCK_SIZE;
            let mut zeros = Vec::new();
            if zeros.try_reserve_exact(zero_bytes).is_err() {
                transaction.abort();
                return Ok(DirectRangePrepare::Unsupported);
            }
            zeros.resize(zero_bytes, 0);
            Some(zeros)
        } else {
            None
        };

        // The range fits one block group, so bitmap, GDT, superblock, and inode
        // are the complete bounded set of transaction homes.
        let allocation = match self.transaction_alloc_direct_range(
            &mut transaction,
            inode.id,
            plan.preferred_first,
            plan.requires_merge,
            plan.count,
        ) {
            Ok(allocation) => allocation,
            Err(error) if error.code() == ErrCode::ENOSPC => {
                transaction.abort();
                return Ok(DirectRangePrepare::Unsupported);
            }
            Err(error) => {
                transaction.abort();
                return Err(error);
            }
        };

        self.prepare_stats.record_call();
        self.prepare_stats.record_requested(plan.count as usize);
        self.prepare_stats
            .record_missing_blocks(plan.count as usize);
        let data_start = if diagnostic {
            self.block_device.diagnostic_cycles()
        } else {
            0
        };
        let initialized = match real_data {
            Some(data) => self.write_direct_range_data(allocation.first, plan.count as usize, data),
            None => self.initialize_direct_range(
                allocation.first,
                plan.count as usize,
                zeros
                    .as_deref()
                    .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?,
            ),
        };
        let data_cycles = if diagnostic {
            self.block_device
                .diagnostic_cycles()
                .wrapping_sub(data_start)
        } else {
            0
        };
        if let Err(error) = initialized {
            self.prepare_stats.record_failure();
            transaction.abort();
            if error.code() == ErrCode::ENOMEM {
                return Ok(DirectRangePrepare::Unsupported);
            }
            return Err(error);
        }

        if let Err(error) = self.stage_direct_append_extent(
            &mut inode,
            plan.start_lblock,
            allocation.first,
            plan.count,
        ) {
            self.prepare_stats.record_failure();
            transaction.abort();
            return Err(error);
        }
        if let Err(error) = self.transaction_stage_inode_with_csum(&mut transaction, &mut inode) {
            self.prepare_stats.record_failure();
            transaction.abort();
            return Err(error);
        }
        self.prepare_stats.record_inode_io();
        if let Err(error) = transaction.defer_or_commit(
            self.block_device.as_ref(),
            self,
            MAX_DEFERRED_JOURNAL_BLOCKS,
        ) {
            self.prepare_stats.record_failure();
            if error.failure != super::journal_transaction::CommitFailure::BeforeCommit {
                self.poison(ErrCode::EIO);
            }
            return Err(error.error);
        }
        self.prepare_stats.record_bitmap_io();
        self.prepare_stats.record_gdt_io();
        self.prepare_stats.record_superblock_io();
        self.prepare_stats.record_inode_io();
        if diagnostic {
            let prepare_cycles = self
                .block_device
                .diagnostic_cycles()
                .wrapping_sub(prepare_start);
            self.block_device
                .record_writeback_data_write(total, data_cycles);
            self.block_device.record_writeback_alloc_extent(
                plan.count as usize,
                prepare_cycles.saturating_sub(data_cycles),
            );
        }
        Ok(match real_data {
            Some(_) => DirectRangePrepare::DataWritten,
            None => DirectRangePrepare::Initialized,
        })
    }

    /// Persist a write to already-mapped blocks without entering the direct
    /// metadata domain.
    ///
    /// A deferred transaction may contain the newest inode/extent image, so
    /// validate mappings through that image and keep both mutation guards held
    /// until its data blocks have been written.  This preserves data-before-
    /// metadata ordering without prematurely committing the deferred batch.
    fn try_prepare_journal_mapped_write(
        &self,
        inode_id: InodeId,
        range: &WriteLogicalRange,
        offset: usize,
        data: &[u8],
    ) -> Result<DirectRangePrepare> {
        let diagnostic = self.block_device.diagnostic_enabled();
        let prepare_start = if diagnostic {
            self.block_device.diagnostic_cycles()
        } else {
            0
        };
        let mut transaction = self.transaction_start(2)?;
        let mut inode = self.transaction_read_inode(&transaction, inode_id)?;
        if inode.inode.mode().bits() == 0 {
            transaction.abort();
            return_error!(ErrCode::EINVAL, "Invalid inode {}", inode_id);
        }
        for lblock in range.first_lblock..=range.last_lblock {
            match self.extent_query_with_range(&inode, lblock) {
                Ok(_) => {}
                Err(error) if error.code() == ErrCode::ENOENT => {
                    transaction.abort();
                    return Ok(DirectRangePrepare::Unsupported);
                }
                Err(error) => {
                    transaction.abort();
                    return Err(error);
                }
            }
        }
        let data_start = if diagnostic {
            self.block_device.diagnostic_cycles()
        } else {
            0
        };
        let data_written = self.write_journaled_mapped_data(&inode, offset, data)?;
        let data_cycles = if diagnostic {
            self.block_device
                .diagnostic_cycles()
                .wrapping_sub(data_start)
        } else {
            0
        };
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
        if end as u64 > inode.inode.size() {
            inode.inode.set_size(end as u64);
            self.transaction_stage_inode_with_csum(&mut transaction, &mut inode)?;
        }
        if let Err(error) = transaction.defer_or_commit(
            self.block_device.as_ref(),
            self,
            MAX_DEFERRED_JOURNAL_BLOCKS,
        ) {
            if error.failure != super::journal_transaction::CommitFailure::BeforeCommit {
                self.poison(ErrCode::EIO);
            }
            return Err(error.error);
        }
        if diagnostic {
            let prepare_cycles = self
                .block_device
                .diagnostic_cycles()
                .wrapping_sub(prepare_start);
            self.block_device
                .record_writeback_data_write(data_written, data_cycles);
            self.block_device.record_writeback_alloc_extent(
                range.block_count as usize,
                prepare_cycles.saturating_sub(data_cycles),
            );
        }
        Ok(DirectRangePrepare::DataWritten)
    }

    fn xattr_checksum_seed(&self) -> Result<Option<MetadataChecksumSeed>> {
        let sb = self.read_super_block_cached();
        if !sb.has_read_only_compatible_feature(SuperBlock::FEATURE_RO_COMPAT_METADATA_CSUM) {
            return Ok(None);
        }
        Ok(Some(sb.metadata_checksum_seed()))
    }

    fn verify_xattr_block_checksum(&self, block_id: PBlockId, block: &XattrBlock) -> Result<()> {
        if let Some(seed) = self.xattr_checksum_seed()? {
            if !block.verify_checksum(seed, block_id) {
                return Err(Ext4Error::new(ErrCode::EIO));
            }
        }
        Ok(())
    }

    fn update_xattr_block_checksum(
        &self,
        block_id: PBlockId,
        block: &mut XattrBlock,
    ) -> Result<()> {
        if let Some(seed) = self.xattr_checksum_seed()? {
            if !block.update_checksum(seed, block_id) {
                return Err(Ext4Error::new(ErrCode::EIO));
            }
        }
        Ok(())
    }

    fn read_extent_or_hole(
        &self,
        file: &InodeRef,
        iblock: LBlockId,
        block_offset: usize,
        buf: &mut [u8],
    ) -> Result<()> {
        match self.extent_query(file, iblock) {
            Ok(fblock) => {
                let block = self.read_block(fblock)?;
                buf.copy_from_slice(block.read_offset(block_offset, buf.len()));
            }
            Err(err) if err.code() == ErrCode::ENOENT => {
                buf.fill(0);
            }
            Err(err) => return Err(err),
        }
        Ok(())
    }

    /// Read a contiguous physical extent run with one block-device request.
    fn read_extent_run(
        &self,
        inode_ref: &InodeRef,
        start_lblock: LBlockId,
        end_lblock: LBlockId,
        start_pblock: PBlockId,
        buf: &mut [u8],
    ) -> Result<usize> {
        let block_count = end_lblock.saturating_sub(start_lblock) as usize;
        let byte_count = block_count
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
        let read_len = buf.len().min(byte_count);
        if read_len == 0 {
            return Ok(0);
        }
        self.ensure_valid_pblock(inode_ref.id, start_pblock, "extent data run")?;
        self.validate_data_blocks(
            start_pblock,
            u64::try_from(block_count).map_err(|_| Ext4Error::new(ErrCode::EFBIG))?,
        )?;
        self.block_device
            .read_blocks(start_pblock, &mut buf[..read_len])?;
        Ok(read_len)
    }

    /// Get file attributes.
    ///
    /// # Params
    ///
    /// * `id` - inode id
    ///
    /// # Return
    ///
    /// A file attribute struct.
    ///
    /// # Error
    ///
    /// `EINVAL` if the inode-table entry is physically free.
    pub fn getattr(&self, id: InodeId) -> Result<FileAttr> {
        let inode = self.read_inode(id)?;
        if inode.inode.mode().bits() == 0 {
            return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
        }

        Ok(Self::file_attr(&inode))
    }

    fn file_attr(inode: &InodeRef) -> FileAttr {
        // Get device number for device nodes
        let rdev = if inode.inode.is_device() {
            inode.inode.device()
        } else {
            (0, 0)
        };

        FileAttr {
            ino: inode.id,
            generation: inode.inode.generation(),
            size: inode.inode.size(),
            blocks: inode.inode.block_count(),
            atime: inode.inode.atime(),
            mtime: inode.inode.mtime(),
            ctime: inode.inode.ctime(),
            crtime: inode.inode.crtime(),
            ftype: inode.inode.file_type(),
            perm: inode.inode.perm(),
            links: inode.inode.link_count(),
            uid: inode.inode.uid(),
            gid: inode.inode.gid(),
            rdev,
        }
    }

    /// Set file attributes.
    ///
    /// # Params
    ///
    /// * `id` - inode id
    /// * `attr` - attributes to set (wrapped in SetAttr struct)
    ///
    /// # Error
    ///
    /// `EINVAL` if the inode is invalid (mode == 0).
    pub fn setattr(&self, id: InodeId, attr: SetAttr) -> Result<()> {
        self.ensure_mutable()?;
        if attr.is_writeback_metadata_only()
            && self.defer_inode_metadata_if_pending(id, |inode| attr.apply_to(&mut inode.inode))?
        {
            return Ok(());
        }
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
        let mut inode = self.read_inode(id)?;
        if inode.inode.mode().bits() == 0 {
            return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
        }
        attr.apply_to(&mut inode.inode);
        self.write_inode_with_csum(&mut inode)?;
        Ok(())
    }

    fn recompute_inode_block_count(&self, inode: &mut InodeRef) -> Result<()> {
        self.prepare_stats.record_block_count_full_traversal();
        let data_blocks = self.extent_all_data_blocks(inode)?.len() as u64;
        let tree_blocks = self.extent_all_tree_blocks(inode)?.len() as u64;
        let sectors_per_block = (BLOCK_SIZE / INODE_BLOCK_SIZE) as u64;
        inode
            .inode
            .set_block_count((data_blocks + tree_blocks) * sectors_per_block);
        Ok(())
    }

    fn update_inode_block_count_after_range(
        &self,
        inode: &mut InodeRef,
        added_data_blocks: usize,
        was_inline_extent_root: bool,
    ) -> Result<()> {
        if was_inline_extent_root && inode.inode.extent_root().header().depth() == 0 {
            let added =
                u64::try_from(added_data_blocks).map_err(|_| Ext4Error::new(ErrCode::EFBIG))?;
            let blocks = inode
                .inode
                .fs_block_count()
                .checked_add(added)
                .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            inode.inode.set_fs_block_count(blocks);
            return Ok(());
        }
        self.recompute_inode_block_count(inode)
    }

    fn ensure_blocks_for_write_range_locked(
        &self,
        inode: &mut InodeRef,
        range: &WriteLogicalRange,
        skip_zero: bool,
    ) -> Result<()> {
        self.prepare_stats.record_call();
        let mut allocation_attempted = false;
        let result = (|| {
            self.prepare_stats.record_requested(range.block_count);
            let mut changed = false;
            let mut added_data_blocks = 0usize;
            let was_inline_extent_root = inode.inode.extent_root().header().depth() == 0;
            let mut missing_lblocks = Vec::new();
            missing_lblocks
                .try_reserve_exact(range.block_count)
                .map_err(|_| Ext4Error::new(ErrCode::ENOMEM))?;
            for iblock in range.first_lblock..=range.last_lblock {
                match self.extent_query(inode, iblock) {
                    Ok(_) => {
                        self.prepare_stats.record_mapped();
                    }
                    Err(err) if err.code() == ErrCode::ENOENT => {
                        self.prepare_stats.record_missing();
                        allocation_attempted = true;
                        missing_lblocks.push(iblock);
                    }
                    Err(err) => return Err(err),
                }
            }
            if !missing_lblocks.is_empty() {
                // Allocate and initialize all missing data before the one-per-
                // group bitmap/GDT/superblock publish.  Extents are installed
                // only after that metadata is durable.
                let allocated =
                    self.alloc_zeroed_data_blocks(inode, missing_lblocks.len(), skip_zero)?;
                let mut allocation_index = 0usize;
                while allocation_index < allocated.len() {
                    let first_lblock = missing_lblocks[allocation_index];
                    let first_pblock = allocated[allocation_index];
                    let mut block_count = 1usize;
                    while allocation_index + block_count < allocated.len()
                        && missing_lblocks[allocation_index + block_count]
                            == first_lblock + block_count as LBlockId
                        && allocated[allocation_index + block_count]
                            == first_pblock + block_count as PBlockId
                        && block_count < u16::MAX as usize
                    {
                        block_count += 1;
                    }
                    let block_count =
                        u32::try_from(block_count).map_err(|_| Ext4Error::new(ErrCode::EFBIG))?;
                    self.extent_query_or_create_preallocated(
                        inode,
                        first_lblock,
                        block_count,
                        first_pblock,
                    )?;
                    for offset in 0..block_count {
                        let iblock = first_lblock
                            .checked_add(offset)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                        let pblock = first_pblock
                            .checked_add(offset as PBlockId)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                        if self.extent_query(inode, iblock)? != pblock {
                            return Err(format_error!(
                                ErrCode::EIO,
                                "extent allocation invariant failed: inode {} iblock {} has unexpected physical block",
                                inode.id,
                                iblock,
                            ));
                        }
                    }
                    added_data_blocks = added_data_blocks
                        .checked_add(block_count as usize)
                        .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                    allocation_index += block_count as usize;
                }
                changed = true;
            }
            if changed {
                self.update_inode_block_count_after_range(
                    inode,
                    added_data_blocks,
                    was_inline_extent_root,
                )?;
                self.write_inode_with_csum(inode)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(()),
            Err(allocation_error) => {
                self.prepare_stats.record_failure();
                if !allocation_attempted {
                    return Err(allocation_error);
                }
                match self
                    .recompute_inode_block_count(inode)
                    .and_then(|()| self.write_inode_with_csum(inode))
                {
                    Ok(()) => Err(allocation_error),
                    Err(recovery_error) => {
                        self.poison(ErrCode::EIO);
                        Err(recovery_error)
                    }
                }
            }
        }
    }

    /// Ensure extents exist for the bytes that will actually be written.
    pub fn allocate_blocks_for_write_range(
        &self,
        id: InodeId,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        self.ensure_mutable()?;
        let Some(range) = Self::checked_write_logical_range(offset, len)? else {
            return Ok(());
        };
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard = self.inode_mutation_locks[self.inode_mutation_lock_index(id)].lock();
        let mut inode = self.read_inode(id)?;
        if inode.inode.mode().bits() == 0 {
            return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
        }
        self.ensure_blocks_for_write_range_locked(&mut inode, &range, false)
    }

    /// Prepare a buffered write by allocating only the written range.
    ///
    /// The caller owns the in-memory visible size used by page-cache writeback
    /// and should call `commit_inode_size()` at fsync/truncate-style sync
    /// boundaries.
    pub fn prepare_buffered_write(
        &self,
        id: InodeId,
        offset: usize,
        len: usize,
        _size: u64,
        _mtime: Option<u32>,
    ) -> Result<()> {
        self.prepare_buffered_write_with_data(id, offset, len, _size, _mtime, None)
            .map(|_| ())
    }

    /// Prepare a buffered write and report whether `real_data` was persisted by
    /// direct-range allocation.
    pub fn prepare_buffered_write_with_data(
        &self,
        id: InodeId,
        offset: usize,
        len: usize,
        _size: u64,
        _mtime: Option<u32>,
        real_data: Option<&[u8]>,
    ) -> Result<bool> {
        self.ensure_mutable()?;
        let Some(range) = Self::checked_write_logical_range(offset, len)? else {
            return Ok(false);
        };
        // A direct data-only write flushes the deferred journal to obtain a
        // stable extent map.  Journaled appends must prepare through their
        // transaction first, otherwise each following write forces a commit.
        if !self.uses_journal() {
            if let Some(data) = real_data.and_then(|data| data.get(..len)) {
                match self.write_data_only(id, offset, data) {
                    Ok(written) if written == len => return Ok(true),
                    Ok(_) => return Err(Ext4Error::new(ErrCode::EIO)),
                    Err(error) if error.code() == ErrCode::ENOENT => {
                        // Has holes — fall through to normal allocation path.
                        // write_data_only() hasn't written anything for ENOENT,
                        // so no partial data to clean up.
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        if self.uses_journal() {
            let outcome = {
                let _metadata_guard = self.lock_transactional_metadata_mutation()?;
                let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
                self.try_prepare_journal_range(id, &range, real_data)?
            };
            match outcome {
                DirectRangePrepare::DataWritten => return Ok(true),
                DirectRangePrepare::Initialized => return Ok(false),
                DirectRangePrepare::Unsupported => {
                    if let Some(data) = real_data.and_then(|data| data.get(..len)) {
                        let outcome = {
                            let _metadata_guard = self.lock_transactional_metadata_mutation()?;
                            let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
                            self.try_prepare_journal_mapped_write(id, &range, offset, data)?
                        };
                        if matches!(outcome, DirectRangePrepare::DataWritten) {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        if self.supports_direct_range_stage() {
            let direct_range_supported = {
                let _metadata_guard = self.lock_direct_metadata_mutation()?;
                let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
                let inode = self.read_inode(id)?;
                if inode.inode.mode().bits() == 0 {
                    return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
                }
                self.direct_range_plan(&inode, &range)?.is_some()
            };
            if !direct_range_supported {
                crate::println!(
                    "[ext4_diag] fallback:no_direct_range lblock={} count={}",
                    range.first_lblock,
                    range.block_count
                );
            }
            if direct_range_supported {
                let outcome = {
                    let _metadata_guard = self.lock_transactional_metadata_mutation()?;
                    let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
                    let mut inode = self.read_inode(id)?;
                    if inode.inode.mode().bits() == 0 {
                        return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
                    }
                    self.try_prepare_direct_range(&mut inode, &range, real_data)?
                };
                match outcome {
                    DirectRangePrepare::DataWritten => return Ok(true),
                    DirectRangePrepare::Initialized => return Ok(false),
                    DirectRangePrepare::Unsupported => {}
                }
            }
        }
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
        loop {
            let probe = self.prepared_extent_probe(id, &range);
            if Self::prepared_extent_hit(probe) {
                return Ok(false);
            }
            let mut inode = self.read_inode(id)?;
            if inode.inode.mode().bits() == 0 {
                return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
            }
            self.ensure_blocks_for_write_range_locked(&mut inode, &range, false)?;
            // The mutation guard serializes same-inode mutation, while the
            // token rejects any mapping commit that raced elsewhere. Never
            // publish an I/O-derived extent under a newer cache epoch.
            if self.publish_prepared_extent(id, &range, Self::prepared_extent_token(probe)) {
                return Ok(false);
            }
        }
    }

    /// Commit cached inode metadata to disk without allocating data blocks.
    pub fn commit_inode_metadata(
        &self,
        id: InodeId,
        size: Option<u64>,
        atime: Option<u32>,
        mtime: Option<u32>,
    ) -> Result<()> {
        self.setattr(
            id,
            SetAttr {
                size,
                atime,
                mtime,
                ..SetAttr::default()
            },
        )
    }

    /// Extends an active writeback journal batch with one inode image.
    ///
    /// The transactional gate excludes direct writers while the pending batch
    /// is taken and restaged.  Callers fall back to their direct path when no
    /// deferred batch exists, preserving legacy metadata semantics.
    fn defer_inode_metadata_if_pending<F>(&self, id: InodeId, update: F) -> Result<bool>
    where
        F: FnOnce(&mut InodeRef),
    {
        if !self.uses_journal() {
            return Ok(false);
        }
        let _metadata_guard = self.lock_transactional_metadata_mutation()?;
        let Some(mut transaction) = self.transaction_start_if_deferred(1)? else {
            return Ok(false);
        };
        let _mutation_guard = self.lock_inode_mutation_for_prepare(id);
        let mut inode = self.transaction_read_inode(&transaction, id)?;
        if inode.inode.mode().bits() == 0 {
            return_error!(ErrCode::EINVAL, "Invalid inode {}", id);
        }
        update(&mut inode);
        self.transaction_stage_inode_with_csum(&mut transaction, &mut inode)?;
        if let Err(error) = transaction.defer_or_commit(
            self.block_device.as_ref(),
            self,
            MAX_DEFERRED_JOURNAL_BLOCKS,
        ) {
            if error.failure != super::journal_transaction::CommitFailure::BeforeCommit {
                self.poison(ErrCode::EIO);
            }
            return Err(error.error);
        }
        Ok(true)
    }

    /// Commit the file size (`i_size`) and optionally `mtime` to disk,
    /// **without** allocating any blocks.
    ///
    /// Call this after successful page-cache write to finalise the new file size.
    pub fn commit_inode_size(&self, id: InodeId, size: u64, mtime: Option<u32>) -> Result<()> {
        self.commit_inode_metadata(id, Some(size), None, mtime)
    }

    /// Resize an inode and release every extent wholly or partially beyond EOF.
    ///
    /// Growing only changes `i_size`; shrinking removes extent tails in
    /// block-group-local transactions before publishing the new size.
    pub fn truncate_inode(&self, inode_id: InodeId, new_size: u64) -> Result<()> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_transactional_metadata_mutation()?;
        let _mutation_guard = self.lock_inode_mutation_for_prepare(inode_id);
        self.flush_deferred_journal()?;

        let inode = self.read_inode(inode_id)?;
        if inode.inode.mode().bits() == 0 {
            return_error!(ErrCode::EINVAL, "Invalid inode {}", inode_id);
        }
        let old_size = inode.inode.size();
        if old_size == new_size {
            return Ok(());
        }

        let keep_blocks = new_size.div_ceil(BLOCK_SIZE as u64);
        let mut tail_zeroed = false;
        loop {
            let mut transaction = self.transaction_start(32)?;
            let mut inode = self.transaction_read_inode(&transaction, inode_id)?;
            let tail = if new_size < old_size && inode.inode.uses_extents() {
                self.extent_tail(&transaction, &inode)?
            } else {
                None
            };

            let tail_offset = new_size % BLOCK_SIZE as u64;
            if !tail_zeroed && new_size < old_size && tail_offset != 0 {
                let last_lblock: LBlockId = (new_size / BLOCK_SIZE as u64)
                    .try_into()
                    .map_err(|_| Ext4Error::new(ErrCode::EFBIG))?;
                match self.extent_query(&inode, last_lblock) {
                    Ok(last_pblock) => {
                        let mut block = self.read_block(last_pblock)?;
                        block.data[tail_offset as usize..].fill(0);
                        self.write_block(&block)?;
                    }
                    // A sparse final block already reads as zero and must not be allocated.
                    Err(error) if error.code() == ErrCode::ENOENT => {}
                    Err(error) => return Err(error),
                }
                tail_zeroed = true;
            }

            let mut finished = true;
            if let Some(tail) = tail {
                let tail_end = tail
                    .start_lblock
                    .checked_add(tail.block_count)
                    .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
                if u64::from(tail_end) > keep_blocks {
                    let tail_end_pblock = tail
                        .start_pblock
                        .checked_add(tail.block_count as PBlockId)
                        .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
                    let super_block = self.read_super_block_cached();
                    let first_data_block = super_block.first_data_block() as PBlockId;
                    if tail.start_pblock < first_data_block
                        || tail_end_pblock > super_block.block_count()
                        || self.journal_owns_block_range(tail.start_pblock, tail_end_pblock)
                    {
                        transaction.abort();
                        return_error!(ErrCode::EIO, "Invalid extent tail for inode {}", inode_id);
                    }
                    let blocks_per_group = super_block.blocks_per_group() as PBlockId;
                    let group_remaining =
                        (tail_end_pblock - 1 - first_data_block) % blocks_per_group + 1;
                    let beyond_eof = u64::from(tail_end)
                        - core::cmp::max(keep_blocks, u64::from(tail.start_lblock));
                    let remove_limit = u32::try_from(core::cmp::min(beyond_eof, group_remaining))
                        .map_err(|_| Ext4Error::new(ErrCode::EFBIG))?;
                    let removed = self
                        .extent_remove_tail_in_transaction(
                            &mut transaction,
                            &mut inode,
                            remove_limit,
                        )?
                        .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
                    self.transaction_dealloc_block_range(
                        &mut transaction,
                        removed.start_pblock,
                        removed.block_count,
                    )?;
                    for metadata in removed.metadata_blocks.iter().copied() {
                        self.transaction_dealloc_block_range(&mut transaction, metadata, 1)?;
                    }
                    let released =
                        removed.block_count as u64 + removed.metadata_blocks.len() as u64;
                    inode.inode.set_fs_block_count(
                        inode
                            .inode
                            .fs_block_count()
                            .checked_sub(released)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?,
                    );
                    finished = false;
                }
            }

            if finished {
                inode.inode.set_size(new_size);
            }
            self.transaction_stage_inode_with_csum(&mut transaction, &mut inode)?;
            if let Err(error) = transaction.commit(self.block_device.as_ref(), self) {
                self.poison(ErrCode::EIO);
                return Err(error.error);
            }
            if finished {
                return Ok(());
            }
        }
    }

    /// Link a newly created inode into `parent`.
    ///
    /// If linking fails, this function frees the newly allocated inode to avoid leaks.
    fn link_new_inode_or_free(
        &self,
        parent: &mut InodeRef,
        child: &mut InodeRef,
        name: &str,
    ) -> Result<()> {
        // Namespace writers hold `namespace_lock`, so this check and the
        // following insertion are one atomic duplicate-name transaction.
        match self.dir_find_entry(parent, name) {
            Err(error) if error.code() == ErrCode::ENOENT => {}
            Ok(_) => {
                if let Err(cleanup_err) = self.free_inode(child) {
                    trace!(
                        "duplicate entry for new inode {} (name {}), cleanup failed: {:?}",
                        child.id,
                        name,
                        cleanup_err
                    );
                    return Err(cleanup_err);
                }
                return_error!(ErrCode::EEXIST, "Entry '{}' already exists", name);
            }
            Err(lookup_err) => {
                if let Err(cleanup_err) = self.free_inode(child) {
                    trace!(
                        "entry lookup failed for new inode {} (name {}), cleanup failed: {:?}; original lookup error: {:?}",
                        child.id,
                        name,
                        cleanup_err,
                        lookup_err
                    );
                    return Err(cleanup_err);
                }
                return Err(lookup_err);
            }
        }
        if let Err(link_err) = self.link_inode(parent, child, name, false) {
            if let Err(cleanup_err) = self.free_inode(child) {
                trace!(
                    "link failed for new inode {} (name {}), cleanup failed: {:?}; original link error: {:?}",
                    child.id,
                    name,
                    cleanup_err,
                    link_err
                );
                return Err(cleanup_err);
            }
            return Err(link_err);
        }
        Ok(())
    }

    /// Create a file. This function will not check the existence of
    /// the file, call `lookup` to check beforehand.
    ///
    /// # Params
    ///
    /// * `parent` - parent directory inode id
    /// * `name` - file name
    /// * `mode` - file type and mode with which to create the new file
    /// * `flags` - open flags
    ///
    /// # Return
    ///
    /// `Ok(inode)` - Inode id of the new file
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOSPC` - No space left on device
    pub fn create(&self, parent: InodeId, name: &str, mode: InodeMode) -> Result<InodeId> {
        self.create_with_owner(parent, name, mode, InodeOwner { uid: 0, gid: 0 })
    }

    pub fn create_with_owner(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        owner: InodeOwner,
    ) -> Result<InodeId> {
        Ok(self
            .create_with_owner_and_attr(parent, name, mode, owner)?
            .ino)
    }

    /// Create a symbolic link whose target is fully initialized before its name
    /// is published in the parent directory.
    pub fn symlink_with_owner_and_attr(
        &self,
        parent: InodeId,
        name: &str,
        target: &[u8],
        owner: InodeOwner,
    ) -> Result<FileAttr> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        let _mutation_guards = self.lock_inode_mutations(&[parent]);
        let mut parent = self.read_inode(parent)?;
        if !parent.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", parent.id);
        }
        let mut child = self.create_symlink_inode_with_owner(target, owner)?;
        self.link_new_inode_or_free(&mut parent, &mut child, name)?;
        Ok(Self::file_attr(&child))
    }

    /// Create and link a file, returning the attributes from the authoritative
    /// in-memory inode used by the namespace transaction.
    pub fn create_with_owner_and_attr(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        owner: InodeOwner,
    ) -> Result<FileAttr> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        let _mutation_guards = self.lock_inode_mutations(&[parent]);
        let mut parent = self.read_inode(parent)?;
        // Can only create a file in a directory
        if !parent.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", parent.id);
        }
        // Create child inode and link it to parent directory
        let mut child = self.create_inode_with_owner(mode, owner.uid, owner.gid)?;
        self.link_new_inode_or_free(&mut parent, &mut child, name)?;
        Ok(Self::file_attr(&child))
    }

    /// Create a device node (character or block device).
    ///
    /// Unlike `create()`, this function:
    /// - Does NOT initialize the extent tree
    /// - Stores the device number in i_block[0..1] (Linux ext4 standard)
    ///
    /// # Params
    ///
    /// * `parent` - parent directory inode id
    /// * `name` - device node name
    /// * `mode` - file type (must include CHARDEV or BLOCKDEV) and permissions
    /// * `major` - major device number
    /// * `minor` - minor device number
    ///
    /// # Return
    ///
    /// `Ok(inode)` - Inode id of the new device node
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOSPC` - No space left on device
    pub fn mknod(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        major: u32,
        minor: u32,
    ) -> Result<InodeId> {
        self.mknod_with_owner(
            parent,
            name,
            mode,
            major,
            minor,
            InodeOwner { uid: 0, gid: 0 },
        )
    }

    pub fn mknod_with_owner(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        major: u32,
        minor: u32,
        owner: InodeOwner,
    ) -> Result<InodeId> {
        Ok(self
            .mknod_with_owner_and_attr(parent, name, mode, major, minor, owner)?
            .ino)
    }

    /// Create and link a device node, returning the attributes from the
    /// authoritative in-memory inode used by the namespace transaction.
    pub fn mknod_with_owner_and_attr(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        major: u32,
        minor: u32,
        owner: InodeOwner,
    ) -> Result<FileAttr> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        let _mutation_guards = self.lock_inode_mutations(&[parent]);
        let mut parent_ref = self.read_inode(parent)?;

        // Can only create in a directory
        if !parent_ref.inode.is_dir() {
            return_error!(
                ErrCode::ENOTDIR,
                "Inode {} is not a directory",
                parent_ref.id
            );
        }

        // Create device inode (uses create_device_inode which sets device number)
        let mut child = self.create_device_inode(mode, major, minor, owner.uid, owner.gid)?;

        // Link to parent directory
        self.link_new_inode_or_free(&mut parent_ref, &mut child, name)?;

        trace!("mknod {} ({}:{}) -> inode {}", name, major, minor, child.id);
        Ok(Self::file_attr(&child))
    }

    /// Read data from a file. This function will read exactly `buf.len()`
    /// bytes unless the end of the file is reached.
    ///
    /// # Params
    ///
    /// * `file` - the file handler, acquired by `open` or `create`
    /// * `offset` - offset to read from
    /// * `buf` - the buffer to store the data
    ///
    /// # Return
    ///
    /// `Ok(usize)` - the actual number of bytes read
    ///
    /// # Error
    ///
    /// * `EISDIR` - `file` is not a regular file
    pub fn read(&self, file: InodeId, offset: usize, buf: &mut [u8]) -> Result<usize> {
        // Get the inode of the file
        let file = self.read_inode(file)?;
        if !file.inode.is_file() {
            return_error!(ErrCode::EISDIR, "Inode {} is not a file", file.id);
        }

        // Read no bytes
        if buf.is_empty() {
            return Ok(0);
        }
        let file_size = file.inode.size() as usize;
        if offset >= file_size {
            return Ok(0);
        }
        // Calc the actual size to read
        let read_size = min(buf.len(), file_size - offset);
        // Calc the start block of reading
        let start_iblock = (offset / BLOCK_SIZE) as LBlockId;
        // Calc the length that is not aligned to the block size
        let misaligned = offset % BLOCK_SIZE;

        let mut cursor = 0;
        let mut iblock = start_iblock;
        let mut read_extent_range: Option<super::data_write::ExtentWriteRange> = None;
        // Read first block
        if misaligned > 0 {
            let read_len = min(BLOCK_SIZE - misaligned, read_size);
            self.read_extent_or_hole(
                &file,
                start_iblock,
                misaligned,
                &mut buf[cursor..cursor + read_len],
            )?;
            cursor += read_len;
            iblock += 1;
        }
        // Continue with full block reads
        while cursor < read_size {
            let remaining = read_size - cursor;
            if remaining >= BLOCK_SIZE {
                let resolved = match read_extent_range {
                    Some(ext) if iblock >= ext.start_lblock && iblock < ext.end_lblock => {
                        let pblock = ext
                            .start_pblock
                            .checked_add((iblock - ext.start_lblock) as PBlockId)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
                        Ok((pblock, ext))
                    }
                    _ => self
                        .extent_query_with_range(&file, iblock)
                        .map(|(pblock, ext)| {
                            read_extent_range = Some(ext);
                            (pblock, ext)
                        }),
                };
                match resolved {
                    Ok((pblock, ext)) => {
                        let block_count =
                            ((remaining / BLOCK_SIZE) as LBlockId).min(ext.end_lblock - iblock);
                        let end_iblock = iblock
                            .checked_add(block_count)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                        let read_len = (block_count as usize)
                            .checked_mul(BLOCK_SIZE)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                        let read = self.read_extent_run(
                            &file,
                            iblock,
                            end_iblock,
                            pblock,
                            &mut buf[cursor..cursor + read_len],
                        )?;
                        cursor += read;
                        iblock = end_iblock;
                        continue;
                    }
                    Err(error) if error.code() == ErrCode::ENOENT => {
                        buf[cursor..cursor + BLOCK_SIZE].fill(0);
                        cursor += BLOCK_SIZE;
                        iblock = iblock
                            .checked_add(1)
                            .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }

            self.read_extent_or_hole(&file, iblock, 0, &mut buf[cursor..read_size])?;
            cursor = read_size;
        }

        Ok(cursor)
    }

    /// Read the target path of a symbolic link (i.e. readlink(2) semantics).
    ///
    /// - Returns the raw byte sequence of the link content (not required to end with '\0')
    /// - For fast symlink (length <= 60), content is stored in inode.i_block (here inode.block[60])
    /// - For non-fast symlink, content is stored in data blocks, reusing extent read path
    pub fn readlink(&self, inode_id: InodeId, offset: usize, buf: &mut [u8]) -> Result<usize> {
        let inode_ref = self.read_inode(inode_id)?;
        if !inode_ref.inode.is_softlink() {
            return_error!(ErrCode::EINVAL, "Inode {} is not a symlink", inode_id);
        }
        if buf.is_empty() {
            return Ok(0);
        }

        let size = inode_ref.inode.size() as usize;
        if offset >= size {
            return Ok(0);
        }

        // fast symlink: content stored inline in inode.i_block
        let inline = inode_ref.inode.inline_block();
        if size <= inline.len() && inode_ref.inode.fs_block_count() == 0 {
            let n = core::cmp::min(buf.len(), size - offset);
            buf[..n].copy_from_slice(&inline[offset..offset + n]);
            return Ok(n);
        }

        // non-fast symlink: stored in data blocks, reuse extent-based read logic
        let read_size = min(buf.len(), size - offset);
        let start_iblock = (offset / BLOCK_SIZE) as LBlockId;
        let misaligned = offset % BLOCK_SIZE;

        let mut cursor = 0;
        let mut iblock = start_iblock;
        if misaligned > 0 {
            let read_len = min(BLOCK_SIZE - misaligned, read_size);
            self.read_extent_or_hole(
                &inode_ref,
                start_iblock,
                misaligned,
                &mut buf[cursor..cursor + read_len],
            )?;
            cursor += read_len;
            iblock += 1;
        }
        while cursor < read_size {
            let read_len = min(BLOCK_SIZE, read_size - cursor);
            self.read_extent_or_hole(&inode_ref, iblock, 0, &mut buf[cursor..cursor + read_len])?;
            cursor += read_len;
            iblock += 1;
        }

        Ok(cursor)
    }

    /// Write data to a file. This function will write exactly `data.len()` bytes.
    ///
    /// # Params
    ///
    /// * `file` - the file handler, acquired by `open` or `create`
    /// * `offset` - offset to write to
    /// * `data` - the data to write
    ///
    /// # Return
    ///
    /// `Ok(usize)` - the actual number of bytes written
    ///
    /// # Error
    ///
    /// * `EISDIR` - `file` is not a regular file
    /// * `ENOSPC` - no space left on device
    pub fn write(&self, file: InodeId, offset: usize, data: &[u8]) -> Result<usize> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let write_size = data.len();
        if write_size == 0 {
            return Ok(0);
        }
        let range = match Self::checked_write_logical_range(offset, write_size)? {
            Some(range) => range,
            None => return Ok(0),
        };
        // Get the inode of the file
        let _mutation_guard =
            self.inode_mutation_locks[self.inode_mutation_lock_index(file)].lock();
        let mut file = self.read_inode(file)?;
        if !file.inode.is_file() {
            return_error!(ErrCode::EISDIR, "Inode {} is not a file", file.id);
        }

        self.ensure_blocks_for_write_range_locked(&mut file, &range, false)?;

        // Write data
        let mut cursor = 0;
        let mut iblock = range.first_lblock;
        while cursor < write_size {
            let block_offset = (offset + cursor) % BLOCK_SIZE;
            let write_len = min(BLOCK_SIZE - block_offset, write_size - cursor);
            let fblock = self.extent_query(&file, iblock)?;
            let mut block = self.read_block(fblock)?;
            block.write_offset(block_offset, &data[cursor..cursor + write_len]);
            self.write_block(&block)?;
            cursor += write_len;
            if cursor < write_size {
                iblock = iblock
                    .checked_add(1)
                    .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            }
        }
        let new_end = offset.checked_add(cursor).ok_or(format_error!(
            ErrCode::EFBIG,
            "write end overflow: offset={} len={}",
            offset,
            cursor
        ))?;
        if new_end > file.inode.size() as usize {
            file.inode.set_size(new_end as u64);
        }
        self.write_inode_with_csum(&mut file)?;

        Ok(cursor)
    }

    /// Create a hard link. This function will not check name conflict,
    /// call `lookup` to check beforehand.
    ///
    /// # Params
    ///
    /// * `child` - the inode of the file to link
    /// * `parent` - the inode of the directory to link to
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOSPC` - no space left on device
    pub fn link(&self, child: InodeId, parent: InodeId, name: &str) -> Result<()> {
        self.ensure_mutable()?;
        let _namespace_guard = self.namespace_lock.lock();
        let _mutation_guards = self.lock_inode_mutations(&[parent, child]);
        let mut parent = self.read_inode(parent)?;
        // Can only link to a directory
        if !parent.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", parent.id);
        }
        let mut child = self.read_inode(child)?;
        // Cannot link a directory
        if child.inode.is_dir() {
            return_error!(ErrCode::EISDIR, "Cannot link a directory");
        }
        // `namespace_lock` makes this check atomic with `link_inode`'s
        // directory insertion below.
        match self.dir_find_entry(&parent, name) {
            Ok(_) => {
                return_error!(ErrCode::EEXIST, "Entry already exists");
            }
            Err(e) if e.code() == ErrCode::ENOENT => {}
            Err(e) => return Err(e),
        }
        if !self.dir_has_insert_space(&parent, &child, name)? {
            // Directory growth is still a direct allocation operation. Do it
            // before entering the journal domain; an empty initialized slot is
            // harmless if the later link is not made durable.
            let _metadata_guard = self.lock_direct_metadata_mutation()?;
            self.prepare_empty_dir_slot(&mut parent)?;
        }
        let _metadata_guard = self.lock_transactional_metadata_mutation()?;
        self.link_inode_transactional(&mut parent, &mut child, name, true)?;
        Ok(())
    }

    /// Unlink a file.
    ///
    /// # Params
    ///
    /// * `parent` - the inode of the directory to unlink from
    /// * `name` - the name of the file to unlink
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOENT` - `name` does not exist in `parent`
    /// * `EISDIR` - `parent/name` is a directory
    pub fn unlink(&self, parent: InodeId, name: &str) -> Result<Option<InodeReclaimHandle>> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_transactional_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        let mut parent_ref = self.read_inode(parent)?;
        // Can only unlink from a directory
        if !parent_ref.inode.is_dir() {
            return_error!(
                ErrCode::ENOTDIR,
                "Inode {} is not a directory",
                parent_ref.id
            );
        }
        // Cannot unlink directory
        let location = self.dir_find_entry_location(&parent_ref, name)?;
        let child_id = location.inode_id;
        let _mutation_guards = self.lock_inode_mutations(&[parent, child_id]);
        parent_ref = self.read_inode(parent)?;
        let mut child = self.read_inode(child_id)?;
        if child.inode.is_dir() {
            return_error!(ErrCode::EISDIR, "Cannot unlink a directory");
        }
        self.unlink_inode(&mut parent_ref, &mut child, name, location)
    }

    /// Helper: Read and validate parent directories for rename operations.
    ///
    /// Returns (parent_ref, Option<new_parent_ref>). If parent == new_parent,
    /// the second element is None to avoid double-locking the same inode.
    fn read_rename_dirs(
        &self,
        parent: InodeId,
        new_parent: InodeId,
    ) -> Result<(InodeRef, Option<InodeRef>)> {
        let parent_ref = self.read_inode(parent)?;
        if !parent_ref.inode.is_dir() {
            return_error!(
                ErrCode::ENOTDIR,
                "Inode {} is not a directory",
                parent_ref.id
            );
        }

        let new_parent_ref = if parent == new_parent {
            None
        } else {
            let np = self.read_inode(new_parent)?;
            if !np.inode.is_dir() {
                return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", np.id);
            }
            Some(np)
        };

        Ok((parent_ref, new_parent_ref))
    }

    /// Helper: Check if `target_dir` is a descendant of `dir_inode`.
    ///
    /// Used to prevent directory cycles in rename operations.
    /// Returns EINVAL if moving a directory into its own subdirectory.
    fn check_ancestor_cycle(&self, dir_inode: InodeId, target_dir: InodeId) -> Result<()> {
        let mut cur = target_dir;
        loop {
            if cur == dir_inode {
                return_error!(
                    ErrCode::EINVAL,
                    "Cannot move directory into its own subdirectory"
                );
            }
            if cur == EXT4_ROOT_INO {
                break;
            }
            let cur_inode = self.read_inode(cur)?;
            match self.dir_find_entry(&cur_inode, "..") {
                Ok(parent_id) if parent_id != cur => cur = parent_id,
                _ => break,
            }
        }
        Ok(())
    }

    /// Rename a directory entry, with POSIX-compliant atomic replace semantics.
    ///
    /// # POSIX Semantics
    /// - If `new_name` doesn't exist: simple rename
    /// - If `new_name` exists and is the same inode as source: no-op, return Ok
    /// - If `new_name` exists and is different inode: **atomically replace** it
    /// - Directory can only replace empty directory
    /// - Type compatibility: file<->file, dir<->dir (no cross-type replace)
    ///
    /// # Params
    ///
    /// * `parent` - the inode of the source directory
    /// * `name` - the name of the file to move
    /// * `new_parent` - the inode of the directory to move to
    /// * `new_name` - the new name of the file
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` or `new_parent` is not a directory, or dir replacing non-dir
    /// * `ENOENT` - `name` does not exist in `parent`
    /// * `EISDIR` - non-dir replacing dir
    /// * `ENOTEMPTY` - target directory is not empty
    /// * `EINVAL` - would create a directory cycle (moving dir into its own subdirectory)
    /// * `ENOSPC` - no space left on device
    pub fn rename(
        &self,
        parent: InodeId,
        name: &str,
        new_parent: InodeId,
        new_name: &str,
    ) -> Result<Option<InodeReclaimHandle>> {
        self.ensure_mutable()?;
        let _namespace_guard = self.namespace_lock.lock();
        let mut reclaim = None;
        // 1. 验证父目录
        let (mut parent_ref, mut new_parent_ref) = self.read_rename_dirs(parent, new_parent)?;

        // 2. 查找源 inode
        let child_id = self.dir_find_entry(&parent_ref, name)?;
        let mut child = self.read_inode(child_id)?;
        let child_is_dir = child.inode.is_dir();

        // 3. 循环检测：防止把目录移到自己的子目录下
        if child_is_dir && parent != new_parent {
            self.check_ancestor_cycle(child_id, new_parent)?;
        }

        // 4. 检查目标是否存在
        let target_dir_ref = new_parent_ref.as_ref().unwrap_or(&parent_ref);
        let existing = self.dir_find_entry(target_dir_ref, new_name).ok();
        let mut mutation_ids = vec![parent, new_parent, child_id];
        if let Some(existing_id) = existing {
            mutation_ids.push(existing_id);
        }
        // A rename which replaces an existing entry is fully transactional;
        // a simple rename uses legacy direct directory mutation helpers which
        // must own the direct side of the metadata gate.  Selecting the gate
        // after checking the target avoids re-entering the direct side while
        // holding a transactional guard (which would surface as EAGAIN when
        // a destination directory needs to grow).
        let _metadata_guard = if existing.is_some() {
            self.lock_transactional_metadata_mutation()?
        } else {
            self.lock_direct_metadata_mutation()?
        };
        let _mutation_guards = self.lock_inode_mutations(&mutation_ids);
        parent_ref = self.read_inode(parent)?;
        new_parent_ref = if parent == new_parent {
            None
        } else {
            Some(self.read_inode(new_parent)?)
        };
        child = self.read_inode(child_id)?;
        let child_file_type = child.inode.file_type();

        match existing {
            Some(existing_id) if existing_id == child_id => {
                // 情况 A：源和目标是同一个 inode（硬链接或同名）
                // POSIX 语义：无操作，返回成功
                return Ok(None);
            }
            Some(existing_id) => {
                // 情况 B：目标存在且是不同 inode → 原子替换
                let mut existing_inode = self.read_inode(existing_id)?;
                let existing_is_dir = existing_inode.inode.is_dir();

                // 4b-1. 类型兼容性检查
                match (child_is_dir, existing_is_dir) {
                    (true, false) => {
                        return_error!(
                            ErrCode::ENOTDIR,
                            "Cannot replace non-directory with directory"
                        );
                    }
                    (false, true) => {
                        return_error!(
                            ErrCode::EISDIR,
                            "Cannot replace directory with non-directory"
                        );
                    }
                    (true, true) => {
                        // 目录替换目录：目标必须为空
                        if !self.dir_is_empty(&existing_inode)? {
                            return_error!(ErrCode::ENOTEMPTY, "Target directory is not empty");
                        }
                    }
                    (false, false) => {
                        // 文件替换文件：OK
                    }
                }

                let existing_link_cnt = existing_inode.inode.link_count();
                // An empty replaced directory loses its only parent entry and
                // Linux clear_nlink()s it even when an old/corrupt on-disk
                // count is unexpectedly greater than two.  Regular files can
                // still have independent hard links.
                let final_target =
                    super::link::namespace_removal_is_final(existing_is_dir, existing_link_cnt);

                // Upper bound of distinct home blocks in the replace set:
                // destination dirent + source dirent + optional child "..";
                // overwritten inode + each logically changed parent inode;
                // and the superblock only for a final target.  The transaction
                // map deduplicates entries which share a directory or inode-
                // table block, so same-parent and same-block cases consume
                // fewer credits without weakening the reservation bound.
                let mut credits = 3; // two dirent blocks + overwritten inode
                if child_is_dir && parent != new_parent {
                    credits += 3; // child ".." + old parent + new parent
                }
                if existing_is_dir && !(child_is_dir && parent != new_parent) {
                    credits += 1; // target parent (new parent already counted above)
                }
                if final_target && self.uses_journal() {
                    credits += 1; // superblock orphan head
                }
                let mut transaction = self.transaction_start_with_deferred_retry(credits)?;

                // Match Linux ext4_rename(): ext4_setent(new), delete(old),
                // ext4_rename_dir_finish(), parent counts, target nlink, and
                // ext4_orphan_add() all belong to this single handle.
                {
                    let target_dir = new_parent_ref.as_mut().unwrap_or(&mut parent_ref);
                    self.transaction_dir_replace_entry(
                        &mut transaction,
                        target_dir,
                        new_name,
                        child_id,
                        child_file_type,
                    )?;

                    if existing_is_dir {
                        target_dir
                            .inode
                            .set_link_count(target_dir.inode.link_count() - 1);
                        self.transaction_stage_inode_with_csum(&mut transaction, target_dir)?;
                    }
                }

                self.transaction_dir_remove_entry(&mut transaction, &parent_ref, name)?;

                if child_is_dir && parent != new_parent {
                    self.transaction_dir_replace_entry(
                        &mut transaction,
                        &child,
                        "..",
                        new_parent,
                        FileType::Directory,
                    )?;

                    parent_ref
                        .inode
                        .set_link_count(parent_ref.inode.link_count() - 1);
                    self.transaction_stage_inode_with_csum(&mut transaction, &mut parent_ref)?;

                    let new_parent_dir = new_parent_ref.as_mut().ok_or(format_error!(
                        ErrCode::EINVAL,
                        "rename: missing new parent reference for directory move"
                    ))?;
                    new_parent_dir
                        .inode
                        .set_link_count(new_parent_dir.inode.link_count() + 1);
                    self.transaction_stage_inode_with_csum(&mut transaction, new_parent_dir)?;
                }

                if final_target {
                    existing_inode.inode.set_link_count(0);
                    if self.uses_journal() {
                        let mut sb = self.read_super_block_cached();
                        self.transaction_orphan_add(
                            &mut transaction,
                            &mut existing_inode,
                            &mut sb,
                        )?;
                    } else {
                        existing_inode.inode.set_next_orphan(0);
                        self.transaction_stage_inode_with_csum(
                            &mut transaction,
                            &mut existing_inode,
                        )?;
                    }
                } else {
                    existing_inode.inode.set_link_count(existing_link_cnt - 1);
                    self.transaction_stage_inode_with_csum(&mut transaction, &mut existing_inode)?;
                }

                let result = if child_is_dir || existing_is_dir {
                    // Directory parent-link updates retain their synchronous
                    // crash-consistency test matrix.
                    transaction.commit(self.block_device.as_ref(), self)
                } else {
                    transaction.defer_or_commit(
                        self.block_device.as_ref(),
                        self,
                        super::link::MAX_DEFERRED_NAMESPACE_BLOCKS,
                    )
                };
                if let Err(error) = result {
                    // Once commit processing starts, failures can leave an
                    // uncertain committed/checkpointed state.  Fail-stop every
                    // subsequent metadata writer on this mount.
                    self.poison(ErrCode::EIO);
                    return Err(error.error);
                }
                if final_target {
                    reclaim = Some(InodeReclaimHandle::new(
                        existing_inode.id,
                        existing_inode.inode.generation(),
                    ));
                }
                // 文件的 link count 不变（只是换了名字/位置）
            }
            None => {
                // 情况 C：目标不存在 → 简单重命名
                // Without a journal, any failure after the first namespace
                // write fail-stops this mount so a partial rename cannot be
                // followed by further allocation or metadata mutation.

                // C-1. 在目标父目录添加新条目（先 add）
                let target_dir = new_parent_ref.as_mut().unwrap_or(&mut parent_ref);
                match self.dir_add_entry_classified(target_dir, &child, new_name) {
                    Ok(()) => {}
                    Err(super::dir::DirAddFailure::Unmodified(error)) => return Err(error),
                    Err(super::dir::DirAddFailure::Indeterminate(error)) => {
                        self.poison(ErrCode::EIO);
                        return Err(error);
                    }
                }

                // C-2. 从源父目录删除旧条目（后 delete）
                self.poison_on_error(self.dir_remove_entry(&parent_ref, name))?;

                // C-3. 目录跨目录移动时，原子更新 ".." 并调整 link count
                if child_is_dir && parent != new_parent {
                    // ".." 原地替换：旧父 → 新父，单次 I/O，无中间态
                    self.poison_on_error(self.dir_replace_entry(
                        &child,
                        "..",
                        new_parent,
                        FileType::Directory,
                    ))?;

                    // 源父目录失去 ".." 引用
                    parent_ref
                        .inode
                        .set_link_count(parent_ref.inode.link_count() - 1);
                    self.poison_on_error(self.write_inode_with_csum(&mut parent_ref))?;

                    // 目标父目录获得 ".." 引用
                    let new_parent_dir = new_parent_ref.as_mut().ok_or(format_error!(
                        ErrCode::EINVAL,
                        "rename: missing new parent reference for directory move"
                    ))?;
                    new_parent_dir
                        .inode
                        .set_link_count(new_parent_dir.inode.link_count() + 1);
                    self.poison_on_error(self.write_inode_with_csum(new_parent_dir))?;
                }
                // 文件：无 ".."，nlink 不变（只换了名字/位置）
                // 目录同目录：".." 已指向正确的父，link count 不变
            }
        }

        Ok(reclaim)
    }

    /// Atomically exchange two directory entries (RENAME_EXCHANGE semantics).
    ///
    /// Both entries must exist. The operation swaps their inode references
    /// in place using `dir_replace_entry`, so directory entries never "disappear".
    ///
    /// # Params
    ///
    /// * `parent` - inode of the directory containing `name`
    /// * `name` - name of the first entry
    /// * `new_parent` - inode of the directory containing `new_name`
    /// * `new_name` - name of the second entry
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` or `new_parent` is not a directory
    /// * `ENOENT` - `name` or `new_name` does not exist
    /// * `EINVAL` - would create a directory cycle
    pub fn rename_exchange(
        &self,
        parent: InodeId,
        name: &str,
        new_parent: InodeId,
        new_name: &str,
    ) -> Result<()> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        // 1. 验证父目录
        let (mut parent_ref, mut new_parent_ref) = self.read_rename_dirs(parent, new_parent)?;

        // 2. 查找两个 inode
        let old_id = self.dir_find_entry(&parent_ref, name)?;
        let target_dir_ref = new_parent_ref.as_ref().unwrap_or(&parent_ref);
        let new_id = self.dir_find_entry(target_dir_ref, new_name)?;
        let _mutation_guards = self.lock_inode_mutations(&[parent, new_parent, old_id, new_id]);
        parent_ref = self.read_inode(parent)?;
        new_parent_ref = if parent == new_parent {
            None
        } else {
            Some(self.read_inode(new_parent)?)
        };
        let old_inode = self.read_inode(old_id)?;
        let old_is_dir = old_inode.inode.is_dir();
        let old_type = old_inode.inode.file_type();
        let new_inode = self.read_inode(new_id)?;
        let new_is_dir = new_inode.inode.is_dir();
        let new_type = new_inode.inode.file_type();

        // 3. 同一 inode → 无操作
        if old_id == new_id {
            return Ok(());
        }

        // 4. 循环检测（仅跨目录时需要，exchange 需要检查双向）
        if parent != new_parent {
            if old_is_dir {
                self.check_ancestor_cycle(old_id, new_parent)?;
            }
            if new_is_dir {
                self.check_ancestor_cycle(new_id, parent)?;
            }
        }

        // 5. 原子交换：原地替换目录项的 inode 引用
        if parent == new_parent {
            self.poison_on_error(self.dir_replace_entry(&parent_ref, name, new_id, new_type))?;
            self.poison_on_error(self.dir_replace_entry(&parent_ref, new_name, old_id, old_type))?;
        } else {
            self.poison_on_error(self.dir_replace_entry(&parent_ref, name, new_id, new_type))?;
            let new_parent_dir = new_parent_ref.as_ref().ok_or(format_error!(
                ErrCode::EINVAL,
                "rename_exchange: missing new parent reference for cross-dir exchange"
            ))?;
            self.poison_on_error(self.dir_replace_entry(
                new_parent_dir,
                new_name,
                old_id,
                old_type,
            ))?;
        }

        // 6. 跨目录时更新目录的 ".." 指向和父目录 link_count
        if parent != new_parent {
            if old_is_dir {
                self.poison_on_error(self.dir_replace_entry(
                    &old_inode,
                    "..",
                    new_parent,
                    FileType::Directory,
                ))?;
                parent_ref
                    .inode
                    .set_link_count(parent_ref.inode.link_count() - 1);
                self.poison_on_error(self.write_inode_with_csum(&mut parent_ref))?;
                let np = new_parent_ref.as_mut().ok_or(format_error!(
                    ErrCode::EINVAL,
                    "rename_exchange: missing new parent reference for old_dir update"
                ))?;
                np.inode.set_link_count(np.inode.link_count() + 1);
                self.poison_on_error(self.write_inode_with_csum(np))?;
            }
            if new_is_dir {
                self.poison_on_error(self.dir_replace_entry(
                    &new_inode,
                    "..",
                    parent,
                    FileType::Directory,
                ))?;
                let np = new_parent_ref.as_mut().ok_or(format_error!(
                    ErrCode::EINVAL,
                    "rename_exchange: missing new parent reference for new_dir update"
                ))?;
                np.inode.set_link_count(np.inode.link_count() - 1);
                self.poison_on_error(self.write_inode_with_csum(np))?;
                parent_ref
                    .inode
                    .set_link_count(parent_ref.inode.link_count() + 1);
                self.poison_on_error(self.write_inode_with_csum(&mut parent_ref))?;
            }
        }

        Ok(())
    }

    /// Create a directory. This function will not check name conflict,
    /// call `lookup` to check beforehand.
    ///
    /// # Params
    ///
    /// * `parent` - the inode of the directory to create in
    /// * `name` - the name of the directory to create
    /// * `mode` - the mode of the directory to create, type field will be ignored
    ///
    /// # Return
    ///
    /// `Ok(child)` - the inode id of the created directory
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOSPC` - no space left on device
    pub fn mkdir(&self, parent: InodeId, name: &str, mode: InodeMode) -> Result<InodeId> {
        self.mkdir_with_owner(parent, name, mode, InodeOwner { uid: 0, gid: 0 })
    }

    pub fn mkdir_with_owner(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        owner: InodeOwner,
    ) -> Result<InodeId> {
        Ok(self
            .mkdir_with_owner_and_attr(parent, name, mode, owner)?
            .ino)
    }

    /// Create and link a directory, returning the attributes from the
    /// authoritative in-memory inode used by the namespace transaction.
    pub fn mkdir_with_owner_and_attr(
        &self,
        parent: InodeId,
        name: &str,
        mode: InodeMode,
        owner: InodeOwner,
    ) -> Result<FileAttr> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _namespace_guard = self.namespace_lock.lock();
        let _mutation_guards = self.lock_inode_mutations(&[parent]);
        let mut parent = self.read_inode(parent)?;
        // Can only create a directory in a directory
        if !parent.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", parent.id);
        }
        // Create file/directory
        let mode = mode & InodeMode::PERM_MASK | InodeMode::DIRECTORY;
        let mut child = self.create_inode_with_owner(mode, owner.uid, owner.gid)?;
        // Add "." entry
        let child_self = child.clone();
        if let Err(error) = self.dir_add_entry(&mut child, &child_self, ".") {
            if self.free_inode(&mut child).is_err() {
                self.poison(ErrCode::EIO);
            }
            return Err(error);
        }
        child.inode.set_link_count(1);
        // Link the new inode
        self.link_new_inode_or_free(&mut parent, &mut child, name)?;
        Ok(Self::file_attr(&child))
    }

    /// Look up a directory entry by name.
    ///
    /// # Params
    ///
    /// * `parent` - the inode of the directory to look in
    /// * `name` - the name of the entry to look for
    ///
    /// # Return
    ///
    /// `Ok(child)`- the inode id to which the directory entry points.
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` is not a directory
    /// * `ENOENT` - `name` does not exist in `parent`
    pub fn lookup(&self, parent: InodeId, name: &str) -> Result<InodeId> {
        let parent = self.read_inode(parent)?;
        // Can only lookup in a directory
        if !parent.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", parent.id);
        }
        self.dir_find_entry(&parent, name)
    }

    /// List all directory entries in a directory.
    ///
    /// # Params
    ///
    /// * `inode` - the inode of the directory to list
    ///
    /// # Return
    ///
    /// `Ok(entries)` - a vector of directory entries in the directory.
    ///
    /// # Error
    ///
    /// `ENOTDIR` - `inode` is not a directory
    pub fn listdir(&self, inode: InodeId) -> Result<Vec<DirEntry>> {
        let inode_ref = self.read_inode(inode)?;
        // Can only list a directory
        if inode_ref.inode.file_type() != FileType::Directory {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", inode);
        }
        self.dir_list_entries(&inode_ref)
    }

    /// Remove an empty directory.
    ///
    /// # Params
    ///
    /// * `parent` - the parent directory where the directory is located
    /// * `name` - the name of the directory to remove
    ///
    /// # Error
    ///
    /// * `ENOTDIR` - `parent` or `child` is not a directory
    /// * `ENOENT` - `name` does not exist in `parent`
    /// * `ENOTEMPTY` - `child` is not empty
    pub fn rmdir(&self, parent: InodeId, name: &str) -> Result<Option<InodeReclaimHandle>> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_transactional_metadata_mutation()?;
        // See unlink(): rmdir also stages a final-link orphan transaction and
        // must start from an empty deferred journal batch.
        self.flush_deferred_journal()?;
        let _namespace_guard = self.namespace_lock.lock();
        let mut parent_ref = self.read_inode(parent)?;
        // Can only remove a directory in a directory
        if !parent_ref.inode.is_dir() {
            return_error!(
                ErrCode::ENOTDIR,
                "Inode {} is not a directory",
                parent_ref.id
            );
        }
        let location = self.dir_find_entry_location(&parent_ref, name)?;
        let child_id = location.inode_id;
        let _mutation_guards = self.lock_inode_mutations(&[parent, child_id]);
        parent_ref = self.read_inode(parent)?;
        let mut child = self.read_inode(child_id)?;
        // Child must be a directory
        if !child.inode.is_dir() {
            return_error!(ErrCode::ENOTDIR, "Inode {} is not a directory", child.id);
        }
        // Child must be empty
        if self.dir_list_entries(&child)?.len() > 2 {
            return_error!(ErrCode::ENOTEMPTY, "Directory {} is not empty", child.id);
        }
        // Remove directory entry
        self.unlink_inode(&mut parent_ref, &mut child, name, location)
    }

    /// Get extended attribute of a file.
    ///
    /// # Params
    ///
    /// * `inode` - the inode of the file
    /// * `name` - the name of the attribute
    ///
    /// # Return
    ///
    /// `Ok(value)` - the value of the attribute
    ///
    /// # Error
    ///
    /// `ENODATA` - the attribute does not exist
    pub fn getxattr(&self, inode: InodeId, name: &str) -> Result<Vec<u8>> {
        let inode_ref = self.read_inode(inode)?;
        let xattr_block_id = inode_ref.inode.xattr_block();
        if xattr_block_id == 0 {
            return_error!(ErrCode::ENODATA, "Xattr {} does not exist", name);
        }
        let xattr_block = XattrBlock::new(self.read_block(xattr_block_id)?);
        self.verify_xattr_block_checksum(xattr_block_id, &xattr_block)?;
        match xattr_block.get(name) {
            Some(value) => Ok(value.to_owned()),
            None => Err(format_error!(
                ErrCode::ENODATA,
                "Xattr {} does not exist",
                name
            )),
        }
    }

    /// Set extended attribute of a file.
    ///
    /// # Params
    ///
    /// * `inode` - the inode of the file
    /// * `name` - the name of the attribute
    /// * `value` - the value of the attribute
    ///
    /// # Error
    ///
    /// `ENOSPC` - xattr block does not have enough space
    pub fn setxattr(&self, inode: InodeId, name: &str, value: &[u8]) -> Result<()> {
        self.ensure_mutable()?;
        self.setxattr_with_flags(inode, name, value, false, false)
    }

    /// Set extended attribute of a file with Linux create/replace semantics.
    ///
    /// Existing xattr blocks are modified on a cloned candidate block first and
    /// written back only after the whole operation succeeds. This preserves the
    /// old value when replacing with a value that does not fit.
    pub fn setxattr_with_flags(
        &self,
        inode: InodeId,
        name: &str,
        value: &[u8],
        create: bool,
        replace: bool,
    ) -> Result<()> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard =
            self.inode_mutation_locks[self.inode_mutation_lock_index(inode)].lock();
        let mut inode_ref = self.read_inode(inode)?;
        let xattr_block_id = inode_ref.inode.xattr_block();
        if xattr_block_id == 0 {
            if replace {
                return_error!(ErrCode::ENODATA, "Xattr {} does not exist", name);
            }
            // lazy allocate xattr block
            let pblock = self.alloc_block(&mut inode_ref)?;
            let old_xattr_block = xattr_block_id;
            let result = (|| {
                let mut xattr_block = XattrBlock::new(self.read_block(pblock)?);
                xattr_block.init();
                if !xattr_block.insert(name, value) {
                    return_error!(
                        ErrCode::ENOSPC,
                        "Xattr block of Inode {} does not have enough space",
                        inode
                    );
                }
                self.update_xattr_block_checksum(pblock, &mut xattr_block)?;
                self.write_block(&xattr_block.block())?;
                inode_ref.inode.set_xattr_block(pblock);
                self.write_inode_with_csum(&mut inode_ref)?;
                Ok(())
            })();
            if let Err(err) = result {
                inode_ref.inode.set_xattr_block(old_xattr_block);
                return match self.dealloc_block(&mut inode_ref, pblock) {
                    Ok(()) => Err(err),
                    Err(rollback_err) => Err(rollback_err),
                };
            }
            return Ok(());
        }

        let xattr_block = XattrBlock::new(self.read_block(xattr_block_id)?);
        self.verify_xattr_block_checksum(xattr_block_id, &xattr_block)?;
        let exists = xattr_block.get(name).is_some();
        if exists && create {
            return_error!(ErrCode::EEXIST, "Xattr {} already exists", name);
        }
        if !exists && replace {
            return_error!(ErrCode::ENODATA, "Xattr {} does not exist", name);
        }

        let mut new_xattr_block = xattr_block;
        if exists {
            let _ = new_xattr_block.remove(name);
        }
        if new_xattr_block.insert(name, value) {
            self.update_xattr_block_checksum(xattr_block_id, &mut new_xattr_block)?;
            self.write_block(&new_xattr_block.block())?;
            Ok(())
        } else {
            return_error!(
                ErrCode::ENOSPC,
                "Xattr block of Inode {} does not have enough space",
                inode
            );
        }
    }

    /// Remove extended attribute of a file.
    ///
    /// # Params
    ///
    /// * `inode` - the inode of the file
    /// * `name` - the name of the attribute
    ///
    /// # Error
    ///
    /// `ENODATA` - the attribute does not exist
    pub fn removexattr(&self, inode: InodeId, name: &str) -> Result<()> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard =
            self.inode_mutation_locks[self.inode_mutation_lock_index(inode)].lock();
        let inode_ref = self.read_inode(inode)?;
        let xattr_block_id = inode_ref.inode.xattr_block();
        if xattr_block_id == 0 {
            return_error!(ErrCode::ENODATA, "Xattr {} does not exist", name);
        }
        let mut xattr_block = XattrBlock::new(self.read_block(xattr_block_id)?);
        self.verify_xattr_block_checksum(xattr_block_id, &xattr_block)?;
        if xattr_block.remove(name) {
            self.update_xattr_block_checksum(xattr_block_id, &mut xattr_block)?;
            self.write_block(&xattr_block.block())?;
            Ok(())
        } else {
            return_error!(ErrCode::ENODATA, "Xattr {} does not exist", name);
        }
    }

    /// List extended attributes of a file.
    ///
    /// # Params
    ///
    /// * `inode` - the inode of the file
    ///
    /// # Returns
    ///
    /// A list of extended attributes of the file.
    pub fn listxattr(&self, inode: InodeId) -> Result<Vec<String>> {
        let inode_ref = self.read_inode(inode)?;
        let xattr_block_id = inode_ref.inode.xattr_block();
        if xattr_block_id == 0 {
            return Ok(Vec::new());
        }
        let xattr_block = XattrBlock::new(self.read_block(xattr_block_id)?);
        self.verify_xattr_block_checksum(xattr_block_id, &xattr_block)?;
        Ok(xattr_block.list())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileType;
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn file_attr_is_derived_from_the_authoritative_in_memory_inode() {
        let mut inode = Box::new(Inode::default());
        inode.set_mode(InodeMode::CHARDEV | InodeMode::from_bits_retain(0o640));
        inode.set_uid(0x12345);
        inode.set_gid(0x23456);
        inode.set_size(0x1_0000_0020);
        inode.set_block_count(17);
        inode.set_atime(11);
        inode.set_mtime(12);
        inode.set_ctime(13);
        inode.set_crtime(14);
        inode.set_link_count(2);
        inode.set_device(259, 0x1_0002);
        let inode = InodeRef::new(42, inode);

        let attr = Ext4::file_attr(&inode);

        assert_eq!(attr.ino, 42);
        assert_eq!(attr.ftype, FileType::CharacterDev);
        assert_eq!(attr.perm.bits(), 0o640);
        assert_eq!(attr.uid, 0x12345);
        assert_eq!(attr.gid, 0x23456);
        assert_eq!(attr.size, 0x1_0000_0020);
        assert_eq!(attr.blocks, 17);
        assert_eq!(
            (attr.atime, attr.mtime, attr.ctime, attr.crtime),
            (11, 12, 13, 14)
        );
        assert_eq!(attr.links, 2);
        assert_eq!(attr.rdev, (259, 0x1_0002));
    }

    struct StubBlockDevice {
        sb_block: Block,
    }

    impl StubBlockDevice {
        fn with_block_count(block_count: u32) -> Self {
            let mut data = [0u8; BLOCK_SIZE];
            let off = BASE_OFFSET;
            data[off..off + 4].copy_from_slice(&block_count.to_le_bytes());
            Self {
                sb_block: Block::new(0, Box::new(data)),
            }
        }
    }

    impl BlockDevice for StubBlockDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            if block_id == 0 {
                Ok(self.sb_block.clone())
            } else {
                Ok(Block::new(block_id, Box::new([0u8; BLOCK_SIZE])))
            }
        }

        fn write_block(&self, _block: &Block) -> Result<()> {
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }
        fn supports_reliable_flush(&self) -> bool {
            true
        }
    }

    fn make_test_fs(block_count: u32) -> Ext4 {
        let block_device = Arc::new(StubBlockDevice::with_block_count(block_count));
        make_test_fs_with_device(block_count, block_device)
    }

    fn make_test_fs_with_device(block_count: u32, block_device: Arc<dyn BlockDevice>) -> Ext4 {
        let block = block_device.read_block(0).unwrap();
        let sb = block.read_offset_as::<SuperBlock>(BASE_OFFSET);
        Ext4 {
            block_device,
            cached_super_block: spin::Mutex::new(sb),
            cached_block_groups: Vec::new(),
            system_metadata_ranges: Vec::new(),
            inode_cache: spin::Mutex::new(crate::ext4::InodeCache::new(16)),
            alloc_lock: spin::Mutex::new(()),
            namespace_lock: spin::Mutex::new(()),
            metadata_mutation_barrier: crate::ext4::MetadataMutationGate::new(),
            poisoned: spin::Mutex::new(None),
            metadata_mode: crate::ext4::MetadataMutationMode::Direct(
                crate::ext4::journal_transaction::DirectTransactionCore::new(block_count as u64)
                    .unwrap(),
            ),
            write_barrier: true,
            direct_restore_clean: false,
            inode_mutation_locks: (0..crate::ext4::INODE_MUTATION_LOCK_SHARDS)
                .map(|_| spin::Mutex::new(()))
                .collect(),
            prepared_extents: spin::Mutex::new(
                crate::ext4::prepared_extent::PreparedExtentCache::new(),
            ),
            prepare_stats: crate::ext4::PrepareStats::new(),
        }
    }

    struct RangeInitDevice {
        sb_block: Block,
        writes: AtomicUsize,
        flushes: AtomicUsize,
        fail_write_at: AtomicUsize,
        fail_flush: AtomicBool,
    }

    impl RangeInitDevice {
        fn new(block_count: u32) -> Self {
            let stub = StubBlockDevice::with_block_count(block_count);
            Self {
                sb_block: stub.sb_block,
                writes: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                fail_write_at: AtomicUsize::new(usize::MAX),
                fail_flush: AtomicBool::new(false),
            }
        }
    }

    impl BlockDevice for RangeInitDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            if block_id == 0 {
                Ok(self.sb_block.clone())
            } else {
                Ok(Block::new(block_id, Box::new([0; BLOCK_SIZE])))
            }
        }

        fn write_block(&self, _block: &Block) -> Result<()> {
            Ok(())
        }

        fn write_blocks(&self, _start: PBlockId, _data: &[u8]) -> Result<()> {
            let write = self.writes.fetch_add(1, Ordering::SeqCst);
            if write == self.fail_write_at.load(Ordering::SeqCst) {
                Err(Ext4Error::new(ErrCode::ENOMEM))
            } else {
                Ok(())
            }
        }

        fn flush(&self) -> Result<()> {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            if self.fail_flush.load(Ordering::SeqCst) {
                Err(Ext4Error::new(ErrCode::EIO))
            } else {
                Ok(())
            }
        }

        fn supports_reliable_flush(&self) -> bool {
            true
        }
    }

    #[test]
    fn direct_range_zero_failure_stops_before_flush() {
        let device = Arc::new(RangeInitDevice::new(128));
        device.fail_write_at.store(1, Ordering::SeqCst);
        let fs = make_test_fs_with_device(128, device.clone());
        let zeros = [0; DIRECT_RANGE_ZERO_CHUNK_BLOCKS * BLOCK_SIZE];

        let error = fs
            .initialize_direct_range(32, DIRECT_RANGE_ZERO_CHUNK_BLOCKS * 2, &zeros)
            .unwrap_err();

        assert_eq!(error.code(), ErrCode::ENOMEM);
        assert_eq!(device.writes.load(Ordering::SeqCst), 2);
        assert_eq!(device.flushes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn direct_range_zero_initialization_defers_flush_failure() {
        let device = Arc::new(RangeInitDevice::new(128));
        device.fail_flush.store(true, Ordering::SeqCst);
        let fs = make_test_fs_with_device(128, device.clone());
        let zeros = [0; DIRECT_RANGE_ZERO_CHUNK_BLOCKS * BLOCK_SIZE];

        fs.initialize_direct_range(32, DIRECT_RANGE_ZERO_CHUNK_BLOCKS * 2, &zeros)
            .unwrap();

        assert_eq!(device.writes.load(Ordering::SeqCst), 2);
        assert_eq!(device.flushes.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn direct_range_plan_filters_small_or_oversized_writes() {
        let fs = make_test_fs(1024);
        let mut inode = Inode::default();
        inode.extent_init();
        let mut inode = InodeRef::new(2, Box::new(inode));

        let small = Ext4::checked_write_logical_range(0, BLOCK_SIZE)
            .unwrap()
            .unwrap();
        assert!(fs.direct_range_plan(&inode, &small).unwrap().is_none());
        let oversized =
            Ext4::checked_write_logical_range(0, (DIRECT_RANGE_MAX_BLOCKS + 1) * BLOCK_SIZE)
                .unwrap()
                .unwrap();
        assert!(fs.direct_range_plan(&inode, &oversized).unwrap().is_none());
        let exact = Ext4::checked_write_logical_range(0, DIRECT_RANGE_MIN_BLOCKS * BLOCK_SIZE)
            .unwrap()
            .unwrap();
        assert!(fs.direct_range_plan(&inode, &exact).unwrap().is_some());

        inode
            .inode
            .set_size((DIRECT_RANGE_MIN_BLOCKS * BLOCK_SIZE) as u64);
        assert!(fs.direct_range_plan(&inode, &exact).unwrap().is_none());
    }

    #[test]
    fn read_extent_or_hole_zero_fills_only_missing_extent() {
        let fs = make_test_fs(16);
        let mut inode = Inode::default();
        inode.extent_init();
        let inode = InodeRef::new(2, Box::new(inode));
        let mut buf = [0x5a; 16];

        fs.read_extent_or_hole(&inode, 0, 0, &mut buf).unwrap();

        assert_eq!(buf, [0; 16]);
    }

    #[test]
    fn metadata_mutation_barrier_separates_direct_and_transactional_writers() {
        let fs = make_test_fs(16);

        let direct = fs.lock_direct_metadata_mutation().unwrap();
        let second_direct = fs.lock_direct_metadata_mutation().unwrap();
        assert_eq!(
            fs.lock_transactional_metadata_mutation()
                .expect_err("exclusive gate must not wait for direct owners")
                .code(),
            ErrCode::EAGAIN
        );
        drop(second_direct);
        drop(direct);

        let transaction = fs.lock_transactional_metadata_mutation().unwrap();
        assert_eq!(
            fs.lock_direct_metadata_mutation()
                .expect_err("direct gate must not wait for exclusive owner")
                .code(),
            ErrCode::EAGAIN
        );
        assert_eq!(
            fs.lock_transactional_metadata_mutation()
                .expect_err("second exclusive owner must be rejected")
                .code(),
            ErrCode::EAGAIN
        );
        drop(transaction);
        drop(fs.lock_transactional_metadata_mutation().unwrap());
        drop(fs.lock_direct_metadata_mutation().unwrap());
    }

    #[test]
    fn metadata_mutation_barrier_notifies_only_after_guard_release() {
        struct Counter(Arc<AtomicUsize>);

        impl crate::MetadataMutationNotifier for Counter {
            fn notify(&self) {
                self.0.fetch_add(1, Ordering::Release);
            }
        }

        let fs = make_test_fs(16);
        let notifications = Arc::new(AtomicUsize::new(0));
        fs.set_metadata_mutation_notifier(Arc::new(Counter(notifications.clone())));

        let first = fs.lock_direct_metadata_mutation().unwrap();
        let second = fs.lock_direct_metadata_mutation().unwrap();
        assert_eq!(notifications.load(Ordering::Acquire), 0);
        drop(first);
        assert_eq!(notifications.load(Ordering::Acquire), 1);
        drop(second);
        assert_eq!(notifications.load(Ordering::Acquire), 2);

        let exclusive = fs.lock_transactional_metadata_mutation().unwrap();
        assert_eq!(notifications.load(Ordering::Acquire), 2);
        drop(exclusive);
        assert_eq!(notifications.load(Ordering::Acquire), 3);
    }

    #[test]
    fn metadata_mutation_barrier_rejects_direct_count_overflow() {
        let fs = make_test_fs(16);
        fs.metadata_mutation_barrier.state.store(
            crate::ext4::METADATA_GATE_DIRECT_MAX,
            core::sync::atomic::Ordering::Relaxed,
        );
        assert_eq!(
            fs.lock_direct_metadata_mutation()
                .expect_err("direct count must not enter the exclusive bit")
                .code(),
            ErrCode::EAGAIN
        );
        fs.metadata_mutation_barrier
            .state
            .store(0, core::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn metadata_mutation_barrier_allows_concurrent_direct_owners() {
        let fs = make_test_fs(16);
        let start = std::sync::Barrier::new(3);
        let release = std::sync::Barrier::new(3);
        let (sender, receiver) = std::sync::mpsc::channel();

        std::thread::scope(|scope| {
            for _ in 0..2 {
                let sender = sender.clone();
                let start = &start;
                let release = &release;
                let fs = &fs;
                scope.spawn(move || {
                    start.wait();
                    let guard = fs.lock_direct_metadata_mutation();
                    sender.send(guard.is_ok()).unwrap();
                    release.wait();
                    drop(guard);
                });
            }
            start.wait();
            assert!(receiver.recv().unwrap());
            assert!(receiver.recv().unwrap());
            assert_eq!(
                fs.lock_transactional_metadata_mutation()
                    .expect_err("both direct guards must remain live")
                    .code(),
                ErrCode::EAGAIN
            );
            release.wait();
        });
        drop(fs.lock_transactional_metadata_mutation().unwrap());
    }

    #[test]
    fn read_extent_or_hole_propagates_extent_corruption() {
        let fs = make_test_fs(16);
        let inode = InodeRef::new(2, Box::new(Inode::default()));
        let mut buf = [0x5a; 16];

        let err = fs
            .read_extent_or_hole(&inode, 0, 0, &mut buf)
            .expect_err("invalid extent root must not be treated as a hole");

        assert_eq!(err.code(), ErrCode::EIO);
        assert_eq!(buf, [0x5a; 16]);
    }

    const TEST_BLOCK_COUNT: usize = 16;
    const TEST_BLOCK_BITMAP: PBlockId = 2;
    const TEST_INODE_BITMAP: PBlockId = 3;
    const TEST_INODE_TABLE: PBlockId = 4;
    const TEST_XATTR_BLOCK: PBlockId = 5;
    const TEST_INITIAL_FREE_BLOCKS: u64 = (TEST_BLOCK_COUNT as u64) - 5;

    struct FailingBlockDevice {
        blocks: spin::Mutex<BTreeMap<PBlockId, Block>>,
        fail_reads: spin::Mutex<Vec<PBlockId>>,
        fail_writes: spin::Mutex<Vec<PBlockId>>,
    }

    impl FailingBlockDevice {
        fn new() -> Self {
            let mut blocks = BTreeMap::new();
            for block_id in 0..TEST_BLOCK_COUNT as PBlockId {
                blocks.insert(block_id, Block::new(block_id, Box::new([0u8; BLOCK_SIZE])));
            }

            let mut sb_block = blocks.remove(&0).unwrap();
            Self::write_u32(&mut sb_block, BASE_OFFSET, 16);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 4, TEST_BLOCK_COUNT as u32);
            Self::write_u32(
                &mut sb_block,
                BASE_OFFSET + 12,
                TEST_INITIAL_FREE_BLOCKS as u32,
            );
            Self::write_u32(&mut sb_block, BASE_OFFSET + 20, 0);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 24, 2);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 28, 2);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 32, TEST_BLOCK_COUNT as u32);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 36, TEST_BLOCK_COUNT as u32);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 40, 16);
            Self::write_u16(&mut sb_block, BASE_OFFSET + 56, 0xef53);
            Self::write_u32(&mut sb_block, BASE_OFFSET + 84, 1);
            Self::write_u16(&mut sb_block, BASE_OFFSET + 88, SB_GOOD_INODE_SIZE as u16);
            Self::write_u16(&mut sb_block, BASE_OFFSET + 254, SB_GOOD_DESC_SIZE as u16);
            blocks.insert(0, sb_block);

            let mut bgdt = blocks.remove(&1).unwrap();
            Self::write_u32(&mut bgdt, 0, TEST_BLOCK_BITMAP as u32);
            Self::write_u32(&mut bgdt, 4, TEST_INODE_BITMAP as u32);
            Self::write_u32(&mut bgdt, 8, TEST_INODE_TABLE as u32);
            Self::write_u16(&mut bgdt, 12, TEST_INITIAL_FREE_BLOCKS as u16);
            blocks.insert(1, bgdt);

            let mut bitmap = blocks.remove(&TEST_BLOCK_BITMAP).unwrap();
            bitmap.data[0] = 0b0001_1111;
            blocks.insert(TEST_BLOCK_BITMAP, bitmap);

            let mut inode_table = blocks.remove(&TEST_INODE_TABLE).unwrap();
            let mut inode = Inode::default();
            inode.set_mode(InodeMode::from_type_and_perm(
                FileType::RegularFile,
                InodeMode::from_bits_retain(0o644),
            ));
            inode.set_link_count(1);
            inode_table.write_offset_as(SB_GOOD_INODE_SIZE, &inode);
            blocks.insert(TEST_INODE_TABLE, inode_table);

            Self {
                blocks: spin::Mutex::new(blocks),
                fail_reads: spin::Mutex::new(Vec::new()),
                fail_writes: spin::Mutex::new(Vec::new()),
            }
        }

        fn write_u16(block: &mut Block, offset: usize, value: u16) {
            block.write_offset(offset, &value.to_le_bytes());
        }

        fn write_u32(block: &mut Block, offset: usize, value: u32) {
            block.write_offset(offset, &value.to_le_bytes());
        }

        fn fail_once_on_read(&self, block_id: PBlockId) {
            self.fail_reads.lock().push(block_id);
        }

        fn fail_once_on_write(&self, block_id: PBlockId) {
            self.fail_writes.lock().push(block_id);
        }

        fn take_failure(list: &mut Vec<PBlockId>, block_id: PBlockId) -> bool {
            if let Some(pos) = list.iter().position(|&id| id == block_id) {
                list.remove(pos);
                true
            } else {
                false
            }
        }

        fn block_bitmap_bit_is_set(&self, bit: usize) -> bool {
            let blocks = self.blocks.lock();
            let block = blocks.get(&TEST_BLOCK_BITMAP).unwrap();
            (block.data[bit / 8] & (1 << (bit % 8))) != 0
        }

        fn bg_free_blocks(&self) -> u64 {
            let blocks = self.blocks.lock();
            let block = blocks.get(&1).unwrap();
            u16::from_le_bytes(block.data[12..14].try_into().unwrap()) as u64
        }

        fn sb_free_blocks(&self) -> u64 {
            let blocks = self.blocks.lock();
            let block = blocks.get(&0).unwrap();
            u32::from_le_bytes(
                block.data[BASE_OFFSET + 12..BASE_OFFSET + 16]
                    .try_into()
                    .unwrap(),
            ) as u64
        }

        fn disk_inode_xattr_block(&self) -> PBlockId {
            let blocks = self.blocks.lock();
            let block = blocks.get(&TEST_INODE_TABLE).unwrap();
            let inode: Inode = block.read_offset_as(SB_GOOD_INODE_SIZE);
            inode.xattr_block()
        }

        fn fill_block(&self, block_id: PBlockId, byte: u8) {
            self.blocks
                .lock()
                .get_mut(&block_id)
                .unwrap()
                .data
                .fill(byte);
        }

        fn block_is_zero(&self, block_id: PBlockId) -> bool {
            self.blocks
                .lock()
                .get(&block_id)
                .unwrap()
                .data
                .iter()
                .all(|byte| *byte == 0)
        }
    }

    impl BlockDevice for FailingBlockDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            if Self::take_failure(&mut self.fail_reads.lock(), block_id) {
                return Err(Ext4Error::new(ErrCode::EIO));
            }
            self.blocks
                .lock()
                .get(&block_id)
                .cloned()
                .ok_or_else(|| Ext4Error::new(ErrCode::EIO))
        }

        fn write_block(&self, block: &Block) -> Result<()> {
            if Self::take_failure(&mut self.fail_writes.lock(), block.id) {
                return Err(Ext4Error::new(ErrCode::EIO));
            }
            self.blocks.lock().insert(block.id, block.clone());
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }
        fn supports_reliable_flush(&self) -> bool {
            true
        }
    }

    fn load_failing_test_fs() -> (Arc<FailingBlockDevice>, Ext4) {
        let block_device = Arc::new(FailingBlockDevice::new());
        let mut fs = Ext4::load(block_device.clone()).unwrap();
        fs.initialize_direct().unwrap();
        (block_device, fs)
    }

    fn assert_xattr_alloc_rolled_back(fs: &Ext4, block_device: &FailingBlockDevice) {
        assert!(!block_device.block_bitmap_bit_is_set(TEST_XATTR_BLOCK as usize));
        assert_eq!(block_device.bg_free_blocks(), TEST_INITIAL_FREE_BLOCKS);
        assert_eq!(block_device.sb_free_blocks(), TEST_INITIAL_FREE_BLOCKS);
        assert_eq!(
            fs.read_block_group(0).unwrap().desc.get_free_blocks_count(),
            TEST_INITIAL_FREE_BLOCKS
        );
        assert_eq!(
            fs.read_super_block_cached().free_blocks_count(),
            TEST_INITIAL_FREE_BLOCKS
        );
        assert_eq!(block_device.disk_inode_xattr_block(), 0);
    }

    fn assert_allocation_state(
        fs: &Ext4,
        block_device: &FailingBlockDevice,
        allocated: bool,
        free_blocks: u64,
    ) {
        assert_eq!(
            block_device.block_bitmap_bit_is_set(TEST_XATTR_BLOCK as usize),
            allocated
        );
        assert_eq!(block_device.bg_free_blocks(), free_blocks);
        assert_eq!(block_device.sb_free_blocks(), free_blocks);
        assert_eq!(
            fs.read_block_group(0).unwrap().desc.get_free_blocks_count(),
            free_blocks
        );
        assert_eq!(
            fs.read_super_block_cached().free_blocks_count(),
            free_blocks
        );
    }

    #[test]
    fn setxattr_rolls_back_when_new_xattr_block_read_fails() {
        let (block_device, fs) = load_failing_test_fs();
        block_device.fail_once_on_read(TEST_XATTR_BLOCK);

        let err = fs
            .setxattr_with_flags(2, "user.rollback", b"value", false, false)
            .unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_xattr_alloc_rolled_back(&fs, &block_device);
    }

    #[test]
    fn setxattr_rolls_back_when_new_xattr_block_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        block_device.fail_once_on_write(TEST_XATTR_BLOCK);

        let err = fs
            .setxattr_with_flags(2, "user.rollback", b"value", false, false)
            .unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_xattr_alloc_rolled_back(&fs, &block_device);
    }

    #[test]
    fn setxattr_rolls_back_when_inode_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        block_device.fail_once_on_write(TEST_INODE_TABLE);

        let err = fs
            .setxattr_with_flags(2, "user.rollback", b"value", false, false)
            .unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_xattr_alloc_rolled_back(&fs, &block_device);
    }

    #[test]
    fn setxattr_rolls_back_when_new_xattr_does_not_fit() {
        let (block_device, fs) = load_failing_test_fs();
        let value = vec![0x5au8; BLOCK_SIZE];

        let err = fs
            .setxattr_with_flags(2, "user.rollback", &value, false, false)
            .unwrap_err();

        assert_eq!(err.code(), ErrCode::ENOSPC);
        assert_xattr_alloc_rolled_back(&fs, &block_device);
    }

    #[test]
    fn block_group_cache_updates_only_after_disk_write_succeeds() {
        let (block_device, fs) = load_failing_test_fs();
        let mut bg = fs.read_block_group(0).unwrap();
        bg.desc.set_free_blocks_count(TEST_INITIAL_FREE_BLOCKS - 1);
        block_device.fail_once_on_write(1);

        let err = fs.write_block_group_with_csum(&mut bg).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_eq!(
            fs.read_block_group(0).unwrap().desc.get_free_blocks_count(),
            TEST_INITIAL_FREE_BLOCKS
        );
        assert_eq!(block_device.bg_free_blocks(), TEST_INITIAL_FREE_BLOCKS);
    }

    #[test]
    fn alloc_block_rolls_back_when_block_group_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        block_device.fail_once_on_write(1);

        let err = fs.alloc_block(&mut inode).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_allocation_state(&fs, &block_device, false, TEST_INITIAL_FREE_BLOCKS);
    }

    #[test]
    fn alloc_block_rolls_back_when_superblock_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        block_device.fail_once_on_write(0);

        let err = fs.alloc_block(&mut inode).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_allocation_state(&fs, &block_device, false, TEST_INITIAL_FREE_BLOCKS);
    }

    #[test]
    fn newly_reused_data_block_is_zeroed_before_mapping() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        block_device.fill_block(TEST_XATTR_BLOCK, 0xa5);

        let pblock = fs.alloc_zeroed_data_block(&mut inode).unwrap();

        assert_eq!(pblock, TEST_XATTR_BLOCK);
        assert!(block_device.block_is_zero(pblock));
        assert_allocation_state(&fs, &block_device, true, TEST_INITIAL_FREE_BLOCKS - 1);
    }

    #[test]
    fn data_block_zero_write_failure_rolls_back_allocation() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        block_device.fill_block(TEST_XATTR_BLOCK, 0xa5);
        block_device.fail_once_on_write(TEST_XATTR_BLOCK);

        let err = fs.alloc_zeroed_data_block(&mut inode).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_allocation_state(&fs, &block_device, false, TEST_INITIAL_FREE_BLOCKS);
        assert!(!block_device.block_is_zero(TEST_XATTR_BLOCK));
    }

    #[test]
    fn dealloc_block_rolls_back_when_block_group_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        let pblock = fs.alloc_block(&mut inode).unwrap();
        assert_eq!(pblock, TEST_XATTR_BLOCK);
        assert_allocation_state(&fs, &block_device, true, TEST_INITIAL_FREE_BLOCKS - 1);
        block_device.fail_once_on_write(1);

        let err = fs.dealloc_block(&mut inode, pblock).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_allocation_state(&fs, &block_device, true, TEST_INITIAL_FREE_BLOCKS - 1);
    }

    #[test]
    fn dealloc_block_rolls_back_when_superblock_write_fails() {
        let (block_device, fs) = load_failing_test_fs();
        let mut inode = fs.read_inode(2).unwrap();
        let pblock = fs.alloc_block(&mut inode).unwrap();
        assert_eq!(pblock, TEST_XATTR_BLOCK);
        assert_allocation_state(&fs, &block_device, true, TEST_INITIAL_FREE_BLOCKS - 1);
        block_device.fail_once_on_write(0);

        let err = fs.dealloc_block(&mut inode, pblock).unwrap_err();

        assert_eq!(err.code(), ErrCode::EIO);
        assert_allocation_state(&fs, &block_device, true, TEST_INITIAL_FREE_BLOCKS - 1);
    }
}
