//! Synchronous, single-writer JBD2 transaction core.
//!
//! This module deliberately does not discover the journal inode.  Mount code
//! must validate the journal superblock and provide the complete logical to
//! physical block map.  Keeping mapping outside the commit path also ensures
//! that no filesystem spin lock is held while block I/O is in flight.
#![allow(dead_code)] // Activated only after every production metadata writer uses handles.

use crate::constants::BLOCK_SIZE;
use crate::ext4_defs::{Block, BlockDevice, JournalCommitReason, JournalFlushPhase};
use crate::jbd2::{
    block_checksum, checksum_seed, commit_checksum, tag_checksum, BlockType, ChecksumMode,
    Features, Header, Superblock, BLOCK_TAIL_BYTES, CRC32C_CHKSUM, FLAG_ESCAPE, FLAG_LAST_TAG,
    FLAG_SAME_UUID, HEADER_BYTES, MAGIC,
};
use crate::prelude::*;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const CLEAN: u8 = 0;
const POISONED: u8 = 1;

/// A fully validated journal mapping and its current allocation cursor.
///
/// `logical_blocks[n]` is the filesystem physical block containing journal
/// logical block `n`.  Entry zero is the JBD2 superblock.
pub struct JournalContext {
    pub superblock: Superblock,
    pub logical_blocks: Arc<[PBlockId]>,
    pub journal_blocks: Arc<BTreeSet<PBlockId>>,
    /// Number of addressable blocks on the filesystem device.
    pub target_blocks: u64,
    /// Next journal logical block to allocate.  It must be in the data ring.
    pub head: u32,
    /// Exact 1024-byte JBD2 superblock image read at mount.
    pub superblock_image: Box<[u8; 1024]>,
    /// Metadata images accumulated before the next journal commit.
    ///
    /// The batch stores owned images instead of a `Transaction<'_>` so no
    /// writer borrow survives the writeback call that created it.
    pub deferred_transaction: Option<DeferredTransaction>,
}

/// Uncommitted metadata images retained for a bounded writeback batch.
pub(crate) struct DeferredTransaction {
    staged: BTreeMap<PBlockId, Box<[u8; BLOCK_SIZE]>>,
}

impl JournalContext {
    pub fn validate(&self) -> Result<()> {
        let sb = &self.superblock;
        if sb.block_size as usize != BLOCK_SIZE
            || (sb.features.checksum == ChecksumMode::V3 && sb.checksum_type != CRC32C_CHKSUM)
            || self.logical_blocks.len() != sb.max_len as usize
            || self.journal_blocks.len() != self.logical_blocks.len()
            || self.head < sb.first
            || self.head >= sb.max_len
            || self.target_blocks == 0
        {
            return Err(Ext4Error::new(ErrCode::ENOTSUP));
        }
        if self
            .logical_blocks
            .iter()
            .any(|block| *block == 0 || !self.journal_blocks.contains(block))
        {
            return Err(Ext4Error::new(ErrCode::EIO));
        }
        Ok(())
    }
}

/// A transaction-private metadata image.  It is intentionally not `Clone`:
/// callers cannot retain a second publishable copy past commit/abort.
pub struct StagedBlock {
    home: PBlockId,
    original: Option<Box<[u8; BLOCK_SIZE]>>,
    image: Box<[u8; BLOCK_SIZE]>,
}

impl StagedBlock {
    pub fn home(&self) -> PBlockId {
        self.home
    }

    pub fn bytes(&self) -> &[u8; BLOCK_SIZE] {
        &self.image
    }
}

/// Publishes metadata images after the selected backend has completed its
/// commit point. Journal commits publish after a durable checkpoint; direct
/// commits publish after every synchronous home-block write has succeeded.
pub trait CachePublisher: Send + Sync {
    /// Publish an in-memory view of a deferred transaction before releasing
    /// the writer token. The images are not durable yet, but subsequent VFS
    /// operations must observe the completed mutation. Implementations must
    /// be infallible and idempotent because the same images are published
    /// again after their checkpoint becomes durable.
    fn publish_pending(&self, _blocks: &BTreeMap<PBlockId, StagedBlock>) {}

    /// Publish already-checkpointed images to in-memory caches.
    ///
    /// This callback runs after the home-block flush, so it must not allocate,
    /// perform I/O, or otherwise fail.  Borrowing the transaction's map also
    /// lets publishers inspect the final image for a particular home block
    /// without building a temporary collection.
    fn publish(&self, blocks: &BTreeMap<PBlockId, StagedBlock>);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitFailure {
    /// No commit record was issued; recovery will ignore any log fragments.
    BeforeCommit,
    /// Commit write or its durability flush failed, so commit is uncertain.
    CommitUncertain,
    /// The transaction committed, but home blocks are not known durable.
    CheckpointFailed,
    /// Home blocks are durable, but the journal tail was not durably cleared.
    TailUpdateFailed,
}

#[derive(Debug)]
pub struct CommitError {
    pub error: Ext4Error,
    pub failure: CommitFailure,
}

/// Owns the single-writer token and poison state.  The token is held only as
/// an atomic bit; no spin guard crosses a block-device operation.
pub struct JournalTransactionCore {
    writer: AtomicBool,
    poison: AtomicU8,
    context: spin::Mutex<JournalContext>,
}

/// Synchronous nojournal transaction backend.
///
/// The writer bit is deliberately separate from the filesystem-wide metadata
/// mutation gate: the gate excludes legacy direct writers, while this token
/// owns one staged transaction within this backend.
pub struct DirectTransactionCore {
    writer: AtomicBool,
    poison: AtomicU8,
    target_blocks: u64,
}

impl DirectTransactionCore {
    pub fn new(target_blocks: u64) -> Result<Self> {
        if target_blocks == 0 {
            return Err(Ext4Error::new(ErrCode::EINVAL));
        }
        Ok(Self {
            writer: AtomicBool::new(false),
            poison: AtomicU8::new(CLEAN),
            target_blocks,
        })
    }

    pub fn is_poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire) != CLEAN
    }

    pub fn can_shutdown(&self) -> bool {
        !self.writer.load(Ordering::Acquire) && !self.is_poisoned()
    }

    pub fn start(&self, credits: usize) -> Result<Transaction<'_>> {
        self.start_with_originals(credits, false)
    }

    pub(super) fn start_direct_range(&self, credits: usize) -> Result<Transaction<'_>> {
        self.start_with_originals(credits, true)
    }

    fn start_with_originals(
        &self,
        credits: usize,
        preserve_originals: bool,
    ) -> Result<Transaction<'_>> {
        if credits == 0 || self.is_poisoned() {
            return Err(Ext4Error::new(if credits == 0 {
                ErrCode::EINVAL
            } else {
                ErrCode::EROFS
            }));
        }
        if self
            .writer
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(Ext4Error::new(ErrCode::EAGAIN));
        }
        Ok(Transaction::new(
            TransactionCoreRef::Direct(self),
            credits,
            preserve_originals,
            None,
        ))
    }

    fn poison(&self) {
        self.poison.store(POISONED, Ordering::Release);
    }
}

impl JournalTransactionCore {
    pub fn new(context: JournalContext) -> Result<Self> {
        context.validate()?;
        Ok(Self {
            writer: AtomicBool::new(false),
            poison: AtomicU8::new(CLEAN),
            context: spin::Mutex::new(context),
        })
    }

    pub fn is_poisoned(&self) -> bool {
        self.poison.load(Ordering::Acquire) != CLEAN
    }

    pub fn can_shutdown(&self) -> bool {
        !self.writer.load(Ordering::Acquire)
            && !self.is_poisoned()
            && self.context.lock().deferred_transaction.is_none()
    }

    pub fn has_pending_transaction(&self) -> bool {
        self.context.lock().deferred_transaction.is_some()
    }

    pub(super) fn deferred_block(&self, home: PBlockId) -> Option<Box<[u8; BLOCK_SIZE]>> {
        self.context
            .lock()
            .deferred_transaction
            .as_ref()
            .and_then(|transaction| transaction.staged.get(&home))
            .cloned()
    }

    /// Start a transaction only when it can extend an existing deferred batch.
    ///
    /// A caller that receives `None` must preserve its normal direct-metadata
    /// behavior.  The writer token makes the pending-batch decision atomic with
    /// taking its staged images, so a durability boundary cannot split the
    /// metadata update from the batch it extends.
    pub(super) fn start_if_deferred(&self, credits: usize) -> Result<Option<Transaction<'_>>> {
        if credits == 0 || self.is_poisoned() {
            return Err(Ext4Error::new(if credits == 0 {
                ErrCode::EINVAL
            } else {
                ErrCode::EROFS
            }));
        }
        if self
            .writer
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(Ext4Error::new(ErrCode::EAGAIN));
        }

        let (deferred_staged, total_credits) = {
            let mut context = self.context.lock();
            let Some(deferred) = context.deferred_transaction.as_ref() else {
                self.writer.store(false, Ordering::Release);
                return Ok(None);
            };
            let total = match deferred.staged.len().checked_add(credits) {
                Some(total) => total,
                None => {
                    self.writer.store(false, Ordering::Release);
                    return Err(Ext4Error::new(ErrCode::E2BIG));
                }
            };
            let fits =
                match required_log_blocks(total, context.superblock.features).and_then(|needed| {
                    ring_len(&context.superblock).map(|available| needed <= available)
                }) {
                    Ok(fits) => fits,
                    Err(error) => {
                        self.writer.store(false, Ordering::Release);
                        return Err(error);
                    }
                };
            if !fits {
                self.writer.store(false, Ordering::Release);
                return Err(Ext4Error::new(ErrCode::E2BIG));
            }
            (context.deferred_transaction.take(), total)
        };
        Ok(Some(Transaction::new(
            TransactionCoreRef::Journal(self),
            total_credits,
            false,
            deferred_staged,
        )))
    }

    pub fn owns_block_range(&self, start: PBlockId, end: PBlockId) -> bool {
        start < end
            && self
                .context
                .lock()
                .journal_blocks
                .range(start..end)
                .next()
                .is_some()
    }

    pub fn start(&self, credits: usize) -> Result<Transaction<'_>> {
        if credits == 0 || self.is_poisoned() {
            return Err(Ext4Error::new(if credits == 0 {
                ErrCode::EINVAL
            } else {
                ErrCode::EROFS
            }));
        }
        if self
            .writer
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(Ext4Error::new(ErrCode::EAGAIN));
        }

        // Reserve against a strict upper bound before any mutation is staged.
        let reservation = {
            let context = self.context.lock();
            let pending = context
                .deferred_transaction
                .as_ref()
                .map_or(0, |transaction| transaction.staged.len());
            let total = pending
                .checked_add(credits)
                .ok_or_else(|| Ext4Error::new(ErrCode::E2BIG))?;
            required_log_blocks(total, context.superblock.features)
                .and_then(|needed| {
                    ring_len(&context.superblock).map(|available| needed <= available)
                })
                .map(|fits| (fits, total))
        };
        let (fits, total_credits) = match reservation {
            Ok(reservation) => reservation,
            Err(error) => {
                self.writer.store(false, Ordering::Release);
                return Err(error);
            }
        };
        if !fits {
            self.writer.store(false, Ordering::Release);
            return Err(Ext4Error::new(ErrCode::E2BIG));
        }
        let deferred_staged = self.context.lock().deferred_transaction.take();
        Ok(Transaction::new(
            TransactionCoreRef::Journal(self),
            total_credits,
            false,
            deferred_staged,
        ))
    }

    /// Commit all metadata images accumulated by deferred writeback batches.
    pub fn flush_deferred_transaction(
        &self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
        reason: JournalCommitReason,
    ) -> core::result::Result<bool, CommitError> {
        if self
            .writer
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(CommitError {
                error: Ext4Error::new(ErrCode::EAGAIN),
                failure: CommitFailure::BeforeCommit,
            });
        }
        let deferred = self.context.lock().deferred_transaction.take();
        let Some(deferred) = deferred else {
            self.writer.store(false, Ordering::Release);
            return Ok(false);
        };
        let transaction = Transaction::from_deferred(self, deferred);
        transaction.commit_journal(device, publisher, reason)?;
        Ok(true)
    }

    fn poison(&self) {
        self.poison.store(POISONED, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
enum TransactionCoreRef<'a> {
    Journal(&'a JournalTransactionCore),
    Direct(&'a DirectTransactionCore),
}

pub struct Transaction<'a> {
    core: TransactionCoreRef<'a>,
    credits: usize,
    staged: BTreeMap<PBlockId, StagedBlock>,
    /// Images from an earlier deferred batch. They stay separate until this
    /// transaction commits so `abort` can restore only the prior batch.
    deferred_staged: Option<DeferredTransaction>,
    preserve_originals: bool,
    owns_writer: bool,
}

impl<'a> Transaction<'a> {
    fn new(
        core: TransactionCoreRef<'a>,
        credits: usize,
        preserve_originals: bool,
        deferred_staged: Option<DeferredTransaction>,
    ) -> Transaction<'a> {
        Transaction {
            core,
            credits,
            staged: BTreeMap::new(),
            deferred_staged,
            preserve_originals,
            owns_writer: true,
        }
    }

    fn from_deferred(
        core: &'a JournalTransactionCore,
        deferred: DeferredTransaction,
    ) -> Transaction<'a> {
        let staged: BTreeMap<PBlockId, StagedBlock> = deferred
            .staged
            .into_iter()
            .map(|(home, image)| {
                (
                    home,
                    StagedBlock {
                        home,
                        original: None,
                        image,
                    },
                )
            })
            .collect();
        Self {
            core: TransactionCoreRef::Journal(core),
            credits: staged.len(),
            staged,
            deferred_staged: None,
            preserve_originals: false,
            owns_writer: true,
        }
    }

    fn total_staged_len(&self) -> usize {
        let deferred = self
            .deferred_staged
            .as_ref()
            .map_or(0, |transaction| transaction.staged.len());
        deferred
            + self
                .staged
                .keys()
                .filter(|home| {
                    self.deferred_staged
                        .as_ref()
                        .map_or(true, |transaction| !transaction.staged.contains_key(home))
                })
                .count()
    }

    fn deferred_image(&self, home: PBlockId) -> Option<Box<[u8; BLOCK_SIZE]>> {
        self.deferred_staged
            .as_ref()
            .and_then(|transaction| transaction.staged.get(&home))
            .cloned()
    }

    fn restore_deferred_staged(&mut self) {
        let Some(deferred) = self.deferred_staged.take() else {
            return;
        };
        let TransactionCoreRef::Journal(core) = self.core else {
            return;
        };
        let mut context = core.context.lock();
        debug_assert!(context.deferred_transaction.is_none());
        context.deferred_transaction = Some(deferred);
    }

    fn absorb_deferred_staged(&mut self) {
        let Some(deferred) = self.deferred_staged.take() else {
            return;
        };
        for (home, image) in deferred.staged {
            self.staged.entry(home).or_insert(StagedBlock {
                home,
                original: None,
                image,
            });
        }
    }
    /// Replace the final image for `home`.  Re-staging the same home block does
    /// not consume another credit and subsequent reads observe the replacement.
    pub fn stage(&mut self, home: PBlockId, image: Box<[u8; BLOCK_SIZE]>) -> Result<()> {
        let was_deferred = self.deferred_image(home).is_some();
        if !self.staged.contains_key(&home)
            && !was_deferred
            && self.total_staged_len() >= self.credits
        {
            return Err(Ext4Error::new(ErrCode::E2BIG));
        }
        self.staged.insert(
            home,
            StagedBlock {
                home,
                original: None,
                image,
            },
        );
        Ok(())
    }

    /// Return the transaction-private final image of `home` for mutation.
    ///
    /// The first access snapshots the device block and consumes one credit;
    /// later accesses return the same image, providing read-your-writes and
    /// naturally merging updates to shared metadata blocks.
    pub fn read_for_update<'tx>(
        &'tx mut self,
        device: &dyn BlockDevice,
        home: PBlockId,
    ) -> Result<&'tx mut [u8; BLOCK_SIZE]> {
        if !self.staged.contains_key(&home) {
            let deferred = self.deferred_image(home);
            if deferred.is_none() && self.total_staged_len() >= self.credits {
                return Err(Ext4Error::new(ErrCode::E2BIG));
            }
            let (image, original) = match deferred {
                Some(image) => (image, None),
                None => {
                    let block = device.read_block(home)?;
                    let original = self.preserve_originals.then(|| block.data.clone());
                    (block.data, original)
                }
            };
            self.staged.insert(
                home,
                StagedBlock {
                    home,
                    original,
                    image,
                },
            );
        }
        Ok(self
            .staged
            .get_mut(&home)
            .expect("staged block was just inserted")
            .image
            .as_mut())
    }

    pub fn read<'tx>(
        &'tx self,
        device: &dyn BlockDevice,
        home: PBlockId,
    ) -> Result<BlockView<'tx>> {
        if let Some(block) = self.staged.get(&home) {
            Ok(BlockView::Staged(block.bytes()))
        } else if let Some(block) = self
            .deferred_staged
            .as_ref()
            .and_then(|transaction| transaction.staged.get(&home))
        {
            Ok(BlockView::Staged(block))
        } else {
            Ok(BlockView::Device(device.read_block(home)?))
        }
    }

    pub fn abort(mut self) {
        self.restore_deferred_staged();
        self.release_writer();
    }

    /// Retain a journal transaction in memory until a bounded batch is ready.
    ///
    /// Data blocks were initialized before their allocation metadata was
    /// staged, so a crash before this batch commits can only leak initialized
    /// storage; it cannot publish an extent that refers to free blocks.
    pub fn defer_or_commit(
        mut self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
        max_deferred_blocks: usize,
    ) -> core::result::Result<(), CommitError> {
        let TransactionCoreRef::Journal(core) = self.core else {
            return self.commit(device, publisher);
        };
        let total = self.total_staged_len();
        if total == 0 {
            self.release_writer();
            return Ok(());
        }
        let must_commit = {
            let context = core.context.lock();
            let required =
                required_log_blocks(total, context.superblock.features).map_err(|error| {
                    CommitError {
                        error,
                        failure: CommitFailure::BeforeCommit,
                    }
                })?;
            let available = ring_len(&context.superblock).map_err(|error| CommitError {
                error,
                failure: CommitFailure::BeforeCommit,
            })?;
            total >= max_deferred_blocks || required.saturating_mul(2) >= available
        };
        if must_commit {
            return self.commit_journal(device, publisher, JournalCommitReason::DeferredThreshold);
        }

        self.absorb_deferred_staged();
        let mut context = core.context.lock();
        debug_assert!(context.deferred_transaction.is_none());
        // Keep deferred read-through blocked on the context lock until cache
        // invalidation/publication and ownership transfer are both complete.
        publisher.publish_pending(&self.staged);
        let staged = core::mem::take(&mut self.staged)
            .into_iter()
            .map(|(home, staged)| (home, staged.image))
            .collect();
        context.deferred_transaction = Some(DeferredTransaction { staged });
        drop(context);
        self.release_writer();
        Ok(())
    }

    pub fn commit(
        mut self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
    ) -> core::result::Result<(), CommitError> {
        if self.staged.is_empty() {
            self.release_writer();
            return Ok(());
        }
        match self.core {
            TransactionCoreRef::Journal(_) => {
                self.commit_journal(device, publisher, JournalCommitReason::Explicit)
            }
            TransactionCoreRef::Direct(_) => self.commit_direct(device, publisher),
        }
    }

    /// Commit a nojournal range allocation in semantic order.
    ///
    /// Allocation homes are written before the inode-table home which makes
    /// the extent reachable.  A failure before the inode write rolls back the
    /// completed allocation homes from their transaction-private snapshots.
    /// Once the inode write is attempted its on-disk state is uncertain, so
    /// the direct backend is poisoned instead of releasing possibly-owned
    /// blocks for reuse.
    pub(super) fn commit_direct_range(
        mut self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
        allocation_homes: &[PBlockId],
        inode_home: PBlockId,
    ) -> core::result::Result<(), CommitError> {
        let TransactionCoreRef::Direct(core) = self.core else {
            return self.fail(
                Ext4Error::new(ErrCode::EINVAL),
                CommitFailure::BeforeCommit,
                false,
            );
        };
        if !device.supports_reliable_flush()
            || inode_home >= core.target_blocks
            || allocation_homes.is_empty()
            || allocation_homes.contains(&inode_home)
            || self.staged.len() != allocation_homes.len() + 1
            || !self.staged.contains_key(&inode_home)
            || allocation_homes.iter().any(|home| {
                *home >= core.target_blocks
                    || !self.staged.contains_key(home)
                    || self.staged[home].original.is_none()
            })
            || self.staged[&inode_home].original.is_none()
        {
            return self.fail(
                Ext4Error::new(ErrCode::EINVAL),
                CommitFailure::BeforeCommit,
                false,
            );
        }

        for (completed, home) in allocation_homes.iter().enumerate() {
            let staged = &self.staged[home];
            if let Err(error) = write_bytes(device, *home, staged.bytes()) {
                // The failed write may have reached the device partially.
                // Rewrite its old image as well as every earlier home before
                // declaring the operation safely aborted.
                let rollback = self.rollback_direct_range(device, &allocation_homes[..=completed]);
                return match rollback {
                    Ok(()) => self.fail(error, CommitFailure::BeforeCommit, false),
                    Err(rollback_error) => {
                        self.fail(rollback_error, CommitFailure::CheckpointFailed, true)
                    }
                };
            }
        }

        // The inode must never become durable before the allocation metadata
        // that owns its blocks. Without this boundary a volatile device cache
        // may persist the inode first and expose its blocks as free after a
        // crash. A failed flush is still before publication, so restore every
        // allocation home while the exclusive transaction owner is held.
        if let Err(error) = device.flush() {
            let rollback = self.rollback_direct_range(device, allocation_homes);
            return match rollback {
                Ok(()) => self.fail(error, CommitFailure::BeforeCommit, false),
                Err(rollback_error) => {
                    self.fail(rollback_error, CommitFailure::CheckpointFailed, true)
                }
            };
        }

        let inode = &self.staged[&inode_home];
        if let Err(error) = write_bytes(device, inode_home, inode.bytes()) {
            return self.fail(error, CommitFailure::CommitUncertain, true);
        }
        publisher.publish(&self.staged);
        self.release_writer();
        Ok(())
    }

    fn rollback_direct_range(
        &self,
        device: &dyn BlockDevice,
        completed_homes: &[PBlockId],
    ) -> Result<()> {
        for home in completed_homes.iter().rev() {
            let original = self.staged[home]
                .original
                .as_ref()
                .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
            write_bytes(device, *home, original)?;
        }
        if !completed_homes.is_empty() && device.supports_reliable_flush() {
            device.flush()?;
        }
        Ok(())
    }

    fn commit_direct(
        mut self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
    ) -> core::result::Result<(), CommitError> {
        let TransactionCoreRef::Direct(core) = self.core else {
            unreachable!()
        };
        if self
            .staged
            .values()
            .any(|block| block.home >= core.target_blocks)
        {
            return self.fail(
                Ext4Error::new(ErrCode::EINVAL),
                CommitFailure::BeforeCommit,
                false,
            );
        }
        // BTreeMap iteration supplies a stable physical-block order without a
        // second allocation. It is deterministic, not crash atomic.
        for staged in self.staged.values() {
            if let Err(error) = write_bytes(device, staged.home, staged.bytes()) {
                return self.fail(error, CommitFailure::CheckpointFailed, true);
            }
        }
        publisher.publish(&self.staged);
        self.release_writer();
        Ok(())
    }

    fn commit_journal(
        mut self,
        device: &dyn BlockDevice,
        publisher: &dyn CachePublisher,
        reason: JournalCommitReason,
    ) -> core::result::Result<(), CommitError> {
        self.absorb_deferred_staged();
        let TransactionCoreRef::Journal(core) = self.core else {
            unreachable!()
        };
        if !device.supports_reliable_flush() {
            return self.fail(
                Ext4Error::new(ErrCode::ENOTSUP),
                CommitFailure::BeforeCommit,
                false,
            );
        }

        // Copy the small allocation state, then release the spin guard before I/O.
        let (sb, mapping, journal_blocks, target_blocks, head, sb_image) = {
            let context = core.context.lock();
            (
                context.superblock,
                Arc::clone(&context.logical_blocks),
                Arc::clone(&context.journal_blocks),
                context.target_blocks,
                context.head,
                context.superblock_image.clone(),
            )
        };
        let needed =
            required_log_blocks(self.staged.len(), sb.features).map_err(|error| CommitError {
                error,
                failure: CommitFailure::BeforeCommit,
            })?;
        if needed
            > ring_len(&sb).map_err(|error| CommitError {
                error,
                failure: CommitFailure::BeforeCommit,
            })?
        {
            return self.fail(
                Ext4Error::new(ErrCode::E2BIG),
                CommitFailure::BeforeCommit,
                false,
            );
        }

        let sequence = sb.sequence;
        let diagnostic = device.diagnostic_enabled();
        let commit_start = if diagnostic {
            device.diagnostic_cycles()
        } else {
            0
        };
        let positions = ring_positions(&sb, head, needed).map_err(|error| CommitError {
            error,
            failure: CommitFailure::BeforeCommit,
        })?;
        if self
            .staged
            .values()
            .any(|block| block.home >= target_blocks || journal_blocks.contains(&block.home))
        {
            return self.fail(
                Ext4Error::new(ErrCode::EINVAL),
                CommitFailure::BeforeCommit,
                false,
            );
        }
        // Finish every fallible format operation before the first write.  Once
        // the active tail reaches disk, all remaining failures are I/O failures
        // which must poison the mount.
        let encoded = encode_log(&sb, sequence, &self.staged).map_err(|error| CommitError {
            error,
            failure: CommitFailure::BeforeCommit,
        })?;
        let commit = encode_commit(&sb, sequence).map_err(|error| CommitError {
            error,
            failure: CommitFailure::BeforeCommit,
        })?;
        let mut active_sb_image = sb_image.clone();
        update_superblock(&mut active_sb_image, sequence, head, &sb).map_err(|error| {
            CommitError {
                error,
                failure: CommitFailure::BeforeCommit,
            }
        })?;
        let next_sequence = sequence.wrapping_add(1);
        let mut clean_sb_image = sb_image;
        update_superblock(&mut clean_sb_image, next_sequence, 0, &sb).map_err(|error| {
            CommitError {
                error,
                failure: CommitFailure::BeforeCommit,
            }
        })?;

        if let Err(error) = write_journal_superblock(device, &mapping, &active_sb_image) {
            return self.fail(error, CommitFailure::BeforeCommit, true);
        }

        debug_assert_eq!(encoded.len() + 1, positions.len());
        // P11: Group contiguous journal physical blocks and issue one
        // write_blocks() per run to amortize I/O submission overhead.
        let mut run_start = 0usize;
        while run_start < encoded.len() {
            let first_phys = mapping[positions[run_start] as usize];
            let mut run_end = run_start + 1;
            while run_end < encoded.len() {
                let expected = first_phys.wrapping_add((run_end - run_start) as u64);
                if mapping[positions[run_end] as usize] != expected {
                    break;
                }
                run_end += 1;
            }
            let run_len = run_end - run_start;
            if run_len > 1 {
                let total_bytes = run_len * BLOCK_SIZE;
                let mut buf = Vec::with_capacity(total_bytes);
                buf.resize(total_bytes, 0u8);
                for (i, src_box) in encoded[run_start..run_end].iter().enumerate() {
                    let dest = &mut buf[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE];
                    let src: &[u8] = src_box.as_ref();
                    dest.copy_from_slice(src);
                }
                if let Err(error) = device.write_blocks(first_phys, &buf) {
                    return self.fail(error, CommitFailure::BeforeCommit, true);
                }
            } else {
                let block: &[u8; BLOCK_SIZE] = encoded[run_start].as_ref();
                if let Err(error) = write_bytes(device, first_phys, block) {
                    return self.fail(error, CommitFailure::BeforeCommit, true);
                }
            }
            run_start = run_end;
        }
        // Single barrier for active-SB + payload (was two separate flushes)
        if let Err(error) =
            flush_journal_phase(device, sequence, JournalFlushPhase::ActiveLog, diagnostic)
        {
            return self.fail(error, CommitFailure::BeforeCommit, true);
        }

        let commit_logical = *positions.last().unwrap();
        if let Err(error) = write_bytes(device, mapping[commit_logical as usize], &commit) {
            return self.fail(error, CommitFailure::CommitUncertain, true);
        }
        if let Err(error) = flush_journal_phase(
            device,
            sequence,
            JournalFlushPhase::CommitRecord,
            diagnostic,
        ) {
            return self.fail(error, CommitFailure::CommitUncertain, true);
        }

        for staged in self.staged.values() {
            if let Err(error) = write_bytes(device, staged.home, staged.bytes()) {
                return self.fail(error, CommitFailure::CheckpointFailed, true);
            }
        }
        if let Err(error) =
            flush_journal_phase(device, sequence, JournalFlushPhase::Checkpoint, diagnostic)
        {
            return self.fail(error, CommitFailure::CheckpointFailed, true);
        }

        publisher.publish(&self.staged);

        if let Err(error) = write_journal_superblock(device, &mapping, &clean_sb_image) {
            return self.fail(error, CommitFailure::TailUpdateFailed, true);
        }
        if let Err(error) =
            flush_journal_phase(device, sequence, JournalFlushPhase::TailUpdate, diagnostic)
        {
            return self.fail(error, CommitFailure::TailUpdateFailed, true);
        }

        device.record_journal_commit(encoded.len().saturating_add(1).saturating_mul(BLOCK_SIZE));
        if diagnostic {
            device.record_writeback_journal_commit(
                sequence,
                self.staged.len(),
                device.diagnostic_cycles().wrapping_sub(commit_start),
                reason,
            );
        }

        {
            let mut ctx = core.context.lock();
            ctx.superblock.sequence = next_sequence;
            ctx.superblock.start = 0;
            ctx.head = ring_next(&ctx.superblock, commit_logical);
            ctx.superblock_image = clean_sb_image;
        }

        self.release_writer();
        Ok(())
    }

    fn fail<T>(
        &mut self,
        error: Ext4Error,
        failure: CommitFailure,
        poison: bool,
    ) -> core::result::Result<T, CommitError> {
        if poison {
            match self.core {
                TransactionCoreRef::Journal(core) => core.poison(),
                TransactionCoreRef::Direct(core) => core.poison(),
            }
        }
        self.restore_deferred_staged();
        self.release_writer();
        Err(CommitError { error, failure })
    }

    fn release_writer(&mut self) {
        if self.owns_writer {
            self.owns_writer = false;
            match self.core {
                TransactionCoreRef::Journal(core) => core.writer.store(false, Ordering::Release),
                TransactionCoreRef::Direct(core) => core.writer.store(false, Ordering::Release),
            }
        }
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        self.restore_deferred_staged();
        self.release_writer();
    }
}

pub enum BlockView<'a> {
    Staged(&'a [u8; BLOCK_SIZE]),
    Device(Block),
}

impl core::ops::Deref for BlockView<'_> {
    type Target = [u8; BLOCK_SIZE];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Staged(bytes) => bytes,
            Self::Device(block) => &block.data,
        }
    }
}

fn ring_len(sb: &Superblock) -> Result<usize> {
    sb.max_len
        .checked_sub(sb.first)
        .map(|n| n as usize)
        .ok_or_else(|| Ext4Error::new(ErrCode::EIO))
}

fn tags_per_descriptor(features: Features) -> Result<usize> {
    let tail = if features.checksum == ChecksumMode::V3 {
        BLOCK_TAIL_BYTES
    } else {
        0
    };
    let overhead = HEADER_BYTES + 16 + tail;
    let available = BLOCK_SIZE
        .checked_sub(overhead)
        .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
    let tags = available / features.tag_bytes();
    if tags == 0 {
        Err(Ext4Error::new(ErrCode::EIO))
    } else {
        Ok(tags)
    }
}

fn required_log_blocks(blocks: usize, features: Features) -> Result<usize> {
    let descriptors = blocks
        .checked_add(tags_per_descriptor(features)? - 1)
        .ok_or_else(|| Ext4Error::new(ErrCode::E2BIG))?
        / tags_per_descriptor(features)?;
    blocks
        .checked_add(descriptors)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| Ext4Error::new(ErrCode::E2BIG))
}

fn ring_next(sb: &Superblock, current: u32) -> u32 {
    if current + 1 == sb.max_len {
        sb.first
    } else {
        current + 1
    }
}

fn ring_positions(sb: &Superblock, head: u32, count: usize) -> Result<Vec<u32>> {
    if count > ring_len(sb)? || head < sb.first || head >= sb.max_len {
        return Err(Ext4Error::new(ErrCode::E2BIG));
    }
    let mut result = Vec::with_capacity(count);
    let mut current = head;
    for _ in 0..count {
        result.push(current);
        current = ring_next(sb, current);
    }
    Ok(result)
}

fn encode_log(
    sb: &Superblock,
    sequence: u32,
    staged: &BTreeMap<PBlockId, StagedBlock>,
) -> Result<Vec<Box<[u8; BLOCK_SIZE]>>> {
    let seed = checksum_seed(&sb.uuid);
    let per_descriptor = tags_per_descriptor(sb.features)?;
    let all = staged.values().collect::<Vec<_>>();
    let mut output = Vec::new();
    for group in all.chunks(per_descriptor) {
        let mut descriptor = Box::new([0; BLOCK_SIZE]);
        Header {
            block_type: BlockType::Descriptor,
            sequence,
        }
        .encode_into(&mut descriptor[..])?;
        let mut journal_data = Vec::with_capacity(group.len());
        let mut offset = HEADER_BYTES;
        for (index, staged) in group.iter().enumerate() {
            let mut image = staged.image.clone();
            let mut flags = if index != 0 { FLAG_SAME_UUID } else { 0 };
            if u32::from_be_bytes(
                image[..4]
                    .try_into()
                    .map_err(|_| Ext4Error::new(ErrCode::EIO))?,
            ) == MAGIC
            {
                image[..4].fill(0);
                flags |= FLAG_ESCAPE;
            }
            if index + 1 == group.len() {
                flags |= FLAG_LAST_TAG;
            }
            descriptor[offset..offset + 4].copy_from_slice(&(staged.home as u32).to_be_bytes());
            match sb.features.checksum {
                ChecksumMode::V3 => {
                    if !sb.features.has_64bit && staged.home > u32::MAX as u64 {
                        return Err(Ext4Error::new(ErrCode::ENOTSUP));
                    }
                    descriptor[offset + 4..offset + 8].copy_from_slice(&flags.to_be_bytes());
                    descriptor[offset + 8..offset + 12]
                        .copy_from_slice(&((staged.home >> 32) as u32).to_be_bytes());
                    descriptor[offset + 12..offset + 16]
                        .copy_from_slice(&tag_checksum(seed, sequence, &image[..]).to_be_bytes());
                }
                ChecksumMode::None => {
                    if staged.home > u32::MAX as u64 {
                        return Err(Ext4Error::new(ErrCode::ENOTSUP));
                    }
                    descriptor[offset + 6..offset + 8]
                        .copy_from_slice(&(flags as u16).to_be_bytes());
                }
            }
            offset += sb.features.tag_bytes();
            if index == 0 {
                descriptor[offset..offset + 16].copy_from_slice(&sb.uuid);
                offset += 16;
            }
            journal_data.push(image);
        }
        if sb.features.checksum == ChecksumMode::V3 {
            let checksum = block_checksum(seed, &descriptor[..])?;
            descriptor[BLOCK_SIZE - 4..].copy_from_slice(&checksum.to_be_bytes());
        }
        output.push(descriptor);
        output.extend(journal_data);
    }
    Ok(output)
}

fn encode_commit(sb: &Superblock, sequence: u32) -> Result<Box<[u8; BLOCK_SIZE]>> {
    let mut block = Box::new([0; BLOCK_SIZE]);
    Header {
        block_type: BlockType::Commit,
        sequence,
    }
    .encode_into(&mut block[..])?;
    if sb.features.checksum == ChecksumMode::V3 {
        let checksum = commit_checksum(checksum_seed(&sb.uuid), &block[..])?;
        block[16..20].copy_from_slice(&checksum.to_be_bytes());
    }
    Ok(block)
}

fn update_superblock(
    image: &mut [u8; 1024],
    sequence: u32,
    start: u32,
    sb: &Superblock,
) -> Result<()> {
    image[24..28].copy_from_slice(&sequence.to_be_bytes());
    image[28..32].copy_from_slice(&start.to_be_bytes());
    if sb.features.checksum == ChecksumMode::V3 {
        image[252..256].fill(0);
        let checksum = crate::jbd2::superblock_checksum(image)?;
        image[252..256].copy_from_slice(&checksum.to_be_bytes());
    }
    Ok(())
}

fn write_journal_superblock(
    device: &dyn BlockDevice,
    mapping: &[PBlockId],
    image: &[u8; 1024],
) -> Result<()> {
    let mut block = device.read_block(mapping[0])?;
    block.data[..1024].copy_from_slice(image);
    device.write_block(&block)
}

fn write_bytes(device: &dyn BlockDevice, id: PBlockId, bytes: &[u8; BLOCK_SIZE]) -> Result<()> {
    device.write_block(&Block::new(id, Box::new(*bytes)))
}

fn flush_journal_phase(
    device: &dyn BlockDevice,
    transaction_id: u32,
    phase: JournalFlushPhase,
    diagnostic: bool,
) -> Result<()> {
    let start = if diagnostic {
        device.diagnostic_cycles()
    } else {
        0
    };
    device.flush()?;
    if diagnostic {
        device.record_writeback_journal_flush(
            transaction_id,
            phase,
            device.diagnostic_cycles().wrapping_sub(start),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    const TEST_MAX_FAST_METADATA_IMAGES: usize = 16;
    const TEST_LARGE_JOURNAL_BLOCKS: u32 = 64;

    struct MemoryDevice {
        volatile: spin::Mutex<BTreeMap<PBlockId, Box<[u8; BLOCK_SIZE]>>>,
        stable: spin::Mutex<BTreeMap<PBlockId, Box<[u8; BLOCK_SIZE]>>>,
        operation: AtomicUsize,
        fail_at: AtomicUsize,
        fail_at_second: AtomicUsize,
        write_order: spin::Mutex<Vec<PBlockId>>,
        reads: AtomicUsize,
        writes: AtomicUsize,
        flushes: AtomicUsize,
        journal_commits: spin::Mutex<Vec<(u32, usize, JournalCommitReason)>>,
        journal_flushes: spin::Mutex<Vec<(u32, JournalFlushPhase)>>,
    }

    impl MemoryDevice {
        fn new() -> Self {
            Self {
                volatile: spin::Mutex::new(BTreeMap::new()),
                stable: spin::Mutex::new(BTreeMap::new()),
                operation: AtomicUsize::new(0),
                fail_at: AtomicUsize::new(usize::MAX),
                fail_at_second: AtomicUsize::new(usize::MAX),
                write_order: spin::Mutex::new(Vec::new()),
                reads: AtomicUsize::new(0),
                writes: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                journal_commits: spin::Mutex::new(Vec::new()),
                journal_flushes: spin::Mutex::new(Vec::new()),
            }
        }

        fn step(&self) -> Result<()> {
            let operation = self.operation.fetch_add(1, Ordering::SeqCst);
            if operation == self.fail_at.load(Ordering::SeqCst)
                || operation == self.fail_at_second.load(Ordering::SeqCst)
            {
                Err(Ext4Error::new(ErrCode::EIO))
            } else {
                Ok(())
            }
        }

        /// Simulate power loss: only the last completed flush epoch survives.
        fn crash(&self) {
            *self.volatile.lock() = self.stable.lock().clone();
        }

        fn stable_block(&self, id: PBlockId) -> Option<Box<[u8; BLOCK_SIZE]>> {
            self.stable.lock().get(&id).cloned()
        }
    }

    impl BlockDevice for MemoryDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(Block::new(
                block_id,
                self.volatile
                    .lock()
                    .get(&block_id)
                    .cloned()
                    .unwrap_or_else(|| Box::new([0; BLOCK_SIZE])),
            ))
        }

        fn write_block(&self, block: &Block) -> Result<()> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            self.step()?;
            self.write_order.lock().push(block.id);
            self.volatile.lock().insert(block.id, block.data.clone());
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            self.step()?;
            *self.stable.lock() = self.volatile.lock().clone();
            Ok(())
        }

        fn supports_reliable_flush(&self) -> bool {
            true
        }

        fn diagnostic_enabled(&self) -> bool {
            true
        }

        fn diagnostic_cycles(&self) -> usize {
            self.operation.load(Ordering::SeqCst)
        }

        fn record_writeback_journal_commit(
            &self,
            transaction_id: u32,
            staged_blocks: usize,
            _cycles: usize,
            reason: JournalCommitReason,
        ) {
            self.journal_commits
                .lock()
                .push((transaction_id, staged_blocks, reason));
        }

        fn record_writeback_journal_flush(
            &self,
            transaction_id: u32,
            phase: JournalFlushPhase,
            _cycles: usize,
        ) {
            self.journal_flushes.lock().push((transaction_id, phase));
        }
    }

    struct Publisher(AtomicUsize);
    impl CachePublisher for Publisher {
        fn publish(&self, blocks: &BTreeMap<PBlockId, StagedBlock>) {
            self.0.fetch_add(blocks.len(), Ordering::SeqCst);
        }
    }

    struct PendingPublisher {
        pending: AtomicUsize,
        committed: AtomicUsize,
    }

    impl CachePublisher for PendingPublisher {
        fn publish_pending(&self, blocks: &BTreeMap<PBlockId, StagedBlock>) {
            self.pending.fetch_add(blocks.len(), Ordering::SeqCst);
        }

        fn publish(&self, blocks: &BTreeMap<PBlockId, StagedBlock>) {
            self.committed.fetch_add(blocks.len(), Ordering::SeqCst);
        }
    }

    #[test]
    fn direct_commit_writes_home_blocks_in_order_then_publishes() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let mut transaction = core.start(3).unwrap();
        transaction.stage(9, Box::new([9; BLOCK_SIZE])).unwrap();
        transaction.stage(2, Box::new([2; BLOCK_SIZE])).unwrap();
        transaction.stage(5, Box::new([5; BLOCK_SIZE])).unwrap();

        transaction.commit(&device, &publisher).unwrap();

        assert_eq!(&*device.write_order.lock(), &[2, 5, 9]);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 3);
        assert!(!core.is_poisoned());
        assert!(core.can_shutdown());
    }

    #[test]
    fn direct_write_failure_never_publishes_and_poisons_backend() {
        let device = MemoryDevice::new();
        device.fail_at.store(1, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let mut transaction = core.start(2).unwrap();
        transaction.stage(2, Box::new([2; BLOCK_SIZE])).unwrap();
        transaction.stage(5, Box::new([5; BLOCK_SIZE])).unwrap();

        let error = transaction.commit(&device, &publisher).unwrap_err();

        assert_eq!(error.failure, CommitFailure::CheckpointFailed);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert!(core.is_poisoned());
        assert_eq!(core.start(1).err().unwrap().code(), ErrCode::EROFS);
    }

    #[test]
    fn direct_validation_failure_releases_writer_without_poison() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(8).unwrap();
        let mut transaction = core.start(1).unwrap();
        transaction.stage(8, Box::new([1; BLOCK_SIZE])).unwrap();

        let error = transaction.commit(&device, &publisher).unwrap_err();

        assert_eq!(error.failure, CommitFailure::BeforeCommit);
        assert!(!core.is_poisoned());
        assert!(core.start(1).is_ok());
    }

    #[test]
    fn direct_backend_allows_only_one_transaction_owner() {
        let core = DirectTransactionCore::new(8).unwrap();
        let transaction = core.start(1).unwrap();
        assert_eq!(core.start(1).err().unwrap().code(), ErrCode::EAGAIN);
        transaction.abort();
        assert!(core.start(1).is_ok());
    }

    #[test]
    fn ordinary_transactions_do_not_retain_rollback_images() {
        let device = MemoryDevice::new();
        let core = DirectTransactionCore::new(8).unwrap();
        let mut transaction = core.start(1).unwrap();

        transaction.read_for_update(&device, 1).unwrap()[0] = 1;

        assert!(transaction.staged[&1].original.is_none());
        transaction.abort();
    }

    #[test]
    fn direct_range_transactions_retain_rollback_images() {
        let device = MemoryDevice::new();
        let core = DirectTransactionCore::new(8).unwrap();
        let mut transaction = core.start_direct_range(1).unwrap();

        transaction.read_for_update(&device, 1).unwrap()[0] = 1;

        assert!(transaction.staged[&1].original.is_some());
        transaction.abort();
    }

    fn staged_direct_range<'a>(
        core: &'a DirectTransactionCore,
        device: &MemoryDevice,
    ) -> Transaction<'a> {
        let mut transaction = core.start_direct_range(4).unwrap();
        for (home, value) in [(9, 9), (2, 2), (5, 5), (7, 7)] {
            transaction
                .read_for_update(device, home)
                .unwrap()
                .fill(value);
        }
        transaction
    }

    #[test]
    fn direct_range_uses_allocation_order_and_inode_last() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap();

        assert_eq!(&*device.write_order.lock(), &[9, 2, 5, 7]);
        assert_eq!(device.flushes.load(Ordering::SeqCst), 1);
        assert_eq!(device.stable_block(9).unwrap().as_slice(), &[9; BLOCK_SIZE]);
        assert!(device.stable_block(7).is_none());
        assert_eq!(publisher.0.load(Ordering::SeqCst), 4);
        assert!(!core.is_poisoned());
    }

    #[test]
    fn direct_range_restores_failed_allocation_home_before_continuing() {
        let device = MemoryDevice::new();
        device.fail_at.store(1, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::BeforeCommit);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert_eq!(&*device.write_order.lock(), &[9, 2, 9]);
        assert_eq!(device.volatile.lock()[&9].as_slice(), &[0; BLOCK_SIZE]);
        assert_eq!(device.volatile.lock()[&2].as_slice(), &[0; BLOCK_SIZE]);
        assert!(!core.is_poisoned());
    }

    #[test]
    fn direct_range_restores_each_failed_allocation_home() {
        for failed_home in 0..3 {
            let device = MemoryDevice::new();
            device.fail_at.store(failed_home, Ordering::SeqCst);
            let publisher = Publisher(AtomicUsize::new(0));
            let core = DirectTransactionCore::new(128).unwrap();
            let transaction = staged_direct_range(&core, &device);

            let error = transaction
                .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
                .unwrap_err();

            assert_eq!(error.failure, CommitFailure::BeforeCommit);
            assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
            for home in [9, 2, 5] {
                let image = device
                    .volatile
                    .lock()
                    .get(&home)
                    .cloned()
                    .unwrap_or_else(|| Box::new([0; BLOCK_SIZE]));
                assert_eq!(image.as_slice(), &[0; BLOCK_SIZE]);
            }
            assert!(!core.is_poisoned());
        }
    }

    #[test]
    fn direct_range_rollback_write_failure_poisons_backend() {
        let device = MemoryDevice::new();
        device.fail_at.store(1, Ordering::SeqCst);
        device.fail_at_second.store(2, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::CheckpointFailed);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert!(core.is_poisoned());
    }

    #[test]
    fn direct_range_rollback_flush_failure_poisons_backend() {
        let device = MemoryDevice::new();
        device.fail_at.store(1, Ordering::SeqCst);
        device.fail_at_second.store(4, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::CheckpointFailed);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert!(core.is_poisoned());
    }

    #[test]
    fn direct_range_inode_write_failure_is_uncertain_and_poisons() {
        let device = MemoryDevice::new();
        device.fail_at.store(4, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::CommitUncertain);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert!(core.is_poisoned());
    }

    #[test]
    fn direct_range_allocation_flush_failure_restores_all_homes() {
        let device = MemoryDevice::new();
        device.fail_at.store(3, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::BeforeCommit);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        for home in [9, 2, 5] {
            assert_eq!(
                device.stable_block(home).unwrap().as_slice(),
                &[0; BLOCK_SIZE]
            );
        }
        assert!(device.stable_block(7).is_none());
        assert!(!core.is_poisoned());
    }

    #[test]
    fn direct_range_allocation_flush_rollback_failure_poisons_backend() {
        let device = MemoryDevice::new();
        device.fail_at.store(3, Ordering::SeqCst);
        device.fail_at_second.store(4, Ordering::SeqCst);
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(128).unwrap();
        let transaction = staged_direct_range(&core, &device);

        let error = transaction
            .commit_direct_range(&device, &publisher, &[9, 2, 5], 7)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::CheckpointFailed);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        assert!(core.is_poisoned());
    }

    #[test]
    fn direct_range_rejects_out_of_range_inode_without_io() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = DirectTransactionCore::new(8).unwrap();
        let mut transaction = core.start_direct_range(4).unwrap();
        for home in [1, 2, 3, 8] {
            transaction.read_for_update(&device, home).unwrap()[0] = 1;
        }

        let error = transaction
            .commit_direct_range(&device, &publisher, &[1, 2, 3], 8)
            .unwrap_err();

        assert_eq!(error.failure, CommitFailure::BeforeCommit);
        assert_eq!(device.writes.load(Ordering::SeqCst), 0);
        assert!(!core.is_poisoned());
    }

    #[test]
    fn read_for_update_merges_shared_block_and_charges_one_credit() {
        let device = MemoryDevice::new();
        let core = JournalTransactionCore::new(context()).unwrap();
        let mut transaction = core.start(1).unwrap();

        transaction.read_for_update(&device, 42).unwrap()[10] = 1;
        transaction.read_for_update(&device, 42).unwrap()[11] = 2;
        assert_eq!(transaction.read(&device, 42).unwrap()[10..12], [1, 2]);
        assert_eq!(
            transaction.read_for_update(&device, 43).unwrap_err().code(),
            ErrCode::E2BIG
        );
    }

    fn context() -> JournalContext {
        context_with_ring(8, 7)
    }

    fn context_with_ring(max_len: u32, head: u32) -> JournalContext {
        let features = Features::validate(
            0,
            crate::jbd2::FEATURE_INCOMPAT_CSUM_V3 | crate::jbd2::FEATURE_INCOMPAT_64BIT,
            0,
        )
        .unwrap();
        let mut image = Box::new([0; 1024]);
        Header {
            block_type: BlockType::SuperblockV2,
            sequence: 7,
        }
        .encode_into(&mut image[..])
        .unwrap();
        image[12..16].copy_from_slice(&(BLOCK_SIZE as u32).to_be_bytes());
        image[16..20].copy_from_slice(&max_len.to_be_bytes());
        image[20..24].copy_from_slice(&1u32.to_be_bytes());
        image[24..28].copy_from_slice(&head.to_be_bytes());
        image[40..44].copy_from_slice(
            &(crate::jbd2::FEATURE_INCOMPAT_CSUM_V3 | crate::jbd2::FEATURE_INCOMPAT_64BIT)
                .to_be_bytes(),
        );
        image[48..64].copy_from_slice(b"0123456789abcdef");
        image[80] = CRC32C_CHKSUM;
        update_superblock(
            &mut image,
            head,
            0,
            &Superblock {
                block_size: BLOCK_SIZE as u32,
                max_len,
                first: 1,
                sequence: 7,
                start: 0,
                errno: 0,
                features,
                uuid: *b"0123456789abcdef",
                checksum_type: CRC32C_CHKSUM,
            },
        )
        .unwrap();
        JournalContext {
            superblock: Superblock::parse(&image[..], BLOCK_SIZE as u32).unwrap(),
            logical_blocks: Vec::from_iter(100..100 + max_len as u64).into(),
            journal_blocks: Arc::new(BTreeSet::from_iter(100..100 + max_len as u64)),
            target_blocks: 1000,
            head,
            superblock_image: image,
            deferred_transaction: None,
        }
    }

    fn assert_counted_commit_shape(homes: usize, expected_reads: usize, expected_writes: usize) {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context_with_ring(
            TEST_LARGE_JOURNAL_BLOCKS,
            TEST_LARGE_JOURNAL_BLOCKS - 1,
        ))
        .unwrap();
        let mut transaction = core.start(homes).unwrap();
        for home in 42..42 + homes as u64 {
            transaction.read_for_update(&device, home).unwrap()[0] = home as u8;
        }

        transaction.commit(&device, &publisher).unwrap();

        // P8 keeps the active-SB and payload barrier merged; the synchronous
        // checkpoint still writes home blocks and the clean superblock.
        assert_eq!(device.reads.load(Ordering::SeqCst), expected_reads);
        assert_eq!(device.writes.load(Ordering::SeqCst), expected_writes);
        assert_eq!(device.flushes.load(Ordering::SeqCst), 4);
        assert_eq!(publisher.0.load(Ordering::SeqCst), homes);
        assert!(!core.is_poisoned());
    }

    #[test]
    fn journal_commit_four_homes_has_expected_fixed_io_shape() {
        // 6 reads: active-SB, clean-SB, and 4 read_for_update calls.
        // 12 writes: active-SB, descriptor, 4 payloads (batched), commit,
        // 4 home blocks, and clean-SB.
        assert_counted_commit_shape(4, 6, 12);
    }

    #[test]
    fn journal_commit_max_fast_metadata_images_has_expected_fixed_io_shape() {
        // 18 reads: active-SB, clean-SB, and 16 read_for_update calls.
        // 36 writes: active-SB, descriptor, 16 payloads (batched), commit,
        // 16 home blocks, and clean-SB.
        assert_counted_commit_shape(TEST_MAX_FAST_METADATA_IMAGES, 18, 36);
    }

    #[test]
    fn journal_commit_records_transaction_reason_and_four_barriers() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context()).unwrap();
        let mut transaction = core.start(1).unwrap();
        transaction.stage(42, Box::new([0x33; BLOCK_SIZE])).unwrap();

        transaction.commit(&device, &publisher).unwrap();

        assert_eq!(
            &*device.journal_commits.lock(),
            &[(7, 1, JournalCommitReason::Explicit)]
        );
        assert_eq!(
            &*device.journal_flushes.lock(),
            &[
                (7, JournalFlushPhase::ActiveLog),
                (7, JournalFlushPhase::CommitRecord),
                (7, JournalFlushPhase::Checkpoint),
                (7, JournalFlushPhase::TailUpdate),
            ]
        );
    }

    #[test]
    fn deferred_writeback_batches_share_one_journal_commit() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context_with_ring(
            TEST_LARGE_JOURNAL_BLOCKS,
            TEST_LARGE_JOURNAL_BLOCKS - 1,
        ))
        .unwrap();

        let mut first = core.start(1).unwrap();
        first.stage(42, Box::new([1; BLOCK_SIZE])).unwrap();
        first
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();
        let mut second = core.start(1).unwrap();
        second.stage(43, Box::new([2; BLOCK_SIZE])).unwrap();
        second
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        assert!(core.has_pending_transaction());
        assert_eq!(device.flushes.load(Ordering::SeqCst), 0);

        assert!(core
            .flush_deferred_transaction(
                &device,
                &publisher,
                JournalCommitReason::DurabilityBoundary,
            )
            .unwrap());
        assert_eq!(device.flushes.load(Ordering::SeqCst), 4);
        assert_eq!(publisher.0.load(Ordering::SeqCst), 2);
        assert!(!core.has_pending_transaction());
    }

    #[test]
    fn deferred_batch_publishes_live_view_before_durability() {
        let device = MemoryDevice::new();
        let publisher = PendingPublisher {
            pending: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
        };
        let core = JournalTransactionCore::new(context_with_ring(
            TEST_LARGE_JOURNAL_BLOCKS,
            TEST_LARGE_JOURNAL_BLOCKS - 1,
        ))
        .unwrap();
        let mut transaction = core.start(1).unwrap();
        transaction.stage(42, Box::new([0xA5; BLOCK_SIZE])).unwrap();

        transaction
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        assert_eq!(publisher.pending.load(Ordering::SeqCst), 1);
        assert_eq!(publisher.committed.load(Ordering::SeqCst), 0);
        assert_eq!(core.deferred_block(42).unwrap()[0], 0xA5);
        assert!(core
            .flush_deferred_transaction(
                &device,
                &publisher,
                JournalCommitReason::DurabilityBoundary,
            )
            .unwrap());
        assert_eq!(publisher.committed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn extending_deferred_batch_uses_total_credit_limit() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context_with_ring(
            TEST_LARGE_JOURNAL_BLOCKS,
            TEST_LARGE_JOURNAL_BLOCKS - 1,
        ))
        .unwrap();

        let mut first = core.start(2).unwrap();
        first.stage(42, Box::new([1; BLOCK_SIZE])).unwrap();
        first.stage(43, Box::new([2; BLOCK_SIZE])).unwrap();
        first
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        let mut extension = core.start(1).unwrap();
        extension.stage(44, Box::new([3; BLOCK_SIZE])).unwrap();
        assert_eq!(
            extension
                .stage(45, Box::new([4; BLOCK_SIZE]))
                .unwrap_err()
                .code(),
            ErrCode::E2BIG
        );
        extension
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        assert!(core
            .flush_deferred_transaction(
                &device,
                &publisher,
                JournalCommitReason::DurabilityBoundary,
            )
            .unwrap());
        assert_eq!(publisher.0.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn deferred_metadata_joins_pending_writeback_until_durability_boundary() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context_with_ring(
            TEST_LARGE_JOURNAL_BLOCKS,
            TEST_LARGE_JOURNAL_BLOCKS - 1,
        ))
        .unwrap();

        // Given data has reached the device before its extent metadata is staged.
        device
            .write_block(&Block::new(50, Box::new([0xD0; BLOCK_SIZE])))
            .unwrap();
        let mut writeback = core.start(1).unwrap();
        writeback.stage(43, Box::new([0xE1; BLOCK_SIZE])).unwrap();
        writeback
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        // When the inode size/timestamp image is prepared for the same batch.
        let mut metadata = core
            .start_if_deferred(1)
            .unwrap()
            .expect("pending writeback must accept compatible metadata");
        metadata.stage(42, Box::new([0xC2; BLOCK_SIZE])).unwrap();
        metadata
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        // Then no DirectMetadataBarrier commit occurs before the sync boundary.
        assert!(core.has_pending_transaction());
        assert_eq!(device.flushes.load(Ordering::SeqCst), 0);
        assert!(device.journal_commits.lock().is_empty());

        assert!(core
            .flush_deferred_transaction(
                &device,
                &publisher,
                JournalCommitReason::DurabilityBoundary,
            )
            .unwrap());

        let journal_commits = device.journal_commits.lock();
        assert_eq!(journal_commits.len(), 1);
        assert_eq!(journal_commits[0].1, 2);
        assert_eq!(
            journal_commits[0].2,
            JournalCommitReason::DurabilityBoundary
        );
        let transaction_id = journal_commits[0].0;
        drop(journal_commits);
        assert_eq!(
            &*device.journal_flushes.lock(),
            &[
                (transaction_id, JournalFlushPhase::ActiveLog),
                (transaction_id, JournalFlushPhase::CommitRecord),
                (transaction_id, JournalFlushPhase::Checkpoint),
                (transaction_id, JournalFlushPhase::TailUpdate),
            ]
        );
        let write_order = device.write_order.lock();
        let data_index = write_order
            .iter()
            .position(|block| *block == 50)
            .expect("data write must be recorded");
        let metadata_index = write_order
            .iter()
            .position(|block| *block == 42)
            .expect("metadata checkpoint must be recorded");
        assert!(data_index < metadata_index);
        device.crash();
        assert_eq!(
            device.stable_block(50).unwrap().as_slice(),
            &[0xD0; BLOCK_SIZE]
        );
        assert_eq!(
            device.stable_block(42).unwrap().as_slice(),
            &[0xC2; BLOCK_SIZE]
        );
        assert_eq!(
            device.stable_block(43).unwrap().as_slice(),
            &[0xE1; BLOCK_SIZE]
        );
    }

    #[test]
    fn metadata_without_pending_writeback_keeps_direct_path_available() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context()).unwrap();

        // Given no deferred writeback metadata exists.
        assert!(core.start_if_deferred(1).unwrap().is_none());

        // When the caller falls back to its normal direct metadata path.
        let mut direct = core.start(1).unwrap();
        direct.stage(42, Box::new([0xD1; BLOCK_SIZE])).unwrap();
        direct.commit(&device, &publisher).unwrap();

        // Then it still commits normally with the complete JBD2 barrier set.
        assert_eq!(
            &*device.journal_commits.lock(),
            &[(7, 1, JournalCommitReason::Explicit)]
        );
        assert_eq!(device.flushes.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn empty_deferred_writeback_never_creates_a_metadata_barrier_commit() {
        let device = MemoryDevice::new();
        let publisher = Publisher(AtomicUsize::new(0));
        let core = JournalTransactionCore::new(context()).unwrap();

        // Given a writeback preparation that did not change any journal home.
        core.start(1)
            .unwrap()
            .defer_or_commit(&device, &publisher, TEST_MAX_FAST_METADATA_IMAGES)
            .unwrap();

        // When a later metadata operation reaches its durability boundary.
        let committed = core
            .flush_deferred_transaction(
                &device,
                &publisher,
                JournalCommitReason::DirectMetadataBarrier,
            )
            .unwrap();

        // Then the empty batch cannot manufacture a four-phase journal commit.
        assert!(!committed);
        assert!(!core.has_pending_transaction());
        assert_eq!(device.flushes.load(Ordering::SeqCst), 0);
        assert!(device.journal_commits.lock().is_empty());
        assert!(device.journal_flushes.lock().is_empty());
    }

    #[test]
    fn encodes_escape_and_wraps_ring_without_losing_home_image() {
        let device = MemoryDevice::new();
        let core = JournalTransactionCore::new(context()).unwrap();
        let publisher = Publisher(AtomicUsize::new(0));
        let mut transaction = core.start(1).unwrap();
        let mut image = Box::new([0x5a; BLOCK_SIZE]);
        image[..4].copy_from_slice(&MAGIC.to_be_bytes());
        transaction.stage(42, image).unwrap();
        transaction.commit(&device, &publisher).unwrap();

        // head=7: descriptor at 7, escaped data wraps to 1, commit at 2.
        assert_eq!(&device.stable_block(101).unwrap()[..4], &[0, 0, 0, 0]);
        assert_eq!(&device.stable_block(42).unwrap()[..4], &MAGIC.to_be_bytes());
        assert_eq!(publisher.0.load(Ordering::SeqCst), 1);
        assert!(!core.is_poisoned());
    }

    #[test]
    fn commit_flush_failure_never_publishes_or_checkpoints() {
        let device = MemoryDevice::new();
        // active-SB + descriptor/data writes share the first flush, followed
        // by the commit-record write; fail its flush (zero-based operation 5).
        device.fail_at.store(5, Ordering::SeqCst);
        let core = JournalTransactionCore::new(context()).unwrap();
        let publisher = Publisher(AtomicUsize::new(0));
        let mut transaction = core.start(1).unwrap();
        transaction.stage(42, Box::new([0x33; BLOCK_SIZE])).unwrap();
        let error = transaction.commit(&device, &publisher).unwrap_err();
        assert_eq!(error.failure, CommitFailure::CommitUncertain);
        assert!(core.is_poisoned());
        assert_eq!(publisher.0.load(Ordering::SeqCst), 0);
        device.crash();
        assert!(device.stable_block(42).is_none());
    }

    #[test]
    fn staging_is_deduplicated_and_reads_its_latest_write() {
        let device = MemoryDevice::new();
        let core = JournalTransactionCore::new(context()).unwrap();
        let mut transaction = core.start(1).unwrap();
        transaction.stage(42, Box::new([1; BLOCK_SIZE])).unwrap();
        transaction.stage(42, Box::new([2; BLOCK_SIZE])).unwrap();
        assert_eq!(transaction.read(&device, 42).unwrap()[0], 2);
        transaction.abort();
        assert!(core.start(1).is_ok());
    }
}
