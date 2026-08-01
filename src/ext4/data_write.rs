use super::Ext4;
use crate::constants::*;
use crate::ext4_defs::{Block, ExtentNode, InodeRef};
use crate::format_error;
use crate::prelude::*;
use crate::return_error;
use core::cmp::min;

struct MappedWriteBlock {
    pblock: PBlockId,
    block_offset: usize,
    data_offset: usize,
    write_len: usize,
}

struct PhysicalWriteRun {
    start: PBlockId,
    data: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(super) struct ExtentWriteRange {
    pub(super) start_lblock: LBlockId,
    pub(super) end_lblock: LBlockId,
    pub(super) start_pblock: PBlockId,
}

impl Ext4 {
    /// Like extent_query() but also returns the covering extent range for caching.
    pub(super) fn extent_query_with_range(
        &self,
        inode_ref: &InodeRef,
        iblock: LBlockId,
    ) -> Result<(PBlockId, ExtentWriteRange)> {
        self.prepare_stats.record_extent_query_attempt();
        let start = self.prepare_stats.phase_start(self.block_device.as_ref());

        let result = (|| {
            let path = self.find_extent(inode_ref, iblock)?;
            let leaf = path.last().ok_or(format_error!(
                ErrCode::EIO,
                "extent_query_with_range: empty search path on inode {}",
                inode_ref.id
            ))?;
            let index = leaf.index.map_err(|_| {
                format_error!(
                    ErrCode::ENOENT,
                    "extent_query_with_range: inode {} iblock {} not found",
                    inode_ref.id,
                    iblock
                )
            })?;

            let block_data: Block;
            let extent_node = if leaf.pblock != 0 {
                self.ensure_valid_pblock(inode_ref.id, leaf.pblock, "extent leaf node")?;
                block_data = self.read_extent_block(inode_ref, leaf.pblock)?;
                ExtentNode::from_bytes(&*block_data.data)
            } else {
                inode_ref.inode.extent_root()
            };

            let extent = extent_node.extent_at(index);
            let start_lblock = extent.start_lblock();
            let end_lblock = start_lblock
                .checked_add(extent.block_count())
                .ok_or_else(|| {
                    format_error!(
                        ErrCode::EIO,
                        "extent end overflow on inode {}",
                        inode_ref.id
                    )
                })?;
            if iblock < start_lblock || iblock >= end_lblock {
                return Err(format_error!(
                    ErrCode::ENOENT,
                    "non-covering extent for iblock {} on inode {}",
                    iblock,
                    inode_ref.id
                ));
            }

            let start_pblock = extent.start_pblock();
            let pblock = start_pblock
                .checked_add((iblock - start_lblock) as PBlockId)
                .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;

            self.ensure_valid_pblock(inode_ref.id, pblock, "extent data block")?;
            self.validate_data_blocks(pblock, 1)?;

            Ok((
                pblock,
                ExtentWriteRange {
                    start_lblock,
                    end_lblock,
                    start_pblock,
                },
            ))
        })();

        self.prepare_stats.record_phase(
            super::PreparePhase::ExtentQuery,
            start,
            self.block_device.as_ref(),
        );
        result
    }

    /// Write pre-allocated data without changing inode metadata.
    ///
    /// Every logical block is mapped before data I/O begins. Physical adjacency,
    /// rather than logical adjacency, determines each block-device write run.
    pub fn write_data_only(&self, file: InodeId, offset: usize, data: &[u8]) -> Result<usize> {
        self.ensure_mutable()?;
        let _metadata_guard = self.lock_direct_metadata_mutation()?;
        let _mutation_guard =
            self.inode_mutation_locks[self.inode_mutation_lock_index(file)].lock();
        let file = self.read_inode(file)?;
        self.write_mapped_data(&file, offset, data)
    }

    /// Write data through an inode image already protected by a journal transaction.
    ///
    /// The caller holds the transactional metadata and inode-mutation guards, so
    /// this bypasses the direct-mutation barrier that would otherwise commit a
    /// deferred journal batch before ordinary writeback finishes.
    pub(super) fn write_journaled_mapped_data(
        &self,
        file: &InodeRef,
        offset: usize,
        data: &[u8],
    ) -> Result<usize> {
        self.write_mapped_data(file, offset, data)
    }

    fn write_mapped_data(&self, file: &InodeRef, offset: usize, data: &[u8]) -> Result<usize> {
        let write_size = data.len();
        if write_size == 0 {
            return Ok(0);
        }
        let range = match Self::checked_write_logical_range(offset, write_size)? {
            Some(range) => range,
            None => return Ok(0),
        };
        if !file.inode.is_file() {
            return_error!(ErrCode::EISDIR, "Inode {} is not a file", file.id);
        }
        let mut mapped = Vec::new();
        mapped
            .try_reserve_exact(range.block_count)
            .map_err(|_| Ext4Error::new(ErrCode::ENOMEM))?;

        let mut cursor = 0;
        let mut iblock = range.first_lblock;
        let mut cached_extent: Option<ExtentWriteRange> = None;
        while cursor < write_size {
            let block_offset = (offset + cursor) % BLOCK_SIZE;
            let write_len = min(BLOCK_SIZE - block_offset, write_size - cursor);
            let resolved = match cached_extent {
                Some(ext) if iblock >= ext.start_lblock && iblock < ext.end_lblock => (|| {
                    let pblock = ext
                        .start_pblock
                        .checked_add((iblock - ext.start_lblock) as PBlockId)
                        .ok_or_else(|| Ext4Error::new(ErrCode::EIO))?;
                    self.ensure_valid_pblock(file.id, pblock, "cached extent data block")?;
                    self.validate_data_blocks(pblock, 1)?;
                    Ok(pblock)
                })(
                ),
                _ => self
                    .extent_query_with_range(&file, iblock)
                    .map(|(pblock, ext)| {
                        cached_extent = Some(ext);
                        pblock
                    }),
            };

            let pblock = match resolved {
                Ok(pblock) => pblock,
                Err(error) => {
                    debug!(
                        "write_data_only: extent lookup FAILED ino={} iblock={} offset={} len={} fs_blkcnt={} size={} err={:?}",
                        file.id,
                        iblock,
                        offset,
                        write_size,
                        file.inode.fs_block_count(),
                        file.inode.size(),
                        error
                    );
                    return Err(error);
                }
            };
            mapped.push(MappedWriteBlock {
                pblock,
                block_offset,
                data_offset: cursor,
                write_len,
            });
            cursor += write_len;
            if cursor < write_size {
                iblock = iblock
                    .checked_add(1)
                    .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            }
        }

        let mut runs = Vec::new();
        runs.try_reserve_exact(mapped.len())
            .map_err(|_| Ext4Error::new(ErrCode::ENOMEM))?;
        let mut run_start = 0;
        while run_start < mapped.len() {
            let mut run_end = run_start + 1;
            while run_end < mapped.len()
                && mapped[run_end - 1].pblock.checked_add(1) == Some(mapped[run_end].pblock)
            {
                run_end += 1;
            }

            let run_blocks = run_end - run_start;
            let run_bytes = run_blocks
                .checked_mul(BLOCK_SIZE)
                .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            let mut run_data = Vec::new();
            run_data
                .try_reserve_exact(run_bytes)
                .map_err(|_| Ext4Error::new(ErrCode::ENOMEM))?;
            run_data.resize(run_bytes, 0);
            for (index, mapped_block) in mapped[run_start..run_end].iter().enumerate() {
                let run_block = &mut run_data[index * BLOCK_SIZE..(index + 1) * BLOCK_SIZE];
                if mapped_block.block_offset == 0 && mapped_block.write_len == BLOCK_SIZE {
                    run_block.copy_from_slice(
                        &data[mapped_block.data_offset..mapped_block.data_offset + BLOCK_SIZE],
                    );
                } else {
                    let block = self.read_block(mapped_block.pblock)?;
                    run_block.copy_from_slice(&block.data[..]);
                    run_block[mapped_block.block_offset
                        ..mapped_block.block_offset + mapped_block.write_len]
                        .copy_from_slice(
                            &data[mapped_block.data_offset
                                ..mapped_block.data_offset + mapped_block.write_len],
                        );
                }
            }
            runs.push(PhysicalWriteRun {
                start: mapped[run_start].pblock,
                data: run_data,
            });
            run_start = run_end;
        }

        for run in runs {
            self.block_device.write_blocks(run.start, &run.data)?;
        }
        Ok(write_size)
    }
}
