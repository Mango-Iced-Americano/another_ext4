use crate::constants::*;
use crate::prelude::*;
use core::any::Any;

/// Interface for serializing and deserializing objects to and from bytes.
///
/// # Safety
///
/// This trait is unsafe because it allows arbitrary memory interpretation.
/// Implementors should guarantee the object is saved in the way defined by
/// functions `from_bytes` and `to_bytes`.
pub unsafe trait AsBytes
where
    Self: Sized,
{
    /// Default implementation that deserializes the object from a byte array.
    fn from_bytes(bytes: &[u8]) -> Self {
        unsafe { core::ptr::read(bytes.as_ptr() as *const Self) }
    }
    /// Default implementation that serializes the object to a byte array.
    fn to_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }
}

/// Common data block descriptor.
#[derive(Debug, Clone)]
pub struct Block {
    /// Physical block id
    pub id: PBlockId,
    /// Raw block data
    pub data: Box<[u8; BLOCK_SIZE]>,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            id: 0,
            data: Box::new([0; BLOCK_SIZE]),
        }
    }
}

impl Block {
    /// Create new block with given physical block id and data.
    pub fn new(block_id: PBlockId, data: Box<[u8; BLOCK_SIZE]>) -> Self {
        Self { id: block_id, data }
    }

    /// Read `size` bytes from `offset` in block data.
    pub fn read_offset(&self, offset: usize, size: usize) -> &[u8] {
        &self.data[offset..offset + size]
    }

    /// Read bytes from `offset` in block data and interpret it as `T`.
    pub fn read_offset_as<T>(&self, offset: usize) -> T
    where
        T: AsBytes,
    {
        T::from_bytes(&self.data[offset..])
    }

    /// Write block data to `offset` with `size`.
    pub fn write_offset(&mut self, offset: usize, data: &[u8]) {
        self.data[offset..offset + data.len()].copy_from_slice(data);
    }

    /// Transform `T` to bytes and write it to `offset`.
    pub fn write_offset_as<T>(&mut self, offset: usize, value: &T)
    where
        T: AsBytes,
    {
        self.write_offset(offset, value.to_bytes());
    }
}

/// Why a journal transaction reached its commit point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JournalCommitReason {
    /// Deferred metadata reached the bounded transaction size.
    DeferredThreshold,
    /// A legacy/direct metadata mutation required a stable extent map.
    DirectMetadataBarrier,
    /// `fsync`/`sync` requested a durability boundary.
    DurabilityBoundary,
    /// Clean unmount is clearing the journal recovery state.
    Shutdown,
    /// A non-writeback metadata operation committed synchronously.
    Explicit,
}

/// The durable barrier phase inside one JBD2 commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JournalFlushPhase {
    /// The active journal superblock and descriptor/data payload are durable.
    ActiveLog,
    /// The commit record is durable.
    CommitRecord,
    /// Checkpointed home metadata blocks are durable.
    Checkpoint,
    /// The clean journal tail update is durable.
    TailUpdate,
}

/// Common interface for block devices.
pub trait BlockDevice: Send + Sync + Any {
    /// Read a block from disk.
    fn read_block(&self, block_id: PBlockId) -> Result<Block>;
    /// Write a block to disk.
    fn write_block(&self, block: &Block) -> Result<()>;
    /// Read a contiguous physical block range into an exact-sized buffer.
    /// Devices may override this to submit fewer, larger requests.
    fn read_blocks(&self, start: PBlockId, data: &mut [u8]) -> Result<()> {
        if data.is_empty() || data.len() % BLOCK_SIZE != 0 {
            return Err(Ext4Error::new(ErrCode::EINVAL));
        }
        for (index, chunk) in data.chunks_exact_mut(BLOCK_SIZE).enumerate() {
            let block_id = start
                .checked_add(index as PBlockId)
                .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            let block = self.read_block(block_id)?;
            chunk.copy_from_slice(&block.data[..]);
        }
        Ok(())
    }
    /// Write an exact-sized buffer to a contiguous physical block range.
    /// The default preserves the existing one-block-at-a-time semantics.
    fn write_blocks(&self, start: PBlockId, data: &[u8]) -> Result<()> {
        if data.is_empty() || data.len() % BLOCK_SIZE != 0 {
            return Err(Ext4Error::new(ErrCode::EINVAL));
        }
        for (index, chunk) in data.chunks_exact(BLOCK_SIZE).enumerate() {
            let block_id = start
                .checked_add(index as PBlockId)
                .ok_or_else(|| Ext4Error::new(ErrCode::EFBIG))?;
            let mut image = Box::new([0; BLOCK_SIZE]);
            image.copy_from_slice(chunk);
            self.write_block(&Block::new(block_id, image))?;
        }
        Ok(())
    }
    /// Make all writes completed before this call durable on stable storage.
    ///
    /// A successful return must not be used for durability unless
    /// [`Self::supports_reliable_flush`] is also true.
    fn flush(&self) -> Result<()>;
    /// Whether [`Self::flush`] provides a power-loss durability barrier.
    fn supports_reliable_flush(&self) -> bool;

    /// Whether the embedding kernel has enabled bounded diagnostic sampling.
    fn diagnostic_enabled(&self) -> bool {
        false
    }

    /// Returns the embedding kernel's monotonic cycle counter when diagnostics are enabled.
    fn diagnostic_cycles(&self) -> usize {
        0
    }

    /// Record one successfully durable journal transaction in the embedding kernel.
    fn record_journal_commit(&self, _bytes: usize) {}

    /// Record successful data-page persistence during writeback.
    fn record_writeback_data_write(&self, _bytes: usize, _cycles: usize) {}

    /// Record allocation and extent preparation excluding the data write itself.
    fn record_writeback_alloc_extent(&self, _pages: usize, _cycles: usize) {}

    /// Record a successful journal transaction with its durable reason.
    fn record_writeback_journal_commit(
        &self,
        _transaction_id: u32,
        _staged_blocks: usize,
        _cycles: usize,
        _reason: JournalCommitReason,
    ) {
    }

    /// Record one successful durable barrier of a journal transaction.
    fn record_writeback_journal_flush(
        &self,
        _transaction_id: u32,
        _phase: JournalFlushPhase,
        _cycles: usize,
    ) {
    }

    /// Record a raw durability-boundary flush outside a journal transaction.
    fn record_writeback_flush_boundary(&self, _reason: JournalCommitReason, _cycles: usize) {}
}

#[cfg(test)]
mod block_device_tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct FlushDevice {
        reliable: bool,
        error: Option<ErrCode>,
    }

    impl BlockDevice for FlushDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            Ok(Block::new(block_id, Box::new([0; BLOCK_SIZE])))
        }

        fn write_block(&self, _block: &Block) -> Result<()> {
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            self.error.map_or(Ok(()), |code| Err(Ext4Error::new(code)))
        }

        fn supports_reliable_flush(&self) -> bool {
            self.reliable
        }
    }

    #[test]
    fn flush_capability_is_independent_from_method_presence() {
        let device: &dyn BlockDevice = &FlushDevice {
            reliable: false,
            error: None,
        };
        assert!(!device.supports_reliable_flush());
        assert!(device.flush().is_ok());
    }

    #[test]
    fn flush_error_is_preserved_through_trait_dispatch() {
        let device: &dyn BlockDevice = &FlushDevice {
            reliable: true,
            error: Some(ErrCode::EIO),
        };
        assert!(device.supports_reliable_flush());
        assert_eq!(device.flush().unwrap_err().code(), ErrCode::EIO);
    }

    struct CountingDevice {
        reads: AtomicUsize,
        writes: AtomicUsize,
    }

    impl BlockDevice for CountingDevice {
        fn read_block(&self, block_id: PBlockId) -> Result<Block> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(Block::new(block_id, Box::new([block_id as u8; BLOCK_SIZE])))
        }

        fn write_block(&self, _block: &Block) -> Result<()> {
            self.writes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }

        fn supports_reliable_flush(&self) -> bool {
            true
        }
    }

    #[test]
    fn contiguous_defaults_preserve_block_semantics() {
        let device = CountingDevice {
            reads: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
        };
        let mut read = [0u8; BLOCK_SIZE * 3];
        device.read_blocks(7, &mut read).unwrap();
        assert_eq!(device.reads.load(Ordering::Relaxed), 3);
        assert!(read[..BLOCK_SIZE].iter().all(|byte| *byte == 7));
        assert!(read[BLOCK_SIZE..BLOCK_SIZE * 2]
            .iter()
            .all(|byte| *byte == 8));
        device.write_blocks(7, &read).unwrap();
        assert_eq!(device.writes.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn contiguous_defaults_reject_invalid_lengths() {
        let device = CountingDevice {
            reads: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
        };
        assert_eq!(
            device.write_blocks(0, &[]).unwrap_err().code(),
            ErrCode::EINVAL
        );
        assert_eq!(
            device
                .write_blocks(0, &[0; BLOCK_SIZE - 1])
                .unwrap_err()
                .code(),
            ErrCode::EINVAL
        );
    }
}
