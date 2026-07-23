//! Bounded advisory cache for successfully prepared positive extents.

use super::{Ext4, WriteLogicalRange};
use crate::prelude::InodeId;

const PREPARED_EXTENT_SLOTS: usize = 64;

#[derive(Clone, Copy)]
struct PreparedExtent {
    inode: InodeId,
    epoch: usize,
    first_lblock: u32,
    last_lblock: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PreparedExtentProbe {
    hit: bool,
    token: usize,
}

pub(super) struct PreparedExtentCache {
    epoch: usize,
    next_slot: usize,
    slots: [Option<PreparedExtent>; PREPARED_EXTENT_SLOTS],
}

impl PreparedExtentCache {
    pub(super) const fn new() -> Self {
        Self {
            epoch: 0,
            next_slot: 0,
            slots: [None; PREPARED_EXTENT_SLOTS],
        }
    }

    fn probe(&self, inode: InodeId, range: &WriteLogicalRange) -> PreparedExtentProbe {
        let hit = self.slots.iter().flatten().any(|entry| {
            entry.inode == inode
                && entry.epoch == self.epoch
                && entry.first_lblock <= range.first_lblock
                && range.last_lblock <= entry.last_lblock
        });
        PreparedExtentProbe {
            hit,
            token: self.epoch,
        }
    }

    fn publish_if_current(
        &mut self,
        inode: InodeId,
        range: &WriteLogicalRange,
        expected_epoch: usize,
    ) -> bool {
        if self.epoch != expected_epoch {
            return false;
        }
        let entry = PreparedExtent {
            inode,
            epoch: self.epoch,
            first_lblock: range.first_lblock,
            last_lblock: range.last_lblock,
        };
        if let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| slot.is_some_and(|current| current.inode == inode))
        {
            *slot = Some(entry);
            return true;
        }
        self.slots[self.next_slot] = Some(entry);
        self.next_slot = (self.next_slot + 1) % PREPARED_EXTENT_SLOTS;
        true
    }

    fn invalidate(&mut self) {
        if self.epoch == usize::MAX {
            self.epoch = 0;
            self.slots.fill(None);
            return;
        }
        self.epoch += 1;
    }
}

impl Ext4 {
    pub(super) fn prepared_extent_probe(
        &self,
        inode: InodeId,
        range: &WriteLogicalRange,
    ) -> PreparedExtentProbe {
        self.prepared_extents.lock().probe(inode, range)
    }

    pub(super) fn prepared_extent_hit(probe: PreparedExtentProbe) -> bool {
        probe.hit
    }

    pub(super) fn prepared_extent_token(probe: PreparedExtentProbe) -> usize {
        probe.token
    }

    pub(super) fn publish_prepared_extent(
        &self,
        inode: InodeId,
        range: &WriteLogicalRange,
        expected_epoch: usize,
    ) -> bool {
        self.prepared_extents
            .lock()
            .publish_if_current(inode, range, expected_epoch)
    }

    pub(super) fn invalidate_prepared_extents(&self) {
        self.prepared_extents.lock().invalidate();
    }
}
