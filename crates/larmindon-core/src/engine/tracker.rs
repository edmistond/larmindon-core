use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::SegmentUpdate;

/// Remaps engine-local segment ids to globally unique ids.
///
/// Engine instances restart their id counters whenever they're recreated, but
/// the frontend transcript persists across sessions (and engine switches), so
/// the ids it sees must never collide. The global counter lives on the
/// `AudioEngine` and survives sessions; each processing thread gets a fresh
/// tracker holding a handle to it.
pub struct SegmentTracker {
    /// engine-local id -> global id, for segments that are still open.
    open: HashMap<u64, u64>,
    next_global: Arc<AtomicU64>,
}

impl SegmentTracker {
    pub fn new(next_global: Arc<AtomicU64>) -> Self {
        Self {
            open: HashMap::new(),
            next_global,
        }
    }

    /// Rewrite `update.segment_id` to its global id, allocating one the first
    /// time an engine-local id is seen. Finalizing an id releases the mapping
    /// (engines never update a segment after finalizing it).
    pub fn remap(&mut self, mut update: SegmentUpdate) -> SegmentUpdate {
        let global = if update.is_final {
            self.open.remove(&update.segment_id)
        } else {
            self.open.get(&update.segment_id).copied()
        };
        let global = global.unwrap_or_else(|| {
            let id = self.next_global.fetch_add(1, Ordering::Relaxed);
            if !update.is_final {
                self.open.insert(update.segment_id, id);
            }
            id
        });
        update.segment_id = global;
        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(segment_id: u64, is_final: bool) -> SegmentUpdate {
        SegmentUpdate {
            segment_id,
            text: String::new(),
            is_final,
        }
    }

    #[test]
    fn transient_updates_keep_their_global_id_until_final() {
        let counter = Arc::new(AtomicU64::new(100));
        let mut tracker = SegmentTracker::new(counter);

        assert_eq!(tracker.remap(update(0, false)).segment_id, 100);
        assert_eq!(tracker.remap(update(0, false)).segment_id, 100);
        assert_eq!(tracker.remap(update(0, true)).segment_id, 100);
        // Engine reusing local id 0 after finalizing gets a fresh global id.
        assert_eq!(tracker.remap(update(0, false)).segment_id, 101);
    }

    #[test]
    fn already_final_segments_allocate_without_retaining() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut tracker = SegmentTracker::new(counter.clone());

        assert_eq!(tracker.remap(update(0, true)).segment_id, 0);
        assert_eq!(tracker.remap(update(1, true)).segment_id, 1);
        assert!(tracker.open.is_empty());
    }

    #[test]
    fn counter_survives_across_trackers() {
        let counter = Arc::new(AtomicU64::new(0));
        let mut first = SegmentTracker::new(counter.clone());
        first.remap(update(0, true));

        let mut second = SegmentTracker::new(counter);
        assert_eq!(second.remap(update(0, true)).segment_id, 1);
    }
}
