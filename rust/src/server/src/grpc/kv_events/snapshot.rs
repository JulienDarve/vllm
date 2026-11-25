// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The blocks one rank's device cache holds, folded from its KV event batches, and the
//! snapshots of them.

use std::collections::BTreeMap;
use std::sync::Arc;

use prost::Message as _;
use prost::encoding::{decode_varint, encode_varint};
use rustc_hash::FxHashMap;

use super::ledger::Ledger;
use super::pb;

/// The medium of the device cache, the only tier tracked.
const DEVICE_MEDIUM: &str = "GPU";
/// Whether a second store of a held block adds a copy with a remove of its own: the engine's
/// block pool caches duplicate blocks under one hash and publishes one remove per copy.
const COUNTS_COPIES: bool = true;
/// Encoded size at which a snapshot response is sent. Arbitrary; gRPC's default limit is 4 MiB.
const RESPONSE_BYTES: usize = 1 << 20;
/// The most blocks and token ids in one store of a snapshot, which keep a store well under
/// `RESPONSE_BYTES`. Arbitrary.
const STORE_BLOCKS: usize = 1024;
const STORE_TOKEN_IDS: usize = 1 << 16;

/// Where blocks are held, as a remove names it besides its medium:
/// `(group_idx, locality, ownership)`.
type Place = (Option<u32>, Option<String>, Option<String>);

fn place(group_idx: Option<u32>, locality: &Option<String>, ownership: &Option<String>) -> Place {
    (group_idx, locality.clone(), ownership.clone())
}

/// What a store said about one of its blocks.
struct StoredBlock {
    hash: pb::BlockHash,
    token_ids: TokenIds,
    /// `None` when the store has no lookahead tokens.
    lookahead_token_ids: Option<TokenIds>,
    /// Everything else; see [`rest_of`].
    rest: Arc<pb::BlockStored>,
}

/// Token ids as varints: most take two or three bytes.
struct TokenIds(Box<[u8]>);

impl TokenIds {
    fn new(ids: &[u32]) -> Self {
        let mut bytes = Vec::new();
        ids.iter().for_each(|&id| encode_varint(id.into(), &mut bytes));
        Self(bytes.into())
    }

    fn to_vec(&self) -> Vec<u32> {
        let mut bytes = &*self.0;
        std::iter::from_fn(|| decode_varint(&mut bytes).ok())
            .map(|id| id as u32)
            .collect()
    }
}

#[derive(Clone)]
struct Held {
    block: Arc<StoredBlock>,
    /// How many blocks of the cache hold it.
    copies: u32,
}

/// The blocks one rank's device cache holds, as of the batch before `next`.
#[derive(Default)]
pub(super) struct Blocks {
    places: BTreeMap<Place, Ledger<Held>>,
    rests: Rests,
    next: u64,
    /// Batches were lost since the cache was last empty: the blocks may lack some that the
    /// cache holds and hold some that it dropped.
    incomplete: bool,
}

impl Blocks {
    /// Applies the next batch of the stream. One not numbered `next` follows lost batches;
    /// the first follows any the engine published before the server connected.
    pub fn apply(&mut self, batch: &pb::KvEventBatch) {
        use pb::kv_cache_event::Event;

        self.incomplete |= batch.sequence_number != self.next;
        for event in &batch.events {
            match &event.event {
                Some(Event::BlockStored(stored)) => self.store(stored),
                Some(Event::BlockRemoved(removed)) => self.remove(removed),
                Some(Event::AllBlocksCleared(_)) => {
                    self.places.clear();
                    self.incomplete = false;
                }
                None => {}
            }
        }
        self.next = batch.sequence_number + 1;
    }

    /// The responses of a snapshot of the blocks held now, `end` last. Taking it copies the
    /// ledgers and shares the blocks, so it can be done under the lock that `apply` takes;
    /// the responses walk every block, so read them off the threads that serve requests.
    pub fn snapshot(&self) -> impl Iterator<Item = pb::GetKvSnapshotResponse> + Send + use<> {
        use pb::get_kv_snapshot_response::Item;

        let ledgers: Vec<_> = self.places.values().cloned().collect();
        let mut left = ledgers.into_iter().flat_map(stores);
        let mut end = Some(pb::KvSnapshotEnd {
            next_sequence_number: self.next,
            complete: !self.incomplete,
        });
        std::iter::from_fn(move || {
            let mut stored = Vec::new();
            let mut bytes = 0;
            while bytes < RESPONSE_BYTES
                && let Some(store) = left.next()
            {
                bytes += store.encoded_len();
                stored.push(store);
            }
            let item = if stored.is_empty() {
                Item::End(end.take()?)
            } else {
                Item::Blocks(pb::KvSnapshotBlocks { stored })
            };
            Some(pb::GetKvSnapshotResponse { item: Some(item) })
        })
    }

    fn store(&mut self, stored: &pb::BlockStored) {
        if stored.reuse_report || stored.medium.as_deref() != Some(DEVICE_MEDIUM) {
            return;
        }
        let size = stored.block_size as usize;
        // The engine leaves null and masked blocks out of the hashes but not out of the
        // tokens: such a store's tokens cannot be told apart per block.
        if size == 0 || stored.token_ids.len() != stored.block_hashes.len() * size {
            return;
        }

        let place = place(stored.group_idx, &stored.locality, &stored.ownership);
        let ledger = self.places.entry(place).or_default();
        // The same for every block of the store, unless they have entries of their own.
        let per_block = !stored.extra_keys.is_empty();
        let mut of_store: Option<Arc<pb::BlockStored>> = None;
        let mut parent = stored.parent_block_hash.as_ref().map(hash_key);
        for (at, hash) in stored.block_hashes.iter().enumerate() {
            let key = hash_key(hash);
            match ledger.get_mut(key) {
                Some(held) => held.copies += u32::from(COUNTS_COPIES),
                None => {
                    let rest = match &of_store {
                        Some(rest) if !per_block => rest.clone(),
                        _ => self.rests.shared(rest_of(stored, at)),
                    };
                    of_store = Some(rest.clone());
                    let lookahead = stored.lookahead_token_ids.get(at);
                    let block = StoredBlock {
                        hash: hash.clone(),
                        token_ids: TokenIds::new(&stored.token_ids[at * size..][..size]),
                        lookahead_token_ids: lookahead.map(|ids| TokenIds::new(&ids.ids)),
                        rest,
                    };
                    let held = Held {
                        block: Arc::new(block),
                        copies: 1,
                    };
                    ledger.store(key, held, parent);
                }
            }
            parent = Some(key);
        }
    }

    fn remove(&mut self, removed: &pb::BlockRemoved) {
        if removed.medium.as_deref() != Some(DEVICE_MEDIUM) {
            return;
        }
        let place = place(removed.group_idx, &removed.locality, &removed.ownership);
        let Some(ledger) = self.places.get_mut(&place) else {
            return;
        };
        for hash in &removed.block_hashes {
            let key = hash_key(hash);
            match ledger.get_mut(key) {
                Some(held) if held.copies > 1 => held.copies -= 1,
                _ => {
                    ledger.remove(key);
                }
            }
        }
    }
}

/// A hash as a ledger key. A digest is keyed by its low 64 bits, the integer the engine
/// publishes for it by default.
fn hash_key(hash: &pb::BlockHash) -> u64 {
    use pb::block_hash::Value;

    match &hash.value {
        Some(Value::UnsignedValue(value)) => *value,
        Some(Value::SignedValue(value)) => value.cast_unsigned(),
        Some(Value::Digest(digest)) => {
            let low = &digest[digest.len().saturating_sub(8)..];
            low.iter().fold(0, |key, &byte| key << 8 | u64::from(byte))
        }
        None => 0,
    }
}

/// What `stored` says about its block `at` besides its hash, parent and tokens: the fields
/// of the whole store and the block's entry of `extra_keys`, without the `session_id`,
/// which describes the event.
fn rest_of(stored: &pb::BlockStored, at: usize) -> pb::BlockStored {
    pb::BlockStored {
        block_hashes: Vec::new(),
        parent_block_hash: None,
        token_ids: Vec::new(),
        block_size: stored.block_size,
        lora_id: stored.lora_id,
        medium: stored.medium.clone(),
        lora_name: stored.lora_name.clone(),
        extra_keys: stored.extra_keys.get(at).cloned().into_iter().collect(),
        group_idx: stored.group_idx,
        kv_cache_spec_kind: stored.kv_cache_spec_kind.clone(),
        kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
        locality: stored.locality.clone(),
        ownership: stored.ownership.clone(),
        session_id: None,
        lookahead_token_ids: Vec::new(),
        reuse_report: false,
    }
}

/// Each [`rest_of`] a held block has, kept once however many blocks have the same: most
/// blocks of a cache do, unless their `extra_keys` differ, as those of multimodal blocks do.
#[derive(Default)]
struct Rests {
    by_encoding: FxHashMap<Vec<u8>, Arc<pb::BlockStored>>,
    /// The size at which the entries no block uses are next dropped.
    sweep_at: usize,
}

impl Rests {
    fn shared(&mut self, rest: pb::BlockStored) -> Arc<pb::BlockStored> {
        if self.by_encoding.len() >= self.sweep_at {
            self.by_encoding.retain(|_, rest| Arc::strong_count(rest) > 1);
            // Arbitrary floor; doubling keeps the sweeps amortized.
            self.sweep_at = (2 * self.by_encoding.len()).max(64);
        }
        let rest = self.by_encoding.entry(rest.encode_to_vec()).or_insert_with(|| Arc::new(rest));
        rest.clone()
    }
}

/// The ledger's blocks as stores, each after the store that holds its parent.
fn stores(ledger: Ledger<Held>) -> impl Iterator<Item = pb::BlockStored> + Send {
    ledger.chains().into_iter().flat_map(move |chain| {
        let parent = chain.parent.and_then(|parent| ledger.get(parent));
        chain_stores(parent.map(|held| held.block.hash.clone()), &chain.blocks)
    })
}

/// One chain as stores: consecutive blocks with the same [`rest_of`] share a store, and each
/// further copy of a block is a store of its own, after the chain's.
fn chain_stores(mut parent: Option<pb::BlockHash>, blocks: &[(u64, Held)]) -> Vec<pb::BlockStored> {
    let mut stores: Vec<pb::BlockStored> = Vec::new();
    let mut copies = Vec::new();
    let mut last: Option<&StoredBlock> = None;
    for (_, held) in blocks {
        let block = &*held.block;
        // A store has lookahead tokens for all of its blocks or for none.
        let joins = last.is_some_and(|last| {
            Arc::ptr_eq(&last.rest, &block.rest)
                && last.lookahead_token_ids.is_some() == block.lookahead_token_ids.is_some()
        });
        let fits = |store: &pb::BlockStored| {
            let blocks = store.block_hashes.len();
            blocks < STORE_BLOCKS && (blocks + 1) * store.block_size as usize <= STORE_TOKEN_IDS
        };
        match stores.last_mut() {
            Some(store) if joins && fits(store) => append(store, block),
            _ => stores.push(store_of(parent.clone(), block)),
        }
        copies.extend((1..held.copies).map(|_| store_of(parent.clone(), block)));
        parent = Some(block.hash.clone());
        last = Some(block);
    }
    stores.extend(copies);
    stores
}

/// A store of `block` alone. Its `rest` holds the block's own entry of `extra_keys`, which
/// `append` puts back.
fn store_of(parent: Option<pb::BlockHash>, block: &StoredBlock) -> pb::BlockStored {
    let mut store = pb::BlockStored {
        parent_block_hash: parent,
        extra_keys: Vec::new(),
        ..(*block.rest).clone()
    };
    append(&mut store, block);
    store
}

/// Adds `block` to the end of a store whose blocks have its `rest`.
fn append(store: &mut pb::BlockStored, block: &StoredBlock) {
    store.block_hashes.push(block.hash.clone());
    store.token_ids.extend(block.token_ids.to_vec());
    store.extra_keys.extend(block.rest.extra_keys.iter().cloned());
    let lookahead = block.lookahead_token_ids.as_ref();
    store
        .lookahead_token_ids
        .extend(lookahead.map(|ids| pb::TokenIds { ids: ids.to_vec() }));
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::pb::kv_cache_event::Event;
    use super::*;

    struct Seeded(u64);

    impl Seeded {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }

        fn one_of(&mut self, of: &[u64]) -> Option<u64> {
            (!of.is_empty()).then(|| of[self.below(of.len() as u64) as usize])
        }
    }

    fn hash(value: u64) -> pb::BlockHash {
        pb::BlockHash {
            value: Some(pb::block_hash::Value::UnsignedValue(value)),
        }
    }

    /// A store of one to three new blocks after `parent`, with or without each optional part.
    fn new_store(rng: &mut Seeded, last_hash: &mut u64, parent: Option<u64>) -> pb::BlockStored {
        let blocks = 1 + rng.below(3) as usize;
        let size = 1 + rng.below(2) as usize;
        // Some take one byte as a varint, some five.
        let id = |rng: &mut Seeded| (rng.below(3) << (30 * rng.below(2))) as u32;
        let (keys, lookahead) = (rng.below(3) == 0, rng.below(4) == 0);
        let block_hashes = (0..blocks).map(|_| {
            *last_hash += 1;
            hash(*last_hash)
        });
        pb::BlockStored {
            block_hashes: block_hashes.collect(),
            parent_block_hash: parent.map(hash),
            token_ids: (0..blocks * size).map(|_| id(rng)).collect(),
            block_size: size as u32,
            medium: Some(DEVICE_MEDIUM.to_string()),
            lora_name: (rng.below(3) == 0).then(|| "adapter".to_string()),
            extra_keys: (0..blocks).filter(|_| keys).map(|_| extra_key(rng.below(2))).collect(),
            lookahead_token_ids: (0..blocks)
                .filter(|_| lookahead)
                .map(|_| pb::TokenIds {
                    ids: vec![rng.below(9) as u32],
                })
                .collect(),
            ..Default::default()
        }
    }

    fn extra_key(value: u64) -> pb::BlockExtraKeys {
        pb::BlockExtraKeys {
            keys: vec![pb::ExtraKeyValue {
                value: Some(pb::extra_key_value::Value::Integer(value as i64)),
            }],
        }
    }

    /// One store per block of `store`, saying of that block what `store` says, without the
    /// session.
    fn per_block(store: &pb::BlockStored) -> Vec<pb::BlockStored> {
        let size = store.block_size as usize;
        let mut parent = store.parent_block_hash.clone();
        let rest = pb::BlockStored {
            block_hashes: Vec::new(),
            token_ids: Vec::new(),
            session_id: None,
            ..store.clone()
        };
        let blocks = store.block_hashes.iter().enumerate().map(|(at, hash)| pb::BlockStored {
            block_hashes: vec![hash.clone()],
            parent_block_hash: parent.replace(hash.clone()),
            token_ids: store.token_ids[at * size..][..size].to_vec(),
            extra_keys: store.extra_keys.get(at).cloned().into_iter().collect(),
            lookahead_token_ids: (store.lookahead_token_ids.get(at).cloned().into_iter()).collect(),
            ..rest.clone()
        });
        blocks.collect()
    }

    /// The stores of a snapshot of `blocks`, in order. Checks that the stores and the
    /// responses keep to their limits.
    fn snapshot_stores(blocks: &Blocks) -> Vec<pb::BlockStored> {
        use pb::get_kv_snapshot_response::Item;

        let stores = blocks.snapshot().filter_map(|response| {
            assert!(response.encoded_len() < 2 * RESPONSE_BYTES);
            match response.item {
                Some(Item::Blocks(blocks)) => Some(blocks.stored),
                _ => None,
            }
        });
        let stores: Vec<_> = stores.flatten().collect();
        for store in &stores {
            let most = STORE_BLOCKS.min(STORE_TOKEN_IDS / store.block_size as usize);
            assert!(store.block_hashes.len() <= most);
        }
        stores
    }

    /// What a snapshot of `blocks` says, as one store per block and copy, by hash. Checks
    /// that each block follows its parent.
    fn snapshot(blocks: &Blocks) -> Vec<pb::BlockStored> {
        let mut said: Vec<_> = snapshot_stores(blocks).iter().flat_map(per_block).collect();
        for (at, block) in said.iter().enumerate() {
            let parent = block.parent_block_hash.as_ref();
            assert!(
                parent.is_none_or(|parent| said[..at].iter().any(|b| &b.block_hashes[0] == parent))
            );
        }
        said.sort_by_key(|block| hash_key(&block.block_hashes[0]));
        said
    }

    fn batch(sequence_number: u64, events: impl IntoIterator<Item = Event>) -> pb::KvEventBatch {
        pb::KvEventBatch {
            sequence_number,
            events: (events.into_iter().map(|event| pb::KvCacheEvent { event: Some(event) }))
                .collect(),
            ..Default::default()
        }
    }

    /// Over seeded stores, removes and clears, a snapshot says what a plain map of the held
    /// blocks says, and a snapshot followed by the next hundred batches gives the state that
    /// every batch gives. Removes take leaves only: a block whose parent was removed is left
    /// out of a snapshot, so a state loaded from one cannot get it back when the parent
    /// returns.
    #[test]
    fn a_snapshot_then_the_later_batches_equals_every_batch() {
        let mut rng = Seeded(0x9E37_79B9_7F4A_7C15);
        // Each held block as one store, with how many copies are held.
        let mut held: BTreeMap<u64, (pb::BlockStored, u32)> = BTreeMap::new();
        let mut blocks = Blocks::default();
        let mut from_snapshot: Option<Blocks> = None;
        let mut last_hash = 0;

        for sequence_number in 0..2000 {
            let hashes: Vec<u64> = held.keys().copied().collect();
            let parents: BTreeSet<u64> = (held.values())
                .filter_map(|(block, _)| block.parent_block_hash.as_ref().map(hash_key))
                .collect();
            let leaves: Vec<u64> =
                hashes.iter().copied().filter(|hash| !parents.contains(hash)).collect();
            let roll = rng.below(20);
            let leaf = rng.one_of(&leaves).filter(|_| roll < 7);
            // Rarely: a clear makes every state equal.
            let event = if rng.below(400) == 0 {
                held.clear();
                Event::AllBlocksCleared(pb::AllBlocksCleared {})
            } else if let Some(leaf) = leaf {
                let copies = &mut held.get_mut(&leaf).expect("a held block").1;
                *copies -= 1;
                if *copies == 0 {
                    held.remove(&leaf);
                }
                Event::BlockRemoved(pb::BlockRemoved {
                    block_hashes: vec![hash(leaf)],
                    medium: Some(DEVICE_MEDIUM.to_string()),
                    ..Default::default()
                })
            } else {
                // A held block again, or new blocks after a held block or none.
                let again = rng.one_of(&hashes).filter(|_| roll == 7);
                let mut store = match again {
                    Some(again) => held[&again].0.clone(),
                    None => {
                        let parent = rng.one_of(&hashes).filter(|_| rng.below(4) != 0);
                        new_store(&mut rng, &mut last_hash, parent)
                    }
                };
                store.session_id = Some("session".to_string());
                match roll {
                    8 => store.medium = Some("CPU".to_string()),
                    9 => store.reuse_report = true,
                    // Its tokens no longer divide among its blocks.
                    10 => store.block_size += 1,
                    _ if again.is_none() || COUNTS_COPIES => {
                        for block in per_block(&store) {
                            held.entry(hash_key(&block.block_hashes[0])).or_insert((block, 0)).1 +=
                                1;
                        }
                    }
                    _ => {}
                }
                Event::BlockStored(store)
            };

            let batch = batch(sequence_number, [event]);
            blocks.apply(&batch);
            from_snapshot.iter_mut().for_each(|blocks| blocks.apply(&batch));

            if sequence_number % 100 == 99 {
                let expected = held.values().flat_map(|(block, copies)| {
                    std::iter::repeat_n(block.clone(), *copies as usize)
                });
                assert_eq!(snapshot(&blocks), expected.collect::<Vec<_>>());
                if let Some(from_snapshot) = &from_snapshot {
                    assert_eq!(snapshot(from_snapshot), snapshot(&blocks));
                }

                let stores = snapshot_stores(&blocks).into_iter().map(Event::BlockStored);
                let loaded = from_snapshot.insert(Blocks::default());
                loaded.apply(&self::batch(sequence_number, stores));
            }
        }
    }

    /// A chain too long for one store is cut into stores, and stores too large for one
    /// response into responses, that still say what was stored.
    #[test]
    fn a_long_chain_is_cut_into_stores_and_responses() {
        for size in [1, 128] {
            let chain = 4 * STORE_BLOCKS + 1;
            let store = pb::BlockStored {
                block_hashes: (1..=chain as u64).map(hash).collect(),
                // Five bytes each.
                token_ids: vec![u32::MAX; chain * size],
                block_size: size as u32,
                medium: Some(DEVICE_MEDIUM.to_string()),
                ..Default::default()
            };
            let mut blocks = Blocks::default();
            blocks.apply(&batch(0, [Event::BlockStored(store.clone())]));
            assert_eq!(snapshot(&blocks), per_block(&store));
        }
    }
}
