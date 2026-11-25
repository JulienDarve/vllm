// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod ledger;
mod relay;
mod snapshot;

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::Stream;
use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::broadcast::{self, error::RecvError};
use tokio::sync::{OnceCell, mpsc};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tonic::{Request, Response, Status};
use tracing::warn;
use vllm_engine_core_client::protocol::handshake::KvEventsConfig;
use vllm_engine_core_client::protocol::kv_events::{
    BlockRemoved, BlockStored, ExternalBlockHash, ExtraKeyValue, KvCacheEvent, KvEventBatch,
};

use self::relay::{KvEventRelay, RETRY_DELAY};
use self::snapshot::Blocks;
use super::{KvEventsServer, pb};
use crate::state::AppState;

pub(crate) type KvEventsGrpcService = KvEventsServer<KvEventsServiceImpl>;

/// Snapshot responses built ahead of the client reading them. Arbitrary.
const SNAPSHOT_RESPONSES_AHEAD: usize = 2;
/// How long a client may take to read a snapshot response before the snapshot
/// is dropped. Arbitrary.
const SNAPSHOT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// gRPC KV event service: re-publishes each engine's ZMQ KV events as typed
/// streams, and serves snapshots of the blocks they add up to.
pub struct KvEventsServiceImpl {
    /// By data-parallel rank; `None` for an engine that publishes no KV events
    /// to this frontend.
    ranks: BTreeMap<u32, Option<Arc<Publisher>>>,
    /// Ends the streams, which would otherwise hold up a graceful shutdown.
    shutdown: CancellationToken,
    _relay_starts: Vec<AbortOnDropHandle<()>>,
}

/// One engine's KV event publisher, as this frontend serves it.
struct Publisher {
    address: PublisherAddress,
    /// Started with the service when `blocks` are kept, so that they hold the
    /// engine's first batch; otherwise by the first subscriber.
    relay: OnceCell<KvEventRelay>,
    /// The blocks the engine's device cache holds, kept unless the engine's
    /// `grpc_snapshot` is off.
    blocks: Option<Arc<Mutex<Blocks>>>,
    /// Held while a snapshot is sent: each holds a copy of `blocks`.
    one_snapshot: Arc<tokio::sync::Mutex<()>>,
}

impl Publisher {
    async fn relay(&self) -> vllm_engine_core_client::Result<&KvEventRelay> {
        let connect = || KvEventRelay::connect(&self.address, self.blocks.clone());
        self.relay.get_or_try_init(connect).await
    }
}

/// Where an engine publishes its KV events for this frontend.
pub(super) struct PublisherAddress {
    endpoint: String,
    topic: String,
}

impl PublisherAddress {
    /// The private address of an engine that publishes KV events to this
    /// frontend, as its ZMQ publisher reports it bound.
    pub(super) fn of(config: Option<&KvEventsConfig>) -> Option<Self> {
        let config =
            config.filter(|config| config.enable_kv_cache_events && config.publisher == "zmq")?;
        Some(Self {
            endpoint: config.frontend_endpoint.clone()?,
            topic: config.topic.clone(),
        })
    }
}

impl KvEventsServiceImpl {
    pub fn new(state: Arc<AppState>, shutdown: CancellationToken) -> Self {
        let ranks: BTreeMap<_, _> = (state.engine_core_client().ready_responses().into_iter())
            .map(|ready| {
                let config = ready.kv_events_config.as_ref();
                let publisher = PublisherAddress::of(config).map(|address| {
                    let snapshots = config.is_some_and(|config| config.grpc_snapshot);
                    Arc::new(Publisher {
                        address,
                        relay: OnceCell::new(),
                        blocks: snapshots.then(Arc::default),
                        one_snapshot: Arc::default(),
                    })
                });
                (ready.data_parallel_rank, publisher)
            })
            .collect();
        let relay_starts = (ranks.values().flatten())
            .filter(|publisher| publisher.blocks.is_some())
            .map(|publisher| AbortOnDropHandle::new(tokio::spawn(start_relay(publisher.clone()))))
            .collect();
        Self {
            ranks,
            shutdown,
            _relay_starts: relay_starts,
        }
    }

    fn publisher(&self, rank: u32) -> Result<&Publisher, Status> {
        let publisher = self.ranks.get(&rank).ok_or_else(|| {
            Status::invalid_argument(format!(
                "data parallel rank {rank} is not connected to this frontend; \
                 connected ranks: {:?}",
                self.ranks.keys().collect::<Vec<_>>()
            ))
        })?;
        publisher.as_deref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "the engine of rank {rank} does not publish KV cache events; \
                 enable them with --kv-events-config"
            ))
        })
    }
}

/// Connects the relay of a publisher whose blocks are kept, however long the
/// publisher takes to answer.
async fn start_relay(publisher: Arc<Publisher>) {
    while let Err(error) = publisher.relay().await {
        warn!(error = %error.as_report(), "failed to connect to the KV event publisher");
        tokio::time::sleep(RETRY_DELAY).await;
    }
}

#[tonic::async_trait]
impl pb::kv_events_server::KvEvents for KvEventsServiceImpl {
    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;
    type GetKvSnapshotStream =
        Pin<Box<dyn Stream<Item = Result<pb::GetKvSnapshotResponse, Status>> + Send>>;

    async fn subscribe_kv_events(
        &self,
        request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let request = request.into_inner();
        let rank = request.data_parallel_rank;
        // A failed connect is retried by the next subscriber.
        let relay = self.publisher(rank)?.relay().await.map_err(|error| {
            Status::unavailable(format!(
                "failed to connect to the KV event publisher of rank {rank}: {}",
                error.as_report()
            ))
        })?;

        Ok(Response::new(Box::pin(kv_event_responses(
            relay.subscribe(),
            rank,
            self.shutdown.clone(),
        ))))
    }

    async fn get_kv_snapshot(
        &self,
        request: Request<pb::GetKvSnapshotRequest>,
    ) -> Result<Response<Self::GetKvSnapshotStream>, Status> {
        let rank = request.into_inner().data_parallel_rank;
        let publisher = self.publisher(rank)?;
        let blocks = publisher.blocks.clone().ok_or_else(|| {
            Status::failed_precondition(format!(
                "KV cache snapshots of rank {rank} are turned off: `grpc_snapshot` is false in \
                 --kv-events-config"
            ))
        })?;
        if publisher.relay.get().is_none() {
            return Err(Status::unavailable(format!(
                "not connected to the KV event publisher of rank {rank} yet"
            )));
        }

        let one_snapshot = publisher.one_snapshot.clone().lock_owned().await;
        let (responses, received) = mpsc::channel(SNAPSHOT_RESPONSES_AHEAD);
        let runtime = tokio::runtime::Handle::current();
        // The snapshot copies the rank's ledgers and walks every block.
        tokio::task::spawn_blocking(move || {
            let _one_snapshot = one_snapshot;
            let snapshot = blocks.lock().snapshot();
            for response in snapshot {
                // A client that went away or stopped reading must not hold up
                // the rank's next snapshot.
                let sent = responses.send_timeout(response, SNAPSHOT_READ_TIMEOUT);
                if runtime.block_on(sent).is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(kv_snapshot_responses(
            received,
            self.shutdown.clone(),
        ))))
    }
}

#[try_stream]
async fn kv_event_responses(
    mut live: broadcast::Receiver<Arc<pb::KvEventBatch>>,
    data_parallel_rank: u32,
    shutdown: CancellationToken,
    mut y: TryYielder<pb::SubscribeKvEventsResponse, Status>,
) -> Result<(), Status> {
    use pb::subscribe_kv_events_response::Item;

    loop {
        let batch = tokio::select! {
            _ = shutdown.cancelled() => {
                return Err(Status::unavailable("the server is shutting down"));
            }
            batch = live.recv() => batch,
        };
        let batch = match batch {
            Ok(batch) => batch,
            // The subscriber sees the skipped batches as a gap in sequence numbers.
            Err(RecvError::Lagged(skipped)) => {
                warn!(data_parallel_rank, skipped, "KV event subscriber lagged");
                continue;
            }
            Err(RecvError::Closed) => {
                return Err(Status::unavailable("the KV event relay stopped"));
            }
        };
        y.yield_ok(pb::SubscribeKvEventsResponse {
            item: Some(Item::Batch(Arc::unwrap_or_clone(batch))),
        })
        .await;
    }
}

#[try_stream]
async fn kv_snapshot_responses(
    mut responses: mpsc::Receiver<pb::GetKvSnapshotResponse>,
    shutdown: CancellationToken,
    mut y: TryYielder<pb::GetKvSnapshotResponse, Status>,
) -> Result<(), Status> {
    loop {
        let response = tokio::select! {
            _ = shutdown.cancelled() => {
                return Err(Status::unavailable("the server is shutting down"));
            }
            response = responses.recv() => response,
        };
        let Some(response) = response else {
            return Ok(());
        };
        y.yield_ok(response).await;
    }
}

fn batch_to_proto(sequence_number: u64, batch: KvEventBatch) -> pb::KvEventBatch {
    pb::KvEventBatch {
        sequence_number,
        ts: batch.ts,
        data_parallel_rank: batch.data_parallel_rank,
        events: batch.events.into_iter().map(event_to_proto).collect(),
    }
}

fn event_to_proto(event: KvCacheEvent) -> pb::KvCacheEvent {
    use pb::kv_cache_event::Event;

    let event = match event {
        KvCacheEvent::BlockStored(stored) => Event::BlockStored(stored_to_proto(stored)),
        KvCacheEvent::BlockRemoved(removed) => Event::BlockRemoved(removed_to_proto(removed)),
        KvCacheEvent::AllBlocksCleared => Event::AllBlocksCleared(pb::AllBlocksCleared {}),
    };
    pb::KvCacheEvent { event: Some(event) }
}

fn stored_to_proto(stored: BlockStored) -> pb::BlockStored {
    let extra_keys =
        stored
            .extra_keys
            .unwrap_or_default()
            .into_iter()
            .map(|keys| pb::BlockExtraKeys {
                keys: keys.unwrap_or_default().into_iter().map(extra_key_to_proto).collect(),
            });

    pb::BlockStored {
        block_hashes: stored.block_hashes.into_iter().map(hash_to_proto).collect(),
        parent_block_hash: stored.parent_block_hash.map(hash_to_proto),
        token_ids: stored.token_ids,
        block_size: stored.block_size,
        lora_id: stored.lora_id,
        medium: stored.medium,
        lora_name: stored.lora_name,
        extra_keys: extra_keys.collect(),
        group_idx: stored.group_idx,
        kv_cache_spec_kind: stored.kv_cache_spec_kind,
        kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
        locality: stored.locality,
        ownership: stored.ownership,
        session_id: stored.session_id,
        lookahead_token_ids: (stored.lookahead_token_ids.unwrap_or_default().into_iter())
            .map(|ids| pb::TokenIds { ids })
            .collect(),
        reuse_report: stored.reuse_report,
    }
}

fn removed_to_proto(removed: BlockRemoved) -> pb::BlockRemoved {
    pb::BlockRemoved {
        block_hashes: removed.block_hashes.into_iter().map(hash_to_proto).collect(),
        medium: removed.medium,
        group_idx: removed.group_idx,
        locality: removed.locality,
        ownership: removed.ownership,
    }
}

fn hash_to_proto(hash: ExternalBlockHash) -> pb::BlockHash {
    use pb::block_hash::Value;

    let value = match hash {
        ExternalBlockHash::Unsigned(value) => Value::UnsignedValue(value),
        ExternalBlockHash::Signed(value) => Value::SignedValue(value),
        ExternalBlockHash::Digest(digest) => Value::Digest(digest),
    };
    pb::BlockHash { value: Some(value) }
}

fn extra_key_to_proto(key: ExtraKeyValue) -> pb::ExtraKeyValue {
    use pb::extra_key_value::Value;

    let value = match key {
        ExtraKeyValue::Text(text) => Value::Text(text),
        ExtraKeyValue::Data(data) => Value::Data(data),
        ExtraKeyValue::Integer(value) => Value::Integer(value),
        ExtraKeyValue::Tuple(items) => Value::Tuple(pb::ExtraKeyTuple {
            items: items.into_iter().map(extra_key_to_proto).collect(),
        }),
    };
    pb::ExtraKeyValue { value: Some(value) }
}
