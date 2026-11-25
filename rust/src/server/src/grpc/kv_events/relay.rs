// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::broadcast;
use tokio_util::task::AbortOnDropHandle;
use tracing::warn;
use vllm_engine_core_client::kv_events::KvEventSubscriber;
use vllm_engine_core_client::protocol::kv_events::KvEventBatch;

use super::snapshot::Blocks;
use super::{PublisherAddress, batch_to_proto, pb};

/// Live batches a subscriber may fall behind by before it skips ahead.
const LIVE_CAPACITY: usize = 1024;

/// Pause after a failed read from the publisher, which fails repeatedly while
/// the connection is down.
pub(super) const RETRY_DELAY: Duration = Duration::from_secs(1);

/// Forwards one engine's KV event batches, as its PUB socket delivers them,
/// to any number of subscribers.
pub(super) struct KvEventRelay {
    live: broadcast::Sender<Arc<pb::KvEventBatch>>,
    _task: AbortOnDropHandle<()>,
}

impl KvEventRelay {
    /// Each batch is applied to `blocks` before it is forwarded, so that a
    /// snapshot of them holds every batch a subscriber was sent before it.
    pub async fn connect(
        publisher: &PublisherAddress,
        blocks: Option<Arc<Mutex<Blocks>>>,
    ) -> vllm_engine_core_client::Result<Self> {
        let subscriber = KvEventSubscriber::connect(&publisher.endpoint, &publisher.topic).await?;
        let (live, _) = broadcast::channel(LIVE_CAPACITY);
        let task = tokio::spawn(follow(subscriber, live.clone(), blocks));
        Ok(Self {
            live,
            _task: AbortOnDropHandle::new(task),
        })
    }

    /// The batches forwarded from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<pb::KvEventBatch>> {
        self.live.subscribe()
    }
}

async fn follow(
    mut subscriber: KvEventSubscriber,
    live: broadcast::Sender<Arc<pb::KvEventBatch>>,
    blocks: Option<Arc<Mutex<Blocks>>>,
) {
    loop {
        let (sequence_number, payload) = match subscriber.next().await {
            Ok(published) => published,
            Err(error) => {
                warn!(error = %error.as_report(), "failed to read KV events from the engine");
                tokio::time::sleep(RETRY_DELAY).await;
                continue;
            }
        };
        if blocks.is_none() && live.receiver_count() == 0 {
            continue;
        }
        let batch = match KvEventBatch::decode(&payload) {
            Ok(batch) => batch,
            // A batch that cannot be decoded leaves a gap in the sequence numbers.
            Err(error) => {
                warn!(sequence_number, error = %error.as_report(), "dropping undecodable KV event batch");
                continue;
            }
        };
        let batch = Arc::new(batch_to_proto(sequence_number, batch));
        if let Some(blocks) = &blocks {
            // The fold walks every block of the batch: keep it off the threads
            // that serve requests. One that does not finish shows as a gap at
            // the next batch.
            let (blocks, batch) = (blocks.clone(), batch.clone());
            let _ = tokio::task::spawn_blocking(move || blocks.lock().apply(&batch)).await;
        }
        // No receiver is not an error: subscribers come and go.
        let _ = live.send(batch);
    }
}
