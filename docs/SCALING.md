# Scaling notes — from 2k vectors to 100M+

The demo indexes ~2,000 embeddings with FAISS `IndexFlatIP` — exact, brute-force inner-product
search. At this size that is the *correct* choice: building anything approximate would add recall
loss for zero latency benefit (a flat scan over 2k × 1024-d fp32 is sub-millisecond). This note
records what changes when the archive grows to Major-TOM scale (the open reference point:
[169M+ precomputed embeddings](https://huggingface.co/Major-TOM) from 62 TB of Sentinel data).

## What stays the same

- **Embed once, reuse everywhere.** The decoupled extract → store → index pipeline is already the
  at-scale shape; only the store format and index type swap out.
- **The encoder dominates cost.** A ViT-L forward pass is ~10–100 ms/tile (GPU); search is
  microseconds–milliseconds. At every scale below, *embedding* throughput — not retrieval — is the
  bottleneck, which is why the heavy pass is a one-time batch job.

## Index structure by scale

1024-d fp32 = 4 KB/vector. The driver is memory, then latency:

| Archive size | Raw vectors | Index | Rationale |
|---|---|---|---|
| ≤ 1M | ≤ 4 GB | `IndexFlatIP` or HNSW | Flat still fits in RAM; HNSW if p99 latency matters |
| 1M – 100M | 4 – 400 GB | IVF-PQ (+ OPQ rotation) | PQ compresses 4 KB → 32–64 B codes (64–128×); IVF probes a few clusters instead of scanning |
| 100M+ | 400 GB+ | Sharded IVF-PQ, DiskANN-style, or GPU index | Single-node RAM is exhausted; shard by geography/time or go disk-resident |

Concrete anchors:

- **PQ64 at 100M vectors ≈ 6.4 GB** of codes — fits on one GPU (`faiss-gpu`), giving
  ~10⁵–10⁶ queries/s; the recall/latency knob is `nprobe` (typical: 90–99% recall@10 at 1–10 ms).
- **HNSW** holds full-precision vectors (no compression) — great recall at single-digit ms, but at
  100M × 4 KB it needs ~400 GB+ RAM plus graph overhead; that is why compressed IVF variants win
  at archive scale.
- Train the IVF/PQ codebooks on a representative sample (~1–5M vectors), not the full set.

## Store schema

`store.py` writes one parquet row per tile with the vector as a per-row array — fine at 2k rows,
inefficient at scale (per-row object overhead, no zero-copy reads). The evolution path:

1. **Arrow `FixedSizeList(float32, 1024)`** columns → zero-copy into FAISS/NumPy, predictable
   row width, fast scans.
2. **Partitioned layout** (`modality=/year=/tile_grid=`) so metadata predicates prune files before
   any vector is touched — this is also how time-window and AOI filters stay cheap.
3. At the top end: an append-only vector store (e.g. Lance, or mmap'd `.npy` shards + a parquet
   manifest) so re-indexing never rewrites vectors.

## Serving shape

- **Shard + merge:** route a query to all (or geo-pruned) shards, merge top-k by score — FAISS
  `IndexShards`/`IndexReplicas` covers single-node; cross-node is a thin RPC layer.
- **Filtered search** (class, date range, AOI) via partition pruning first, ID-selector second —
  pushing filters into the index beats post-filtering top-k.
- **Updates:** new scenes arrive as batch embed jobs appending a shard; background merge/retrain
  of codebooks happens off the query path.

## Latency budget (demo vs. target)

| Stage | This repo (P40, fp32) | At scale |
|---|---|---|
| Embed 1 tile | O(10) ms amortized, batched ViT-L — order of magnitude, not benchmarked here | same per-GPU; scale horizontally, it's embarrassingly parallel |
| Search 2k vectors | < 1 ms (exact flat scan) | 1–10 ms @ 100M (IVF-PQ, tuned `nprobe`) |
| End-to-end "find similar" | dominated by embed | dominated by embed — cache query embeddings |

The honest summary: nothing in this repo's architecture has to change to scale — the flat index
and row-wise parquet are deliberate small-scale choices, and each has a named, well-trodden
replacement once the archive outgrows them.
