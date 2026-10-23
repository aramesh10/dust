Dust Optimizations:

Reducing peak-memory:

1. Dust is already compute-bound for the given model at one draw. We can reduce peak memory utilization by running draws one at a time rather than `nrep` draws at once. No e2e time regression.
2. A gradient estimate needs the error estimate and the inputs. Dust runs a full forward pass and saves the result of every linear and embedding layer of the transformer block.
   1. Here show trade off between adding forward pass to recompute vs savin the memory itself
   2. The same saved block inputs let a draw that jitters a site in block L rerun the network from block L's saved input instead of from the embedding, since noise in block L cannot change the blocks before it; this part saves compute (about a fifth of the block passes), not memory.
3. Noise is generated on demand from seed rather than stored
4. Cross-entropy is computed `LOSS_CHUNK` = 2048 tokens at a time, since each token's loss depends only on its own logits, so the full float32 logits for all 16,384 tokens (~0.75 GiB with the softcap temporaries) never exist at once.
5. Each gradient is added to SGD's momentum buffer (`buf = μ·buf + g`) as soon as it is estimated and then dropped, instead of keeping every gradient until `optimizer.step()`; the weights all move together at the end of the step (`p -= lr·buf`), so every draw still sees the same weights and the update is exactly SGD with momentum.
6. Each block's attention-output target is estimated in that block's pass and used right away by its attention internals (q, k, v, gate, value embeddings), so only one block's target exists at a time instead of all eight being kept until the end.

---

Noise Optimizations:

1. Routed keys by index (Cut, 8 GiB). The replay's own-query response moves each query's 256 strongest key logits. dust_noise gathered a copy of those keys and values for every query in every layer, duplicating each key ~256 times. Each query now keeps the indices, and the routed Triton kernel reads the layer's clean keys and values through them.
2. MLP units gathered per layer (Cut, 3.5 GiB). The replay restricts each MLP to its token's top-512 units, and their weights were gathered per token for all layers before the replay. They are now gathered inside the replay as each layer is entered.
3. Replay blocks of 256 tokens instead of 512 (Cut). Halves the moved streams and readout logits. No effect on the peak at width 512, but these buffers grow with W^2 and W*V and reach ~17 GiB at width 2048. Changes the shapes of the bf16 matmuls, so estimates differ in rounding (~1e-5 relative).
4. Attention maps per sequence, never cached (2 GiB). All layers' maps were built during the query probes, which don't use them, and kept until the end of the step. The value credit now builds each sequence's map when it needs it.
5. Embedding probes 4 at a time instead of 16 (Boost, ~1.5 GiB), so the first attention stage's readout holds 4 copies of the batch's float32 logits instead of 16.
6. Captured bf16 layer inputs (both c_proj inputs) are stored as bf16 and widened when read, instead of stored as float32 copies.
7. Query probe draws are signed in place (1 GiB). Each batch is Hadamard rows times two sign patterns; multiplying in place keeps one 512 MiB copy instead of three.

---

---

Results:

Dust: 1M tokens, population 4096 (8 draws per pass), batch 8 × 2048 tokens, GPT 8 layers × 512 (8 heads), one H100 80GB HBM3, torch 2.10. Mean of 3 steps after 1 warmup. Stages are cumulative (stages/dust/stageN.py).

| Stage | Added | Peak allocated (GiB) | Peak reserved (GiB) | Step (s) |
|---|---|---|---|---|
| 0 | dust.py | 12.12 | 14.93 | 69.3 |
| 1 | One draw at a time | 6.44 (−47%) | 6.52 | 79.0 |
| 2 | Block-input checkpoints, layers recomputed | 2.67 (−78%) | 2.96 | 65.6 |
| 3 | Noise regenerated from seeds | 1.91 (−84%) | 2.46 | 65.9 |
| 4 | Cross-entropy in 2048-token chunks | 1.72 (−86%) | 1.77 | 68.5 |
| 5 | Gradients folded into momentum | 1.31 (−89%) | 1.38 | 68.2 |
| 6 | Per-block attention targets (dust_optimized.py) | 1.09 (−91%) | 1.23 | 66.2 |

Noise: 1M tokens, batch 4 × 2048 tokens, same GPT, dust_noise's recipe (64 query draws per head cycle, 512 routed MLP units, 256 routed keys, 128 future probes). Boost and Cut timed separately on one H100 80GB HBM3, torch 2.10. Mean of 3 steps after 1 warmup (5 for the Cut baseline). Stages are cumulative (stages/noise/stageN.py); a stage that leaves a phase's code unchanged was not rerun for that phase and repeats the row above.

| Stage | Added | Cut peak (GiB) | Cut reserved (GiB) | Cut step (s) | Boost peak (GiB) | Boost reserved (GiB) | Boost step (s) |
|---|---|---|---|---|---|---|---|
| 0 | dust_noise.py | 18.69 | 23.05 | 15.04 | 8.92 | 11.86 | 1.10 |
| 1 | Routed keys by index | 10.84 (−42%) | 15.17 | 15.02 | 8.92 | 11.86 | 1.21 |
| 2 | MLP units gathered per layer | 9.27 (−50%) | 11.17 | 15.17 | 8.92 | 11.86 | 1.21 |
| 3 | 256-token replay blocks | 9.27 (−50%) | 9.91 | 15.22 | 8.92 | 11.86 | 1.21 |
| 4 | Attention maps per sequence | 7.37 (−61%) | 7.91 | 15.30 | 6.92 (−22%) | 9.86 | 1.18 |
| 5 | Embedding probes 4 at a time | 7.37 (−61%) | 7.91 | 15.30 | 6.00 (−33%) | 6.36 | 1.09 |
| 6 | Captured inputs kept in bf16 | 6.92 (−63%) | 7.46 | 15.17 | 5.57 (−38%) | 5.89 | 1.09 |
| 7 | Query draws signed in place (dust_noise_optimized.py) | 6.92 (−63%) | 7.09 | 15.28 | 5.09 (−43%) | 5.39 | 1.11 |

Scaling: largest model that fits one H100 80GB. Width 512-4096 (powers of two, which dust_noise's Hadamard probes need), 64 dimensions per head, MLP 4× width, sequence 2048, vocab 4096, all estimator settings as above; Dust's 8-block draw recipe is stretched to the depth. At a fixed width every memory term is linear in depth, so each method is measured at depths 4 and 8 and the line a + b·depth is solved for 78 GiB allocated. Dust runs one draw chunk per site (--memory-only), which has the same shapes and peak as the full population (checked at 8 × 512: 1.09 GiB both for dust_optimized). Noise's limit is Cut's, its larger phase.

Peak allocated (GiB):

| Model (depth × width, params) | Dust | Dust optimized | Noise Cut | Noise Cut optimized | Noise Boost | Noise Boost optimized |
|---|---|---|---|---|---|---|
| 4 × 512, 21M | 9.69 | 0.92 | 10.83 | 4.35 | 6.38 | 3.74 |
| 8 × 512, 38M | 11.62 | 1.09 | 18.27 | 6.52 | 8.52 | 4.69 |
| 4 × 1024, 67M | 13.56 | 1.95 | 24.79 | 8.46 | 10.67 | 7.30 |
| 8 × 1024, 126M | 17.76 | 2.48 | 39.63 | 12.87 | 15.17 | 9.36 |
| 4 × 2048, 235M | 27.34 | 4.56 | 67.58 | 24.94 | 21.65 | 14.92 |
| 8 × 2048, 453M | 37.13 | 6.37 | OOM | 30.42 | 31.30 | 19.70 |
| 4 × 4096, 872M | 60.01 | 12.03 | OOM | OOM | 45.61 | 32.16 |
| 8 × 4096, 1711M | OOM | 18.66 | OOM | OOM | OOM | 53.84 |

Depth limit (parameters) per width:

| Width | Dust | Dust optimized | Noise | Noise optimized |
|---|---|---|---|---|
| 512 | 146 (0.62B) | 1805 (7.57B) | 38 (0.16B) | 118 (0.50B) |
| 1024 | 65 (0.96B) | 576 (8.46B) | 18 (0.27B) | 66 (0.98B) |
| 2048 | 22 (1.22B) | 166 (9.07B) | 4 (0.24B) | 42 (2.31B) |
| 4096 | 4 (0.87B) | 43 (9.04B) | OOM | OOM |
| Largest | ~1.2B | ~9B | ~0.27B | ~2B |

- The optimizations raise the limit about 7.5× for Dust and 7× for noise.
- The originals run out of memory a layer or two short of the fitted line, from fragmentation near the limit: Dust at 24 × 2048 and noise at 40 × 512 were measured OOM, so their limits are set two layers lower. Dust at 8 × 4096 and noise at 8 × 2048 were OOM, so 4 is the measured depth there. Dust optimized at 43 × 4096 was measured at 76.7 GiB. At width 4096 noise Cut runs out of memory at depth 4 in both versions; in the optimized one the replay's 256-token block buffers alone are about 55 GiB at this width.
- Noise optimized grows faster with depth than the depth 4-8 line says: at 64 × 512 it measured 42.7 GiB against 36.9 predicted. Its 512 limit uses depths 8 and 64; its 1024 and 2048 limits come from depths 4 and 8 and are upper bounds.
- Dust optimized is now bounded by its float32 weights and SGD momentum: 1.66 GiB per layer at width 4096, of which 12·W²·8 bytes is 1.5 GiB. Noise optimized at width 2048 has a 19 GiB fixed cost from the replay's block buffers, which grow with W² and W·vocab.
- At these sizes compute, not memory, is the limit: a memory-only Dust optimized step at 43 × 4096 took 530 s, and one noise Cut step at 64 × 512, including compilation, took 64 minutes.
