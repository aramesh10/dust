# DUST memory optimizations

Peak-memory reductions for DUST (`dust.py`) and the noise record (`dust_noise.py`). 
Results in [NOTES.md](NOTES.md).

## Layout

| Path | Contents |
|---|---|
| `model.py`, `dust.py`, `dust_noise.py` | The model and the two original algorithms, unchanged |
| `dust_optimized.py` | DUST with all six optimizations (standalone) |
| `dust_noise_optimized.py` | dust_noise with all seven optimizations; stage 7, built on `stages/noise/stage6.py` |
| `stages/dust/stageN.py` | DUST with optimizations 1 to N |
| `stages/noise/stageN.py` | dust_noise with optimizations 1 to N |
| `benchmark.py` | Times steps and records peak memory for any method, stage or model size |
| `check_noise.py` | Compares a noise stage's gradients with dust_noise's on one batch |
| `memory_report.py` | Summarizes a CUDA memory snapshot by allocating line and method |
| `modal_app.py` | Runs the above on Modal H100s |

## Running

```bash
python benchmark.py --method optimized --population 4096           # dust, stage1..5, optimized
python benchmark.py --method noise-cut-stage3                         # noise-{boost,cut}[-stageN|-opt]
python benchmark.py --method noise-cut-opt --n-layer 16 --n-embd 1024 # scaled model
python benchmark.py --method dust --population 4096 --memory-only     # one draw chunk per site: same peak, faster
python benchmark.py --method noise-cut --memory-snapshot --output runs/bench/x.json
python memory_report.py runs/bench/x.pickle
python check_noise.py --module stages.noise.stage2
```

On Modal (one H100 per run; results are copied to `runs/bench/`, snapshots stay in the `dust-runs` volume):

```bash
modal run modal_app.py::prepare --tokens 1M                                   # data into dust-data volume
modal run modal_app.py::benchmark --method optimized --population 4096
modal run modal_app.py::benchmark --method noise-cut-stage3 --steps 3 --warmup 1
modal run modal_app.py::benchmark --method noise-cut-opt --n-layer 16 --n-embd 1024
modal run modal_app.py::benchmark --method dust --population 4096 --memory-only
modal run modal_app.py::benchmark --method noise-cut --eager
modal run modal_app.py::benchmark --method noise-cut --memory-snapshot --name snap-noise-cut
modal volume get dust-runs bench/snap-noise-cut.pickle runs/bench/snap-noise-cut.pickle
modal run modal_app.py::check_noise --module stages.noise.stage2 --chain-tokens 512
modal run modal_app.py::dust --tokens 1M --population 256 --optimized         # training
modal run modal_app.py::noise --tokens 1M --gpus 8
```

## License

Released under the [MIT License](LICENSE).
