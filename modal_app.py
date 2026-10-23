"""Run data preparation, the backpropagation baseline, and DUST on Modal.

    modal run modal_app.py::prepare --tokens 1M
    modal run modal_app.py::backprop --tokens 1M --name bp-1m
    modal run modal_app.py::dust --tokens 1M --population 256 --name 1m-p256
    modal run modal_app.py::dust --tokens 1M --population 256 --optimized
    modal run modal_app.py::noise --tokens 1M --gpus 8
    modal run modal_app.py::benchmark --method backprop --steps 5
    modal run modal_app.py::check_noise

Data and runs persist in the Modal Volumes `dust-data` and `dust-runs`.
Download a run with `modal volume get dust-runs bp-1m runs/bp-1m`.
"""
from pathlib import Path
import signal
import subprocess
import time

import modal


GPU = 'H100'
POPULATION_GPUS = {256: 1, 1024: 1, 4096: 4, 16384: 8}
CODE = '/root/dust'
DATA = '/data'
RUNS = '/runs'

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('tokenizers==0.23.1', 'datasets==4.8.4')
    .add_local_dir(
        '.', remote_path=CODE,
        ignore=['data', 'runs', '.venv', '.git', '**/__pycache__', 'compare'],
    )
)
data_volume = modal.Volume.from_name('dust-data', create_if_missing=True)
runs_volume = modal.Volume.from_name('dust-runs', create_if_missing=True)
volumes = {DATA: data_volume, RUNS: runs_volume}

app = modal.App('dust', image=image)


def run(command):
    print('$', ' '.join(command), flush=True)
    try:
        subprocess.run(command, cwd=CODE, check=True)
    finally:
        runs_volume.commit()


@app.function(volumes=volumes, cpu=8, memory=32768, timeout=4 * 60 * 60)
def prepare_data(tokens: str):
    manifest = Path(DATA) / 'manifest.json'
    start = time.time()
    try:
        command = ['python', 'prepare_data.py', '--tokens', tokens, '--output', DATA]
        returncode = subprocess.run(command, cwd=CODE).returncode
    finally:
        data_volume.commit()
    # The abandoned FineWeb stream can abort the interpreter at exit (SIGABRT)
    # after all files are saved; the manifest is written last, so it marks success.
    if returncode == -signal.SIGABRT and manifest.is_file() and manifest.stat().st_mtime >= start:
        print('prepare_data.py aborted during shutdown after writing all files; ignoring.')
    elif returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


@app.function(volumes=volumes, gpu=GPU, timeout=4 * 60 * 60)
def train_backprop(args: list[str]):
    run(['python', 'baselines/backprop.py', '--data-dir', DATA, *args])


@app.function(volumes=volumes, gpu=GPU, timeout=24 * 60 * 60)
def train_dust(args: list[str], gpus: int, script: str = 'dust.py'):
    if gpus == 1:
        run(['python', script, '--data-dir', DATA, *args])
    else:
        run(['torchrun', '--standalone', f'--nproc_per_node={gpus}', script, '--data-dir', DATA, *args])


@app.function(volumes=volumes, gpu=GPU, timeout=4 * 60 * 60)
def run_benchmark(args: list[str], name: str):
    output = f'{RUNS}/bench/{name}.json'
    run(['python', 'benchmark.py', '--data-dir', DATA, '--output', output, *args])
    return Path(output).read_text()


@app.function(volumes=volumes, gpu=GPU, timeout=60 * 60)
def run_check_noise(args: list[str]):
    run(['python', 'check_noise.py', '--data-dir', DATA, *args])


def options(tokens, name, seed, steps):
    args = ['--tokens', tokens, '--output', f'{RUNS}/{name}', '--seed', str(seed)]
    return args + (['--steps', str(steps)] if steps else [])


@app.local_entrypoint()
def prepare(tokens: str = 'both'):
    prepare_data.remote(tokens)


@app.local_entrypoint()
def backprop(tokens: str = '1M', name: str = '', seed: int = 42, steps: int = 0,
             lr: float = 0.0, momentum: float = -1.0):
    args = options(tokens, name or f'bp-{tokens.lower()}', seed, steps)
    args += ['--lr', str(lr)] if lr else []
    args += ['--momentum', str(momentum)] if momentum >= 0 else []
    train_backprop.remote(args)


@app.local_entrypoint()
def dust(tokens: str = '1M', population: int = 256, name: str = '', seed: int = 42,
         steps: int = 0, gpus: int = 0, optimized: bool = False):
    gpus = gpus or POPULATION_GPUS[population]
    prefix = 'opt-' if optimized else ''
    args = options(tokens, name or prefix + f'{tokens.lower()}-p{population}', seed, steps)
    args += ['--population', str(population)]
    script = 'dust_optimized.py' if optimized else 'dust.py'
    train_dust.with_options(gpu=f'{GPU}:{gpus}' if gpus > 1 else GPU).remote(args, gpus, script)


@app.local_entrypoint()
def noise(tokens: str = '1M', name: str = '', seed: int = 0, steps: int = 0, gpus: int = 1):
    args = options(tokens, name or f'noise-{tokens.lower()}', seed, steps)
    train_dust.with_options(gpu=f'{GPU}:{gpus}' if gpus > 1 else GPU).remote(args, gpus, 'dust_noise.py')


@app.local_entrypoint()
def benchmark(method: str = 'dust', tokens: str = '1M', population: int = 256, steps: int = 5,
              warmup: int = 1, gpu: str = GPU, name: str = '', memory_snapshot: bool = False,
              grad_mode: str = 'no_grad', eager: bool = False, n_layer: int = 8, n_embd: int = 512,
              memory_only: bool = False):
    name = name or (f'{method}-{tokens.lower()}' + (f'-p{population}' if method != 'backprop' else '')
                    + f'-{gpu.lower()}')
    args = ['--method', method, '--tokens', tokens, '--population', str(population),
            '--steps', str(steps), '--warmup', str(warmup), '--grad-mode', grad_mode]
    args += ['--memory-snapshot'] if memory_snapshot else []
    args += ['--eager'] if eager else []
    args += ['--n-layer', str(n_layer), '--n-embd', str(n_embd)] + (['--memory-only'] if memory_only else [])
    result = run_benchmark.with_options(gpu=gpu).remote(args, name)
    local = Path('runs') / 'bench' / f'{name}.json'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(result)
    print(f'Saved {local}')


@app.local_entrypoint()
def check_noise(module: str = 'dust_noise_optimized', chain_tokens: int = 0):
    run_check_noise.remote(['--module', module] + (['--chain-tokens', str(chain_tokens)] if chain_tokens else []))
