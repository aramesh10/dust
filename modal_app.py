"""Run data preparation, the backpropagation baseline, and DUST on Modal.

    modal run modal_app.py::prepare --tokens 1M
    modal run modal_app.py::backprop --tokens 1M --name bp-1m
    modal run modal_app.py::dust --tokens 1M --population 256 --name 1m-p256
    modal run modal_app.py::benchmark --method backprop --steps 5

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
def train_dust(args: list[str], gpus: int):
    if gpus == 1:
        run(['python', 'dust.py', '--data-dir', DATA, *args])
    else:
        run(['torchrun', '--standalone', f'--nproc_per_node={gpus}', 'dust.py', '--data-dir', DATA, *args])


@app.function(volumes=volumes, gpu=GPU, timeout=4 * 60 * 60)
def run_benchmark(args: list[str], name: str):
    output = f'{RUNS}/bench/{name}.json'
    run(['python', 'benchmark.py', '--data-dir', DATA, '--output', output, *args])
    return Path(output).read_text()


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
         steps: int = 0, gpus: int = 0):
    gpus = gpus or POPULATION_GPUS[population]
    args = options(tokens, name or f'{tokens.lower()}-p{population}', seed, steps)
    args += ['--population', str(population)]
    train_dust.with_options(gpu=f'{GPU}:{gpus}' if gpus > 1 else GPU).remote(args, gpus)


@app.local_entrypoint()
def benchmark(method: str = 'dust', tokens: str = '1M', population: int = 256, steps: int = 5,
              warmup: int = 1, gpu: str = GPU, name: str = ''):
    name = name or (f'{method}-{tokens.lower()}' + (f'-p{population}' if method == 'dust' else '')
                    + f'-{gpu.lower()}')
    args = ['--method', method, '--tokens', tokens, '--population', str(population),
            '--steps', str(steps), '--warmup', str(warmup)]
    result = run_benchmark.with_options(gpu=gpu).remote(args, name)
    local = Path('runs') / 'bench' / f'{name}.json'
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(result)
    print(f'Saved {local}')
