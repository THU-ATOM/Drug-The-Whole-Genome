# Installation

Use Linux with Bash for the supplied commands. GPU benchmarking and pocket
encoding use CUDA internally; a successful import on Windows or macOS does not
establish support for those paths. Run commands from the repository root.

## Existing Docker environment

The repository provides an environment recipe based on Ubuntu 20.04, CUDA 12.1.1,
Python 3.9 and PyTorch 2.1.2. Build with `docker/` as the context because the
Dockerfile copies `requirements.txt` from that directory:

```bash
docker build -t drugclip:local docker/
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" -w /workspace drugclip:local bash
```

The recipe includes Uni-Core compilation and extra tools. It is not a dependency
lock: some dependencies are unpinned. This refactor has not rebuilt the image.
Known build reports and remaining caveats are listed in
[troubleshooting](TROUBLESHOOTING.md#installation-and-platform-errors).

## Existing Python environment

The package list is `docker/requirements.txt`, not a root-level requirements file.
It is not a complete standalone installer: the Dockerfile installs Uni-Core and
additional dependencies separately. Match Uni-Core's compiled extensions to the
Python, PyTorch and CUDA versions in your environment; do not install the old
commented CUDA 11.7 / PyTorch 1.13.1 wheel into the CUDA 12.1 / PyTorch 2.1.2 stack.

Before downloading a large library, inspect the environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import unicore, rdkit, lmdb, h5py, zstandard; import unimol; print('imports OK')"
python unimol/retrieval.py --help
```

For chunked screening, also check `import torch_scatter`. Errors from the import
checks should be resolved before running a model. Record `git rev-parse HEAD`,
`python --version`, `pip freeze` and `nvidia-smi` with an experiment.

## CPU and GPU modes

`retrieval.sh` and `encode_mols.sh` retain their CPU default. Select `DEVICE=cuda`
to use a GPU and `CUDA_VISIBLE_DEVICES=0` (or the positional GPU argument for
encoding) to select the physical device. Retrieval uses fp16 only in CUDA mode;
molecule encoding retains fp32. `test.sh` and `encode_pocket.sh` require CUDA in
their task implementations. CPU fallback is not supported throughout the project.

## Lightweight development checks

These checks need Python, NumPy and Bash, but no model data or GPU:

```bash
python -m unittest discover -s tests -v
bash tests/test_wrappers.sh
python -m compileall -q unimol utils
```

The tests cover orchestration and scoring with deterministic encoder doubles.
They do not replace a real checkpoint run. See [validation](VALIDATION.md).
