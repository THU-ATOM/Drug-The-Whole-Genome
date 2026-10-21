# Installation

Run the supplied commands from the repository root in Linux with Bash.
Benchmarking and pocket encoding require an NVIDIA GPU with CUDA.

## Docker

The repository provides an environment recipe based on Ubuntu 20.04, CUDA 12.1.1,
Python 3.9 and PyTorch 2.1.2. Build with `docker/` as the context because the
Dockerfile copies `requirements.txt` from that directory:

```bash
docker build -t drugclip:local docker/
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" -w /workspace drugclip:local bash
```

The recipe compiles Uni-Core and installs additional tools. Some dependencies
are unpinned, so save the installed versions with your run. For build errors, see
[troubleshooting](TROUBLESHOOTING.md#installation-and-platform-errors).

## Existing Python environment

Install the packages listed in `docker/requirements.txt` along with the Uni-Core
and additional dependency steps in the Dockerfile. Match Uni-Core's compiled
extensions to your Python, PyTorch and CUDA versions. The commented wheel targets
CUDA 11.7 / PyTorch 1.13.1; the Docker recipe uses CUDA 12.1 / PyTorch 2.1.2.

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

`retrieval.sh` and `encode_mols.sh` default to CPU. Select `DEVICE=cuda`
to use a GPU and `CUDA_VISIBLE_DEVICES=0` (or the positional GPU argument for
encoding) to select the physical device. Retrieval uses fp16 only in CUDA mode;
molecule encoding uses fp32. `test.sh` and `encode_pocket.sh` require CUDA.

## Lightweight development checks

These checks need Python, NumPy and Bash, but no model data or GPU:

```bash
python -m unittest discover -s tests -v
bash tests/test_wrappers.sh
python -m compileall -q unimol utils
```

Test coverage and results are recorded in [validation](VALIDATION.md).
