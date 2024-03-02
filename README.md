# GNN-based Autoregressive PDE Solver based on GraphCast

# TODO: Write readme

Build and activate the environment (Check [JAX compatibility](https://jax.readthedocs.io/en/latest/installation.html) first):
```bash
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train a model:
```bash
python -m graphneuralpdesolver.train --datadir '../../data/' --experiment E1 --resolution 128
```

## Euler cluster
Steps for setting up the environment on the Euler cluster:
```bash
module purge
module load gcc/8.2.0 python_gpu/3.11.2
cd ~/venvs
python -m virtualenv venv-NAME
source venv-NAME/bin/activate
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jraph flax matplotlib h5py optax
```
