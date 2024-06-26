# MPGNO: Message Passing Graph Neural Operator

MPGNO is a graph-based neural network framework for operator learning. Once trained, it can be applied autoregressively on the initial condition of a time-dependant partial differential equation (PDE) to estimate the solution of the PDE at a later time.

Here is a schematic of the graph structure used in MPGNO for a 1D problem with periodic boundary conditions, although it has been primarily developed and validated on 2D problems:
<p align="center"> <img src="assets/multimesh-periodic.svg" alt="multimesh-periodic" width="500"/> </p>


## Setup

Build and activate the environment (Check [JAX compatibility](https://jax.readthedocs.io/en/latest/installation.html) first):
```bash
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train a model with default settings:
```bash
python -m mpgno.train python --datadir 'path_to_data' --datapath 'relative_path_to_dataset' --epochs 100 --batch_size 4 --n_train 512 --n_valid 256
```

### Euler cluster
Steps for setting up the environment on the Euler cluster:
```bash
module purge
module load stack/.2024-05-silent gcc/13.2.0 python/3.11.6_cuda
cd ~/venvs
python -m pip install --user virtualenv
python -m virtualenv venv-NAME
source venv-NAME/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"
pip install jraph flax matplotlib h5py optax pandas seaborn
```
