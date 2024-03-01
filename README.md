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
