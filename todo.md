# PRIORITY

- Implement the new idea for grid-mesh connectivity in 1D

- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets

- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works

# EXPERIMENT

- compute loss for ALL the predictions (not only the last one)


- Training curriculum as in GraphCast
    - First start with only one unrolling step
    - Then increase the unrolling steps

# LATER

- Consider parallel computing with pmap

- parameterize dtype

- Imrpove the momory footprint
    - It works with smaller batches
    - Still, why memory is being accumulated over epochs? Is something jitted multiple times? Make sure by creating the jitted function only once.
    - Concatenations in the model could be the issue
    - jax.vmap might solve this

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations
