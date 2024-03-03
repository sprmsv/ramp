
# PRIORITY

- Add the mixed training approach
    - optionally, compute loss for ALL the predictions (not only the last one)
    - increase unroll_steps in a training curriculum

- Implement the new idea for grid-mesh connectivity in 1D

- Implement training curriculum as in GraphCast
    - First start with only one unrolling step
    - Then increase the unrolling steps
    - In Brandstetter, the number of unrolling steps is chosen randomly

- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets

- Consider parallel computing with pmap

- parameterize dtype

- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works

# LATER

- Imrpove the momory footprint
    - It works with smaller batches
    - Still, why memory is being accumulated over epochs? Is something jitted multiple times? Make sure by creating the jitted function only once.
    - Concatenations in the model could be the issue
    - jax.vmap might solve this

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations
