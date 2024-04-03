# PRIORITY

- Looks like updating the gradient on the batch is not really the greatest idea.. Confirm on a big dataset before changing it..

- Run experiments with the whole and partial dataset using A100 (increase bsz)

- Use different overlap_factor for mesh2grid

- Do something for the long compilation time..
    - Check this post: https://github.com/google/jax/issues/10596
    - It is some latent features.. but why are they being constant folded?
    - Check np in setup
    - Something is probably np.array instead of jnp

- Adopt the 1D datasets to the new structure

- Move functions outside of train (?)

- Rethink the architecture: encoded coordinates, mean squared relative error, etc.
    - Decrease the latent_size of the grid nodes!! 128 is an overkill for only the coordinates and the solution.. also saves memory

# 1D datasets from Equer and Welling (?)

- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets
- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works

# EXPERIMENT

- Train with fewer time steps (128 / 64) and try extrapolating

# LATER

- Multihost JAX training

- Rethink the normalization
    - Try normalizing the residuals again, but think it through
- Normalize the inputs and residuals based on the time index
    -> Different statistics per timestep
    -> Imagine an explosion, the statistics are very different across time

- Recheck the randomness logic and ensure reproducibility

- Check LLM tasks (translation, prompt-based text generation, auto-completion)
- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)

- parameterize dtype

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations
