# PRIORITY

- Change the pmap strategy
    - use lax.pmean to aggregate the gradients and losses inside a pmap
    - Update the state for a WHOLE batch
- Use different overlap_factor for mesh2grid
- Tune learning rate and run experiments..

- Do something for the long compilation time..
    - Check this post: https://github.com/google/jax/issues/10596
    - It is some latent features.. but why are they being constant folded?
    - Check np in setup
    - Something is probably np.array instead of jnp

- Adopt the 1D datasets to the new structure

- Rethink the architecture: encoded coordinates, mean squared relative error, etc.
    - Decrease the latent_size of the grid nodes!! 128 is an overkill for only the coordinates and the solution.. also saves memory

REASONS FOR LONG TRAINING TIMES + Memory consumption COMPARED TO CNOs:
    1. Many more computations per network parameter.
    2. Many more representations per network parameter -> excessive memory
    3. Rollouts in training (applying the model multiple times)
    4. Many lead times in training, we solve and evaluate for all times, not only the final time

# 1D datasets from Equer and Welling

- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets
- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works

# EXPERIMENT

- Train with fewer time steps (128 / 64) and try extrapolating

# LATER

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
