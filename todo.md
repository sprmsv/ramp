# PRIORITY

- Run experiments with different datasets
    - Start with direct_steps=1, bm, sin_easy
    - Increase unroll_steps for a few epochs
    - Repeat the recipe for other datasets
    - Put the results in a report
    - Increase direct_steps
        - First, try without learned_correction in the MLP
        - Add learned_correction with care and see if it brings any good

- Automize the training curriculum

- Tune parameters and check TRYs

- Profile speed and performance vs batch_size

- Run experiments with the whole and partial dataset

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

- It has been confirmed that updating the gradients more often is better.
    - Consider changing the pmap strategy to speed up the convergence.
    - This might require direct_steps * num_lead_times to be dividable by the number of processors...

- Multihost JAX training

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
