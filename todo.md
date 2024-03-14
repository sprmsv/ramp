# PRIORITY

- Report
    - Check 600 and 601 + update report (change commit + dataset std)
    - Create new errors plots (with L1)
    - Add your notes

- Change the pmap strategy
- Use pmap for evaluation too
- Use different overlap_factor for mesh2grid

- Do something for the long compilation time..
    - Check this post: https://github.com/google/jax/issues/10596
    - It is some latent features.. but why are they being constant folded?

- Adopt the 1D datasets to the new structure

- Rethink the architecture: encoded coordinates, mean squared relative error, etc.
    - Decrease the latent_size of the grid nodes!! 128 is an overkill for only the coordinates and the solution.. also saves memory

# NEXT MEETING

REASONS FOR LONG TRAINING TIMES + Memory consumption COMPARED TO CNOs:
    1. Many more computations per network parameter.
    2. Many more representations per network parameter -> excessive memory
    3. Rollouts in training (applying the model multiple times)
    4. Many lead times in training, we solve and evaluate for all times, not only the final time

AUTOREGRESSIVE TRAINING:
    3. LSTM/Transformer idea (highlight the difference with Equer)
    4. Without time-bundling and LSTM/Transformer, we are not treating the input as a sequence, we are ignoring the history of the sequence !!

PROPER NORMALIZATION:
    - Normalizing the structural node and edge features

SEGMENT MEAN INSTEAD OF SUM + screenshots

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
