# PRIORITY

- Validate the numbers reported in the trainings
    - Try with the same statistics (load them)

- Run another ablation study
    - Increase the message-passing steps
    - Decreasse the mesh grid to 32x32
    - Fewer samples / more epochs to get the full potential of the models
    - This time, also try MM128 with OF0.1 (for discontinuities..)
    - Separate the latent_size of the grid nodes and the mesh nodes

- Experiment with all datasets

- Work on the training time (possibly parallelization..)
    - Profile speed and performance vs batch_size

# EXPERIMENTS

- Try with optax.linear_onecycle_schedule

- Train with different n_train to see how it scales with more data

- Try mean squared relative error

- Experiment with other physical features

- Try the old normalization strategy again..

## Autoregressive error accumulation
I am out of ideas.

## Grouping phenomenon
- Try increasing overlap_factor
- Try more mesh nodes (128x128)
- Try masking the edges/nodes in message-passing

### Error on steep gradients
I suspect this is a consequent of the smoothing effect. Explore with different architectures.

### Structural artifact in solutions
- Explore with different architectures.
- Try convolutional or message-passing step after the decoder

## Sequential data
- Implement and try the LSTM idea from presentation-240315-updates
- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

## OTHER

- Train with fewer time steps and try extrapolating

# 1D datasets from Equer and Welling (?)
- Adopt the 1D datasets to the new structure
- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets
- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works

# LATER

- Move functions outside of main::train (?)

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
