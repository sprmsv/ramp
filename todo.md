# NEXT STEPS


- Experiment with all datasets
    - (MM32/16, MLP1, OF1) might be better
    - Train with both T08 and T09

- Train with many (>8) GPUs and big batches to show off the speed
- Profile and report the time with different T and G

- Check E104 and add new results to the report for brownian_bridge

# EXPERIMENTS

- Try the old normalization strategy again..

- Try with optax.linear_onecycle_schedule
    - Maybe you can increase the learning rate and converge in fewer epochs this way

- Start from parameters of brownian_bridge and train for another dataset

- Run ablation studies with models that fully reach their potential !!
    - Data reach regime
    - Long trainings

- Experiment with other physical features

- Try mean squared relative error

- Train with different n_train to see how it scales with more data

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

- Train with continuious time and try inter- and extrapolating

# LATER

- Normalize the inputs and residuals based on the time index
    -> Different statistics per timestep
    -> Imagine an explosion, the statistics are very different across time

- Recheck the randomness logic and ensure reproducibility

- Check LLM tasks (translation, prompt-based text generation, auto-completion)
- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

### 1D datasets from Equer and Welling (?)
- Adopt the 1D datasets to the new structure
- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets
- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works
