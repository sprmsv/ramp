# NEXT STEPS

- Conclude E199 for the learning rate scheduler
    - Fix learning rate hyperparameters
    - If necessary, relaunch the current experiments with new learning rates
    - Launch model selection for a compressible dataset
    - Try weight_decay=1e-02//1e-04 with the best settings for several datasets

- Data scaling experiments
    * (E200) Start with one/two dataset:
        - kh, shear_layer, and sines
    * Do for riemann_kh if time permits

- Invest a bit more on the compressible datasets
    - Try correction again with t_inp and tau
    - Consider using BatchNorm instead of LayerNorm
    - Experiment with normalization axes on different datasets
    - Predict output directly instead of residuals
    - (E201) Train without downsampling the trajectories (easier dynamics)
    - Carry out model selection
    - Try data augmentation
        - Try shifting first (approved)
        - Try repeating but this is physically incorrect

- Try mean squared relative error

- Read the GenCast paper for uncertainty quantification

- Experiment with direct_steps and jump_steps too
    - Update the evaluation functions first
    - Evaluate separately with different methods for autoregressive prediction

- Run ablation studies with models that fully reach their potential !!
    * Data reach regime

- Fix everything and start reproducing experiments (with repeats) for reporting in the thesis


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

# LOW PRIORITY / CODE PERFORMANCE

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.

- Re-implement segment_sum to avoid constant folding
    - Not sure there is an easy solution

- Normalize the inputs and residuals based on the time index
    -> Different statistics per timestep
    -> Imagine an explosion, the statistics are very different across time

- Recheck the randomness logic and ensure reproducibility

- Check LLM tasks (translation, prompt-based text generation, auto-completion)
- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

## 1D datasets from Equer and Welling (?)
- Adopt the 1D datasets to the new structure
- RE-GENERATE THE DATA !! PARAMETERS ARE REPEATED !!
- Change the structure of the data and support all datasets
- Adopt the models from the other works and extend your repo
- Reproduce the experiments of the other works
