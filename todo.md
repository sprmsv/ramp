# NEXT STEPS

- Add the recent work to your update slides (Check E142/E143)
    - Check the overfitting of E143 with lighter models
    - Check the L2-DR effect with each normalization
- Check E144/E145/E146 for normalization axes

- Launch E150: New architecture
    - Launch with two normalizations: axis=(0,) and axis=(0, 1,)

- Experiment with the new datasets
- Add data augmentation

## Challenging datasets

- Predict output directly instead of residuals
- Train without downsampling the trajectories (easier dynamics)

## Other

- Update cutoff, evaluation, and assertions to have a fair comparison for different jump_steps
    - Temporarily, evaluate on 14 with J=1
    - Make sure to check direct evaluation too

- Add the ignored pairs to your training
    - This way you can increase direct_steps even for short trajectories
- Debug jump_steps

# EXPERIMENTS

- Try with other learning rates and decays
    - Try with optax.linear_onecycle_schedule
    - Maybe you can increase the learning rate and converge in fewer epochs this way

- Try mean squared relative error

- Run ablation studies with models that fully reach their potential !!
    - Data reach regime
    - Long trainings

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
