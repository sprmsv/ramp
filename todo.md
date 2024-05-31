# NEXT STEPS

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (FNO), (U-Net), (CNO), (GNOT), (CNN)

- Evaluate time inter- and extrapolation::
    - Pick the conditional normalization mode
        * No cond_norm
        * Nonlinear + unique output
        * Nonlinear + non-unique output
        * Linear + unique output
        * Linear + non-unique output

    - Prune the use of conditional normalization
        * Turn off at encoder/processor/decoder
        * Switch off at message/aggregation MLPs
        * Tune latent_size
        * Try use_t=off and/or use_tau=off
        * Try concatenating tau to the edges

- Evaluate the effect of `direct_steps` on the final error
    * Compare the final performance with `direct_steps=1`
    * With or without scheduling `direct_steps`?
    * As a data augmentation technique

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * 1000 epochs

- Adapt for other boundary conditions (e.g., open, Robin)

- Get space-continuous outputs
    - Evaluate space inter- and extrapolation

- adapt for unstructured grids

- adapt for time-independent problems


## Other
- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking
    - Separate the evaluation script

- Combine initial conditions
    * Foundation model
    * Different state domains

- Try data augmentation
    - Try shifting first (approved)
    - Try repeating (physically incorrect)

## Sequential data
- Implement and try the LSTM idea from presentation-240315-updates
- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

# LOW PRIORITY / CODE PERFORMANCE

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.

- Re-implement segment_sum to avoid constant folding
    - Not sure there is an easy solution

- Check LLM tasks (translation, prompt-based text generation, auto-completion)
- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations
