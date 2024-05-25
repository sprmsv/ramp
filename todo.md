# NEXT STEPS

- Add Bogdan's metric (make sure to normalize)

- Add your current experiments to the thesis

- Experiment with wave_gauss, wave_layer, and allen_cahn

- The results with `direct_steps` are strange: think about it and recheck the codes
    - Inspect the performance of E226/E227

- Evaluate time interpolation
    * Needs `direct_steps`
    * With or without scheduling `direct_steps`?
    * Try `conditional_normalization` with different latent_sizes?

- Evaluate time extrapolation

- Noise control
    * train with clean data
    * Infer with noisy data (1%)
    * Compare with clean ground-truth

- Get space-continuous outputs
    - Evaluate space inter- and extrapolation

- adapt for unstructured grids

- adapt for time-independent problems

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * 1000 epochs


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
