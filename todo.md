# NEXT STEPS

- Check E224/E225

- Try data augmentation
    - Try shifting first (approved)
    - Try repeating (physically incorrect)

- Start reproducing experiments (with 5 repeats) for reporting in the thesis

- Add uncertainty to the errors
    * No need to retrain anything
    - Separate the evaluation script

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
