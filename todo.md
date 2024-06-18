# NEXT STEPS

- Update slides ::
    - Check E295: wave_equation :: Add example plots (E299)
    - Check E297 (+ add to slides)
        - Check test figures (E299) for tau=4,6 and other fractional tau
        - Update learning curves with E298 !! DR-4 is not affected either !! + better stability
        - gradient cut (E298) vs. gradient flow (E297)

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * 1200 / 2000 epochs

- How does the DFT of the predictions look like?

- wave_equation: do not learn identity map for source
    - Allows you use residual/derivative stepping

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (scOT), (FNO), (U-Net), (CNO), (GNOT)

- Extend for unstructured grids
    - Take input at any point and give output at any point
    - Build the graphs and edges on the fly based on the input
- Extend for time-independent problems
- Extend for other boundary conditions (e.g., open, Robin)

## Other
- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking

- Try data augmentation
    - Try shifting first (approved)
    - Try repeating (physically incorrect)

- Check the internal features of the graph and try to make sense of them

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

## Sequential data
- Implement and try the LSTM idea from presentation-240315-updates
- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

# LOW PRIORITY / CODE PERFORMANCE

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.

- Re-implement segment_sum to avoid constant folding
    - Not sure there is an easy solution

- Check LLM tasks (translation, prompt-based text generation, auto-completion)

- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)
