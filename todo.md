# NEXT STEPS

- E3xx: Check final experiments

- Inspect intermediates of different datasets (add to slides)
    - Try decoding the intermediates of the processor
    - Try more processor steps than 18

- How does the DFT of the predictions look like?

- experiment with edge_latent_size=64
- experiment with num_processor_repetitions !! + correct it in the thesis

- Experiment without parameter sharing
    - Check overfitting and overall performance
    - Compute parameter efficiency

## Other
- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking

- Try data augmentation
    - Try shifting first (approved)
    - Try repeating (physically incorrect)

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (scOT), (FNO), (U-Net), (CNO), (GNOT)

- wave_equation: do not learn identity map for source
    - Allows you use residual/derivative stepping

- Extend for unstructured grids
    - Take input at any point and give output at any point
    - Build the graphs and edges on the fly based on the input
- Extend for time-independent problems
- Extend for other boundary conditions (e.g., open, Robin)

## Sequential data
- Implement and try the LSTM idea from presentation-240315-updates
- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

# LOW PRIORITY / CODE PERFORMANCE

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.

- Re-implement segment_sum to avoid constant folding
    - Not sure there is an easy solution

- Check LLM tasks (translation, prompt-based text generation, auto-completion)

- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)
