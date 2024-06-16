# NEXT STEPS

- Check E291 (test): Add resolution-invariance results
- Check E295: wave_equation
- Check E296: Benchmark U-Net to make sure about MPGNO
- Check E299: How many epochs?

- Implement the new idea for fractional tau
    - tau=.5 unrolled twice ~ tau=1
    - tau=.2 unrolled five times ~ tau=1

- Continuous-discrete equivariance
    * Mask the nodes during training (p_max=0.5)
        - This masking "can" be different in the encoder and the decoder
    * Use edge length in the message-passing
        - Weighted average of the messages with distances as weights

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * Fix what needs to be fixed
        - How many epochs? 2000? 1000? 600?
    * 1000 epochs

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
