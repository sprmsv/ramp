# NEXT STEPS

- Add the first benchmarks: U-Net / U-ViT / FNO / CNO

- E260: Compare and fix the stepping method
    - der and res >> out

- Evaluate time inter- and extrapolation:
    - E260: Compare D=1 with D=4 and D=7
    - E260: Compare different steppers
    * Try concatenate_t=off and/or concatenate_tau=off

- Evaluate the effect of `direct_steps` on the final error
    - E260: Compare the final performance with `direct_steps=1`
    * Start from trained parameters with direct_steps=1 (turn off scheduling)
    - Train/val gap !! try without tau and t

- Check the wave_equation datasets (E261)
    * Concatenate a feature for the boundary nodes
    * Try to avoid overfitting (MP steps, latent size, etc.)
    - Train/val gap !! try without tau and t

- Investigate the effect of removing the periodic connections for periodic BC (E261 and E262)

- Continuous-discrete equivariance
    * Mask the nodes during training (p_max=0.5)
        - This masking "can" be different in the encoder and the decoder
    * Use edge length in the message-passing
        - Weighted average of the messages with distances as weights

- Investigate performance with other resolutions
    * Train with 64x64
    * Large overlap factors (4 or 8)
    * Remove grid message-passing

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * Fix what needs to be fixed
    * 1000 epochs

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (scOT), (FNO), (U-Net), (CNO), (GNOT), (CNN)

- Extend for unstructured grids
    - Take input at any point and give output at any point
    - Build the graphs and edges on the fly based on the input
- Extend for time-independent problems
- Extend for other boundary conditions (e.g., open, Robin)

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

- Check the internal features of the graph and try to make sense of them

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
