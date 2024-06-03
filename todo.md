# NEXT STEPS

- Evaluate time inter- and extrapolation::
    - Compare D=1 with D=4 and D=7
    * Try concatenate_t=off and/or concatenate_tau=off

- Evaluate the effect of `direct_steps` on the final error
    * Compare the final performance with `direct_steps=1`
    * With or without scheduling `direct_steps`?

- Remove periodic connections and try again with wave_equation datasets
    * Concatenate a feature for the boundary nodes

- Investigate the effect of edge masking with OF=4 on the final error
    * grid2mesh, multimesh, mesh2grid

- Continuous-discrete equivariance
    * Mask the nodes during training (p_max=0.5)
        - This masking "can" be different in the encoder and the decoder
    * Use edge length in the message-passing
        - Weighted average of the messages with distances as weights
    * If you're too misreable, plot the edge connections to be absolutely sure about them

- Investigate performance with other resolutions (Check E251)
    * Train with 64x64
    * Large overlap factors (4 or 8)
    * Remove grid message-passing
    - Also check the effect of changing mesh resolution

- Move evaluations and plots to a script
    * Create a test set in Dataset
    * Add downsampling in Dataset

- Reproduce experiments for reporting in the thesis
    * pull the latest version
    * 1000 epochs

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (FNO), (U-Net), (CNO), (GNOT), (CNN)

- Get space-continuous outputs
    - Evaluate space inter- and extrapolation

- Adapt for other boundary conditions (e.g., open, Robin)

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
