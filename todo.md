# NEXT STEPS

- E3xx: Check final experiments
- Check out [model size measures](https://medium.com/@nikitamalviya/evaluation-of-object-detection-models-flops-fps-latency-params-size-memory-storage-map-8dc9c7763cfe)

- Inspect intermediates of different datasets (add to slides)
    - Try decoding the intermediates of the processor
    - Try more processor steps than 18

- How does the DFT of the predictions look like?

- Get rid of the NVIDIA driver compatiblity message: parallel compilation possibly faster

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

# Future work

- Read gladstone2024mesh + li2020multipole and present them to Mishra
    - Read them with details and be careful !
    - The ideas are VERY similar and the performance is close
    - Maybe we need to benchmark gladstone2024mesh too
    - We should be careful with what we focus on:
        - a possibility is focusing on this new paradigm for down and upsampling layers

- experiment with num_processor_repetitions !! + correct it in the thesis

- Experiment without parameter sharing
    - Check overfitting and overall performance
    - Compute parameter efficiency

- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking

- Try data augmentation
    - Try shifting first (approved)
    - Try repeating (physically incorrect)

- Experiment with the baseline models
    - GNO, brandstetter, MeshGraphNets, (scOT), (FNO), (U-Net), (CNO), (GNOT)

- wave_equation: do not learn identity map for source
    - Allows you use residual/derivative stepping

- Extend for unstructured grids
    - Take input at any point and give output at any point
    - Build the graphs and edges on the fly based on the input
- Extend for time-independent problems
- Extend for other boundary conditions (e.g., open, Robin)

- Add multi-level SRGNO:
    - Make the encoder and decoder modular
        - Define graph downsampling and upsampling layers
        - You can give up the long-range connections in mesh to allow for unstructured "mesh"
        - Or improve the long-range connection strategy to allow for unstructured "mesh" (Check )
    - Apply encoder with multiple mesh resolutions
        1. All from the grid
        2. Hierarchical, down-scale step by step
    - Apply message-passing on all the meshes independently
    - Decode from multiple mesh resolutions
        1. All directly to the grid
        2. Hierarchical, up-scale step by step

## Sequential data
- Implement and try the LSTM idea from presentation-240315-updates
- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

# LOW PRIORITY / CODE PERFORMANCE

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.

- Re-implement segment_sum to avoid constant folding
    - Not sure there is an easy solution

- Check LLM tasks (translation, prompt-based text generation, auto-completion)

- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)
