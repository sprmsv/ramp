# Project Updates

- Limitations
    - Speed
    - Graph construction for non-periodic BC
    - Imposing boundary conditions

- Inference is always without edge masking

- Direct evaluation works best for wave equation
- For some of the other datasets, a larger tau max is used

- OF=1 is actually pretty good, unline the previous settings
    - OF=1.5 might be just enough (2x faster)
    - Checked the graphs visually
    - Verifying with E614

- Improvement in fractional pairing (more robust)
    - Derivative stepping
        - (tau = 1)
            - Infer with (tau_mid=0)
            - Train with (tau_int=1)
            * Alternative: infer with (tau_mid > 1) train with (tau_int < 0)
        - (tau >= 2)
            - Infer with (tau_mid >= 1)
            - Train with fractional (tau_int > .2)
    - Output and residual stepping
        - (tau = 1)
            - Infer with (tau_mid=1)
            - Train with (tau_int=0)
        - (tau >= 2)
            - Infer with (tau_mid >= 1)
            - Train with fractional (tau_int > 0)

# NEXT STEPS

- Check E614
- Check E615
- Check E81x results
- Check learning curves of Poisson-Gauss and run with smaller LR if necessary

- Repeat E613 with new baseline architecture
    - OF=1.5
    - RML=6  # Fewer MPS
    - MPS=(18 or 12)
- Tune number of epochs with the new hyperparameters
- Experiment with the new settings (all datasets)

## SOME UNANSWERED QUESTIONS

- Why NS-SVS and NS-Sines do not generalize on time??

# Future work

## Experiments

- Add more discretization invariance tests
    1. Shuffle nodes and edges
    2. Multiple random sub-samples of the full mesh
    3. Super-resolution and sub-resolution
    4. Trained on grid, validated on unstructured mesh
    5. Different x_inp and x_out (not supported currently)

## Literature

- Quick overview of the recent literature
- Read gladstone2024mesh + li2020multipole and present them to Mishra
    - Read them with details and be careful !
    - The ideas are VERY similar and the performance is close
    - Maybe we need to benchmark gladstone2024mesh too
    - We should be careful with what we focus on:
        - a possibility is focusing on this new paradigm for down and up-scaling layers

## Architectural Experiments

- experiment with num_processor_repetitions !!
    - Check overfitting and overall performance
    - Compute parameter efficiency

- Try Encode-FNO/CNO-Decode (exactly like GINO)

## Uncertainty

- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking
- Inspect uncertainty over rollout steps
- Inspect uncertainty with different tau
- Inspect uncertainty with extrapolation


## Benchmarks

- Compare model size
    - Number of parameters
    - FLOPs / MADD
    - Inference time (improve your benchmarking)

## General Boundary Conditions
- Extend for general boundary conditions (e.g., open, Robin)
- Impose Dirichlet boundary conditions differently

## Data Augmentation

- Try shifting first (approved)
- Try repeating (physically incorrect)

## Variable known parameters

- Extend autoregressive and unrollings to variable c

## Variable mesh

- Check RIGNO.variable_mesh and try it
- Extend autoregressive and unrollings to variable x

## Multi-level RIGNO:
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

## Interpretation

- More systematic approaches
    - engineered input
        - Single Fourier mode
        - Single Riemann problems
    - Perturbed input
        - keep/remove Fourier modes
        - Perturb IC parameters

## Pre-training
- Read thesis

## Adaptive inference
- Adaptive time step
- Adaptive remeshing

## Sequential data

- Implement and try the LSTM idea from presentation-240315-updates

- Simpler than LSTM: Just sum the hidden mesh nodes before decoder

- Check LLM tasks (translation, prompt-based text generation, auto-completion)

- Check RNNs and transformers (the whole training scheme changes.. and becomes faster!!)


# Code and Performance

- Get rid of the NVIDIA driver compatiblity message: parallel compilation possibly faster

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.
