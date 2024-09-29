# Project Updates

- Limitations
    - Speed
    - Graph construction for non-periodic BC
        - Needs better methods
    - Imposing boundary conditions
        - Can improve performance a lot

# NEXT STEPS

- Check E617
    - Replace the E612 table with plots
    - Replace E615 plots

- Check E821 (NS-SL and SVS) with 2500 epochs -- is it any better?
- Update data scaling with E82x (?)

- Write the updates on fractional pairing strategy on the updates document

## SOME UNANSWERED QUESTIONS

- Why NS-SVS and NS-Sines do not generalize on time??

# Future work

## Architectural Experiments

- experiment with num_processor_repetitions !!
    - Check overfitting and overall performance
    - Compute parameter efficiency

## Uncertainty

- Add uncertainty to the errors
    * No need to retrain anything, just use edge masking
- Inspect uncertainty over rollout steps
- Inspect uncertainty with different tau
- Inspect uncertainty with extrapolation


## General Boundary Conditions
- Extend for general boundary conditions (e.g., open, Robin)
- Impose Dirichlet boundary conditions differently

## Data Augmentation

- Try shifting first (approved)
- Try repeating (physically incorrect)

## Thorough study on the time stepping strategies

- A more general approach using the SOTA neural operators

- Tests
    - Continuity in time tests
    - Fractional time delta
    - Inter- and extrapolation in t (from both ends)
    - Inter- and extrapolation in tau (from both ends)

- Fractional pairing strategy
    - Fine-tuning or from scratch?
    - Warm-up when fine-tuning?
    - Try to assure robustness


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
