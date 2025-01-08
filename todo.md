# TODOs

- Compute loss on the validation set too
- Do a mini validation more often and use it for best_fn

## Code and Performance

- Update the logging formatter

- Get rid of the NVIDIA driver compatiblity message: parallel compilation possibly faster

- Add setup.py with setuptools and read use cases:
    - %pip install --upgrade https://github.com/.../master.zip

- Write docstring and type annotations

- Try to understand why without preloading the dataset, loading batches takes longer with more GPUs or a larger model.
