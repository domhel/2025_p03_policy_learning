# 2025_p03_policy_learning
Students: Alexander Lehrmann (3821464), Dominik Helfenstein (3401790), Sindre Myklebost Moldsvor (3817591)

## Quick Start

1. Make sure you got the dataset from [IPA at HuggingFace](https://huggingface.co/datasets/ipa-intelligent-mobile-manipulators/studytable_open_drawer). Make sure you have the LFS (large file system) extension installed for git. (Not sure if it can be loaded directly as a submodule)
1. Install [uv](https://github.com/astral-sh/uv) package manager for Python
1. Run `uv sync` to get the project's dependencies
1. Run `uv run 00_example_depth.py` or run it in VSCode which activates the virtual env automatically. This example will render the first video of the dataset as a depth estimation using matplotlib.