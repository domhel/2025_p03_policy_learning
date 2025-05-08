# 2025_p03_policy_learning
Students: Alexander Lehrmann (3821464), Dominik Helfenstein (3401790), Sindre Myklebost Moldsvor (3817591)

## Quick Start

1. Make sure you got the dataset from [IPA at HuggingFace](https://huggingface.co/datasets/ipa-intelligent-mobile-manipulators/studytable_open_drawer). Make sure you have the LFS (large file system) extension installed for git. (Not sure if it can be loaded directly as a submodule)
1. Install [uv](https://github.com/astral-sh/uv) package manager for Python
1. Run `uv sync` to get the project's dependencies
1. Run `uv run 00_example_depth.py` or run it in VSCode which activates the virtual env automatically. This example will render the first video of the dataset as a depth estimation using matplotlib.


## Strategy:
	Agenda 08.05.: 
   - Umsetzung Data Import wie in E-Mail beschrieben
	• Entwurf der Sub-Schritte

· Milestone 1: 
	• Apply monocular depth estimation to our dataset and generate a new dataset. It should include 3 original images and their estimated depth images. In total 6 images. 
	• Liefergegenstand: 
		○ Ordner mit 6 subfoldern, 3x orig, 3x mp4 Video mit depth Estimation
	• Subtask:
		○ Pipeline()
· Milestone 2: 
	• Apply color-based segmentation on the original dataset and create a new dataset. The color segmentation should assign 1(255) to drawer handle and the cube. The rest will be 0. 
· Milestone 3:
 Train and test the models in the simulation. 30 times each. Performance metric: success rate ![image](https://github.com/user-attachments/assets/253cf999-4613-4fc2-9fbe-61a9db00b78a)
