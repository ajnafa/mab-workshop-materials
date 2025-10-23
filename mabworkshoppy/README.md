# Python Code for Multi-Arm Bandits Workshop

This folder contains the python code for implementing different
approaches to multi-arm bandit algorithms based on Thompson Sampling. 
It requires the [uv tool](https://docs.astral.sh/uv/) and virtual environment and dependency management.

## Getting Started

The following instructions assume you are starting at the root 
directory of the workshop folder. First, navigate to the `mabworkshoppy` subfolder.

```bash
cd mabworkshoppy
```

Next, create and activate the virtual environment using `uv` with the specified Python version:

```bash
uv venv --python 3.12
.venv\Scripts\activate # Windows
# source .venv/bin/activate # macOS/Linux
```

Finally, install the required dependencies:

```bash
uv sync
```

## Running the Code

To run the simple Thompson Sampling demo, execute the following command:

```bash
uv run ts_demo.py
```