# Setting Up Your Environment with uv

This guide will walk you through installing `uv` (the fast Python package installer and resolver) and creating a Python virtual environment for Evolution Gym and related packages.

## Installing uv

### For Windows
Open PowerShell and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### For macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or reload your shell configuration to ensure the `uv` command is available.

## Step-by-Step Environment Setup

### 1. Initialize uv
```bash
uv init
```
This initializes uv in your current directory.

### 2. Create a Virtual Environment
```bash
uv venv --python=python3.10
```
This creates a virtual environment using Python 3.10.

### 3. Install Required Packages
Run the following commands to install all necessary packages:

```bash
uv add evogym
uv add ipykernel -U --force-reinstall
uv add torch
uv add matplotlib
uv add --upgrade setuptools
uv add tqdm
uv add imageio
```

### Package Details

- **evogym**: Evolution Gym framework
- **ipykernel**: Kernel for Jupyter notebooks
- **torch**: PyTorch machine learning library
- **matplotlib**: Data visualization library
- **setuptools**: Package development tools
- **tqdm**: Progress bar utility
- **imageio**: Library for reading and writing image data

## Activation and Usage

### For Windows
```powershell
.venv\Scripts\activate
```

### For macOS/Linux
```bash
source .venv/bin/activate
```

After activation, you can start using the installed packages in your Python scripts or Jupyter notebooks.

For any issues with the environment setup, check the [uv documentation](https://github.com/astral-sh/uv) or package-specific resources.