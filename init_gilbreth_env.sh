#!/bin/bash

# WHAT: Purge old loaded modules and load Gilbreth's central resources.
# WHY: Prevents conflicts between default loaded OS libraries and the ones we need for PyTorch.
module purge
module load anaconda cuda

# WHAT: Clean Conda's cache before building.
# WHY: If a previous installation failed mid-download, the corrupted tarball stays in the cache. 
#      This forces Conda to pull fresh, clean packages.
echo "Cleaning Conda cache..."
conda clean --all -y

# WHAT: Build the environment strictly from the declarative YAML file.
# WHY: This creates the `mrn` environment with all dependencies resolved mathematically at once.
echo "Building environment from conda_env.yml..."
conda env create -f conda_env.yml

# WHAT: Export the Conda Library Path (The Gilbreth GLIBCXX Fix)
# WHY: HPC clusters like Gilbreth often run older enterprise Linux kernels. 
#      Modern pip packages will crash looking for GLIBCXX. This tells the system to use 
#      Conda's modern C++ standard library instead of the outdated OS library.
conda activate mrn
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo "Initialization complete. Run the confirmation tests."