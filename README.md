![alt text](docs/images/SCENIC+_Logo_v5.png "SCENIC+")
[![Documentation Status](https://readthedocs.org/projects/scenicplus/badge/?version=development)](https://scenicplus.readthedocs.io/?badge=main)


# SCENIC+ single-cell eGRN inference

`SCENIC+` is a python package to build gene regulatory networks (GRNs) using combined or separate single-cell gene expression (scRNA-seq) and single-cell chromatin accessibility (scATAC-seq) data.

> [!NOTE]  
> **Updates:**
> 
> August 30th 2024: Perturbation simulation got updated, please find tutorial [here](https://scenicplus.readthedocs.io/en/latest/Perturbation_simulation.html#Tutorial:-Perturbation-simulation)

> [!TIP]
> We did a live webinar on using the SCENIC+ workflow. You can rewatch it on [YouTube](https://www.youtube.com/watch?v=QW63LLd1XC8)

## Documentation 

Extensive documentation and tutorials are available at [read the docs](https://scenicplus.readthedocs.io/).

## Installing

### Recommended: Install with uv (Fast & Cross-Platform)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It's the recommended way to install SCENIC+ as it handles dependencies efficiently and works on macOS, Linux, and Windows.

**Requirements:**
- Python 3.11 (Python 3.12 not yet supported due to dependency incompatibilities)
- HDF5 library (see platform-specific instructions in [UV_SETUP.md](UV_SETUP.md))

**Quick Start:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/aertslab/scenicplus
cd scenicplus

# Install dependencies and SCENIC+
uv sync

# Run Python with uv
uv run python your_script.py
```

For detailed setup instructions, troubleshooting, and platform-specific requirements, see [UV_SETUP.md](UV_SETUP.md).

### Alternative: Install with conda/pip

If you prefer conda, you can install SCENIC+ in a new conda environment:

```bash
conda create --name scenicplus python=3.11 -y
conda activate scenicplus
git clone https://github.com/aertslab/scenicplus
cd scenicplus
pip install .
```

## Questions?

* If you have **technical questions or problems**, such as bug reports or ideas for new features, please open an issue under the issues tab.
* If you have **questions about the interpretation of results or your analysis**, please start a Discussion under the Discussions tab.


## References

[Bravo Gonzalez-Blas, C. & De Winter, S. *et al.* (2022). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks](https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1)

## license

The SCENIC+ [license](https://github.com/aertslab/scenicplus/blob/main/LICENCE.txt) covers [SCENIC+](https://github.com/aertslab/scenicplus), [pycisTarget](https://github.com/aertslab/pycistarget), [pycisTopic](https://github.com/aertslab/pycisTopic) and the [cisTarget dabases](https://resources.aertslab.org/cistarget/).
