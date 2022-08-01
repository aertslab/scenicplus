# SCENIC+

SCENIC+ is a python package to build gene regulatory networks (GRNs) using combined or separate single-cell gene expression (scRNA-seq) and single-cell chromatin accessibility (scATAC-seq) data.

## Installing

To install SCENIC+:

```bash
git clone https://github.com/aertslab/scenicplus
cd scenicplus
pip install .
```

Depending on your pip version, you may need to run this pip command instead:

```bash
pip install -e .
```

## Creating a Docker/Singularity Image

To build a Docker image, and then create a Singularity image from this:

```bash
# Clone repositories 
git clone https://github.com/aertslab/pySCENIC.git
git clone https://github.com/aertslab/LoomXpy.git
git clone https://github.com/aertslab/pycisTopic.git
git clone https://github.com/aertslab/pycistarget.git
git clone https://github.com/aertslab/scenicplus.git

# Login
podman login docker.io

# Build image
podman build -t aertslab/scenicplus:latest . -f scenicplus/Dockerfile

# Export to oci 
podman save --format oci-archive --output scenicplus_img.tar localhost/aertslab/scenicplus

# Build to singularity
singularity build scenicplus.sif oci-archive://scenicplus_img.tar

# Add all binding paths where you would need to access
singularity exec -B /lustre1,/staging,/data,/vsc-hard-mounts,/scratch scenicplus.sif ipython3
```

## Check version

To check your SCENIC+ version:

```bash
import scenicplus
scenicplus.__version__
```

## Documentation

Documentation is available at: scenicplus.readthedocs.io


## References
1. Bravo Gonzalez-Blas, C. *et al.* (2020). Identification of genomic enhancers through spatial integration of single-cell transcriptomics and epigenomics. [Molecular Systems Biology](https://www.embopress.org/doi/full/10.15252/msb.20209438)

2. Janssens, J., Aibar, S., Taskiran, I.I. *et al.* (2022) Decoding gene regulation in the fly brain. [Nature](https://www.nature.com/articles/s41586-021-04262-z)
