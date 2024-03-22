![alt text](docs/images/SCENIC+_Logo_v5.png "SCENIC+")
[![Documentation Status](https://readthedocs.org/projects/scenicplus/badge/?version=development)](https://scenicplus.readthedocs.io/en/development/?badge=development)


# SCENIC+ single-cell eGRN inference

`SCENIC+` is a python package to build gene regulatory networks (GRNs) using combined or separate single-cell gene expression (scRNA-seq) and single-cell chromatin accessibility (scATAC-seq) data.

## Documentation 

Extensive documentation and tutorials are available at [read the docs](https://scenicplus.readthedocs.io/en/development/).

## Installing

To install SCENIC+ (in a Linux environment):

We highly recommend to install SCENIC+ in a new conda environment.

```bash

$ conda create --name scenicplus python=3.11 -y
$ conda activate scenicplus
$ git clone https://github.com/aertslab/scenicplus
$ cd scenicplus
$ git checkout development
$ pip install .

```

## Questions?

* If you have **technical questions or problems**, such as bug reports or ideas for new features, please open an issue under the issues tab.
* If you have **questions about the interpretation of results or your analysis**, please start a Discussion under the Discussions tab.


## References

[Bravo Gonzalez-Blas, C. & De Winter, S. *et al.* (2022). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks](https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1)
