.. _install:

*******
Install
*******

Regular install
===============

To install SCENIC+ run:

.. code-block:: bash

    git clone https://github.com/aertslab/scenicplus
    cd scenicplus
    pip install .

Depending on your pip version, you may need to run this pip command instead:

.. code-block:: bash

    pip install -e .

Creating a Docker/Singularity image
===================================

To build a Docker image, and then create a Singularity image from this:

.. code-block:: bash

    # Clone repositories 
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

Checking version
================

To check your SCENIC+ version

.. code-block:: python

    import scenicplus
    scenicplus.__version__

