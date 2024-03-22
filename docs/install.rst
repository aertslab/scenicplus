.. _install:

*******
Install
*******

Installing in a conda environment (recommended)
===============================================

We highly recommend installing SCENIC+ in a new conda environment.

Create new environment:

.. code-block:: bash

   $ conda create --name scenicplus python=3.11


Install SCENIC+ in environment:

.. code-block:: bash

   $ conda activate scenicplus
   $ git clone https://github.com/aertslab/scenicplus
   $ cd scenicplus
   $ git checkout development
   $ pip install .

Checking version
================

To check your SCENIC+ version

.. code-block:: python

    >>> import scenicplus
    >>> scenicplus.__version__
    '1.0a1'


