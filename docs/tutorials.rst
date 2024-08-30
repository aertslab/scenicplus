.. _tutorials:

**********
Tutorials
**********

In this section we present tutorials on how to run SCENIC+.

Basic SCENIC+ tutorial on human brain multiome data
===================================================

SCENIC+ involves several steps, these are detailed below.
For all the tutorials we will make use of a small dataset (3k cells) freely available
on the `website of 10X genomics <https://www.10xgenomics.com/datasets/frozen-human-healthy-brain-tissue-3-k-1-standard-1-0-0>`_.
This is a multiome dataset on human healthy brain tissues.

**Step 1: preprocess scATAC-seq data using pycisTopic**

The first step is to preprocess the scATAC-seq side of the data using pycisTopic.

Get started with this tutorial by clicking `here <https://pycistopic.readthedocs.io/en/latest/notebooks/human_cerebellum.html>`_ 👈

**Step 2: preprocess scRNA-seq data using Scanpy**

Next we have a very small tutorial on how to preprocess the scRNA-seq side of the data.
This tutorial highligths the bare minimum preprocessing steps. For more information 
we refer the reader to the `Scanpy tutorials <https://scanpy.readthedocs.io/en/stable/tutorials.html>`_.

Get started with this tutorial by clicking :ref:`here <Preprocessing the scRNA-seq data>` 👈

**Step 3: (Optional but recommended) generate a custom cisTarget database**

Optionally, a custom cisTarget database can be generated on consensus peaks.
We provide precomputed cisTarget databases for `human <https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/screen/mc_v10_clust/region_based/>`_, 
`mouse <https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/screen/mc_v10_clust/region_based/>`_ and `fly <https://resources.aertslab.org/cistarget/databases/drosophila_melanogaster/dm6/flybase_r6.02/mc_v10_clust/region_based/>`_ on our resources website. However, using a custom database could produce better results (i.e. 
potentially, more target regions will be discovered).

Get started with this tutorial by clicking :ref:`here <Creating custom cistarget database>` 👈

**Step 4: Run SCENIC+ using SnakeMake**

Finally, we will run the SCENIC+ workflow using SnakeMake.

Get started with this tutorial by clicking :ref:`here <Running SCENIC+>` 👈

Advanced downstream analysis
============================

**Perturbation simulation**

The output of SCENIC+ can be used to simulate TF knockouts.

To see how, click :ref:`here <Tutorial: Perturbation simulation>` 👈


   .. toctree::
    :hidden:
    :maxdepth: 3
    
    human_cerebellum_scRNA_pp.ipynb
    human_cerebellum_ctx_db.ipynb
    human_cerebellum.ipynb
    Perturbation_simulation.ipynb



