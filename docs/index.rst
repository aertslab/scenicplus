Welcome to SCENIC+'s documentation!
===================================

SCENIC+ is a python package to build enhancer driven gene regulatory networks (GRNs) using combined or separate single-cell
gene expression (scRNA-seq) and single-cell chromatin accessibility (scATAC-seq) data.

SCENIC+ makes use of several python packages:

* pycisTopic for enhancer candidate identification and topic modeling (`read the docs page <https://pycistopic.readthedocs.io/en/latest/>`_).
* pycistarget for motif enrichment analysis in enhancer candidates (`read the docs page <https://pycistarget.readthedocs.io/en/latest/>`_).
* create_cisTarget_databases for creating of custom cistarget databases (`github page <https://github.com/aertslab/create_cisTarget_databases>`_)

.. toctree::
   :hidden:

   install
   tutorials
   faqs
   api


SCENIC+ Flow chart
------------------
.. image:: _images/flow_chart_SCENIC+.png
   :width: 800
   :align: center
   :alt: SCENIC+ flow chart
