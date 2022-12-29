**************
FAQs
**************


1. I have a cisTopic analysis in R, can I still use SCENIC+?
=============================================================

Yes, you can! There is no need to repeat the topic modelling in Python. For this, you will need to do the following in R first::


	# Export data to python
	library(arrow)
	path <- OUTDIR
	cisTopic_obj <- readRDS(PATH_TO_YOUR_CISTOPICOBJECT)
	modelMat <- modelMatSelection(cisTopic_obj, 'cell', 'Probability')
	modelMat <- as.data.frame(modelMat)
	write_feather(modelMat, sink=paste0(path, 'model_to_pycisTopic/cell_topic.feather'))
	modelMat <- modelMatSelection(cisTopic_obj, 'region', 'Probability', all.regions=TRUE)
	modelMat <- as.data.frame(modelMat)
	write_feather(modelMat, sink=paste0(path, 'model_to_pycisTopic/topic_region.feather'))
	# You can also save the count.matrix
	ct <- cisTopicObject@count.matrix
	ct <- as.data.frame(ct)
	write_feather(ct, sink=paste0(path, 'model_to_pycisTopic/count_matrix.feather'))


And then in Python::



	import pycisTopic
	## 1. Initialize cisTopic object
	from pycisTopic.cistopic_class import *
	# Load count matrix
	matrix_path=PATH_TO_FRAGMENTS_MATRIX
	fragment_matrix = pd.read_feather(matrix_path)
	cisTopic_obj = create_cistopic_object(fragment_matrix)
	# Also add the cell annotation, cell_data should be a pandas df with cells as rows (cell names as index) and variables as columns
	cisTopic_obj.add_cell_data(cell_data)
	## 2. Add model
	from pycisTopic.utils import *
	model = load_cisTopic_model(PATH_TO_THE_FOLDER_WHERE_YOU_SAVED_DATA_FROM_R)
	cistopic_obj.add_LDA_model(model)
	# You can continue with the rest of the pipeline.


Feel free to adapt this function if you do not want to work in feather format::


	def load_cisTopic_model(path_to_cisTopic_model_matrices):
		metrics = None
		coherence = None
		marg_topic = None
		topic_ass = None
		cell_topic = pd.read_feather(path_to_cisTopic_model_matrices + "cell_topic.feather")
		cell_topic.index = ["Topic" + str(x) for x in range(1, cell_topic.shape[0] + 1)]
		topic_region = pd.read_feather(
			path_to_cisTopic_model_matrices + "topic_region.feather"
		)
		topic_region.index = ["Topic" + str(x) for x in range(1, topic_region.shape[0] + 1)]
		topic_region = topic_region.T
		parameters = None
		model = cisTopicLDAModel(
			metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
		)
		return model

2. I have an analysis with another tool (e.g. Signac/ArchR), can I still use SCENIC+?
=====================================================================================

Here you have two options:

* **Start from the count matrix**: If you have performed QC steps and generated your fragment count matrix with another tool, you can directly start with the creation of the cisTopic object. In `this tutorial <https://pycistopic.readthedocs.io/en/latest/Toy_melanoma-RTD.html>`_ you have an example on how to do this.
* **Skip the topic modelling**: We do not recommend this option as we have not tested it, but technically it is possible. You would need to get 1) a denoised fragments count matrix and 2) region sets. The region sets will be used as input for cisTarget. The region sets could be Differentially Accessible Regions (DARs); however we do recommend to use topics as well as they can capture regions with accessibility patterns without the need of specifying contrasts. If only using DARs you will have to make sure to make the right contrats (for example, all cell types versus the rest; higher cell type groups versus rest, etc). For example, in the cortex data set the topics reveal a topic of regions shared across all neurons; this pattern would be lost if you only do contrasts per cell type and not higher groups. In conclusion, it is possible to plug it in code-wise; but we can't assure results will be as good as the workflow we propose.

3. I have data from another species then human, mouse or fruit fly. Can I still use SCENIC+?
============================================================================================

Yes, you can but it does require some extra work.

* For motif enrichment analysis you will need a custom motif-to-TF annotation, this annotation should look like [this](https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl). Feel free to open a discussions post if you have trouble generating this file.
* The `run_pycistarget` function should be run with the parameter `species` set to `'custom'` and the custom transcriptome annotation should be given using the parameter `custom_annot`. This should be a Pandas Dataframe with the following columns: `Chromosome`, `Start`, `Strand`, `Gene` and `Transcript_type`, which only contains **protein coding** genes.
* Before running the `run_scenicplus` wrapper function the user should run `get_search_space` manually with custom annotation and chromosome sizes provided using the parameters `pr_annot` and `pr_chromsizes`. For more information, please check out the pbmc tutorial.