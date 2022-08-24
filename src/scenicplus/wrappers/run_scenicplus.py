"""Wrapper functions to run SCENIC+ analysis

After the SCENIC+ object has been generated the complete SCENIC+ workflow can be run with a single function. 

Following operation will be done:

1. Defining a search space surounding each gene
2. Region to gene linking & TF to gene linking
3. eGRN building
4. eGRN scoring (target gene and region based AUC and RSS)
5. Calculating TF-cistrome correlations
6. Dimensionality reductions
7. export to UCSC tracks and to loom file

In case the process is killed prematurely, the function can be restared and the workflow will resume from the last step that was succesfully completed.

"""


from scenicplus.scenicplus_class import SCENICPLUS, create_SCENICPLUS_object
from scenicplus.preprocessing.filtering import *
from scenicplus.cistromes import *
from scenicplus.enhancer_to_gene import get_search_space, calculate_regions_to_genes_relationships, GBM_KWARGS
from scenicplus.enhancer_to_gene import export_to_UCSC_interact 
from scenicplus.utils import format_egrns, export_eRegulons
from scenicplus.eregulon_enrichment import *
from scenicplus.TF_to_gene import *
from ..grn_builder.gsea_approach import build_grn
from scenicplus.dimensionality_reduction import *
from scenicplus.RSS import *
from scenicplus.diff_features import *
from scenicplus.loom import *
from typing import Dict, List, Mapping, Optional, Sequence
import os
import dill
import time


def run_scenicplus(scplus_obj: 'SCENICPLUS',
    variable: List[str],
    species: str,
    assembly: str,
    tf_file: str,
    save_path: str,
    biomart_host: Optional[str] = 'http://www.ensembl.org',
    upstream: Optional[List] = [1000, 150000],
    downstream: Optional[List] = [1000, 150000],
    region_ranking: Optional['CisTopicImputedFeatures'] = None,
    gene_ranking: Optional['CisTopicImputedFeatures'] = None,
    simplified_eGRN: Optional[bool] = False,
    calculate_TF_eGRN_correlation: Optional[bool] = True,
    calculate_DEGs_DARs: Optional[bool] = True,
    export_to_loom_file: Optional[bool] = True,
    export_to_UCSC_file: Optional[bool] = True,
    tree_structure: Sequence[str] = (),
    path_bedToBigBed: Optional[str] = None,
    n_cpu: Optional[int] = 1,
    _temp_dir: Optional[str] = '',
    **kwargs
    ):
    """
    Wrapper to run SCENIC+
    
    Parameters
    ---------
    scplus_obj: `class::SCENICPLUS`
        A SCENICPLUS object.
    variables: List[str]
        Variables to use for RSS, TF-eGRN correlation and markers.
    species: str
        Species from which data comes from. Possible values: 'hsapiens', 'mmusculus', 'dmelanogaster'
    assembly: str
        Genome assembly to which the data was mapped. Possible values: 'hg38'
    tf_file: str
        Path to file containing genes that are TFs
    save_path: str
        Folder in which results will be saved
    biomart_host: str, optional
        Path to biomart host. Make sure that the host matches your genome assembly
    upstream: str, optional
        Upstream space to use for region to gene relationships
    downstream: str, optional
        Upstream space to use for region to gene relationships
    region_ranking: `class::CisTopicImputedFeatures`, optional
        Precomputed region ranking
    gene_ranking: `class::CisTopicImputedFeatures`, optional
        Precomputed gene ranking
    simplified_eGRN: bool, optional
        Whether to output simplified eGRNs (only TF-G sign rather than TF-G_R-G)
    calculate_TF_eGRN_correlation: bool, optional
        Whether to calculate the TF-eGRN correlation based on the variables
    calculate_DEGs_DARs: bool, optional
        Whether to calculate DARs/DEGs based on the variables
    export_to_loom_file: bool, optional
        Whether to export data to loom files (gene based/region based)
    export_to_UCSC_file: bool, optional
        Whether to export region-to-gene links and eregulons to bed files
    tree_structure: sequence, optional
        Tree structure for loom files
    path_bedToBigBed: str, optional
        Path to convert bed files to big bed when exporting to UCSC (required if files are meant to be
        used in a hub)
    n_cpu: int, optional
        Number of cores to use
    _temp_dir: str, optional
        Temporary directory for ray
    """
    
    # Create logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('SCENIC+_wrapper')
    
    start_time = time.time()
    
    check_folder = os.path.isdir(save_path)
    if not check_folder:
        os.makedirs(save_path)
        log.info("Created folder : "+ save_path)

    else:
        log.info(save_path + " folder already exists.")
    
    if 'Cistromes' not in scplus_obj.uns.keys():
        log.info('Merging cistromes')
        merge_cistromes(scplus_obj)
    
    
    if 'search_space' not in scplus_obj.uns.keys():
        log.info('Getting search space')
        get_search_space(scplus_obj,
                     biomart_host = biomart_host,
                     species = species,
                     assembly = assembly, 
                     upstream = upstream,
                     downstream = downstream)
                 
    if 'region_to_gene' not in scplus_obj.uns.keys():
        log.info('Inferring region to gene relationships')
        calculate_regions_to_genes_relationships(scplus_obj, 
                        ray_n_cpu = n_cpu, 
                        _temp_dir = _temp_dir,
                        importance_scoring_method = 'GBM',
                        importance_scoring_kwargs = GBM_KWARGS,
                        **kwargs)
                        
    if 'TF2G_adj' not in scplus_obj.uns.keys():
        log.info('Inferring TF to gene relationships')
        calculate_TFs_to_genes_relationships(scplus_obj, 
                        tf_file = tf_file,
                        ray_n_cpu = n_cpu, 
                        method = 'GBM',
                        _temp_dir = _temp_dir,
                        key= 'TF2G_adj',
                        **kwargs)
                        
    if 'eRegulons' not in scplus_obj.uns.keys():
        log.info('Build eGRN')
        build_grn(scplus_obj,
                 min_target_genes = 10,
                 adj_pval_thr = 1,
                 min_regions_per_gene = 0,
                 quantiles = (0.85, 0.90, 0.95),
                 top_n_regionTogenes_per_gene = (5, 10, 15),
                 top_n_regionTogenes_per_region = (),
                 binarize_using_basc = True,
                 rho_dichotomize_tf2g = True,
                 rho_dichotomize_r2g = True,
                 rho_dichotomize_eregulon = True,
                 rho_threshold = 0.05,
                 keep_extended_motif_annot = True,
                 merge_eRegulons = True, 
                 order_regions_to_genes_by = 'importance',
                 order_TFs_to_genes_by = 'importance',
                 key_added = 'eRegulons',
                 cistromes_key = 'Unfiltered',
                 disable_tqdm = False, 
                 ray_n_cpu = n_cpu,
                 _temp_dir = _temp_dir,
                 **kwargs)
                 
    if 'eRegulon_metadata' not in scplus_obj.uns.keys():
        log.info('Formatting eGRNs')
        format_egrns(scplus_obj,
                      eregulons_key = 'eRegulons',
                      TF2G_key = 'TF2G_adj',
                      key_added = 'eRegulon_metadata')

                    
    if 'eRegulon_signatures' not in scplus_obj.uns.keys():
        log.info('Converting eGRNs to signatures')
        get_eRegulons_as_signatures(scplus_obj,
                                     eRegulon_metadata_key='eRegulon_metadata', 
                                     key_added='eRegulon_signatures')
                                     
    if simplified_eGRN is True:
        md = scplus_obj.uns['eRegulon_signatures']['Gene_based']
        names = list(set([x.split('_(')[0][:len(x.split('_(')[0]) - 2] for x in md.keys()]))
        scplus_obj.uns['eRegulon_signatures']['Gene_based'] = {x:list(set(sum([value for key, value in md.items() if key.startswith(x)], []))) for x in names}
        scplus_obj.uns['eRegulon_signatures']['Gene_based'] = {x+'_('+str(len(scplus_obj.uns['eRegulon_signatures']['Gene_based'][x]))+'g)': scplus_obj.uns['eRegulon_signatures']['Gene_based'][x] for x in scplus_obj.uns['eRegulon_signatures']['Gene_based'].keys()}

        md = scplus_obj.uns['eRegulon_signatures']['Region_based']
        names = list(set([x.split('_(')[0][:len(x.split('_(')[0]) - 2] for x in md.keys()]))
        scplus_obj.uns['eRegulon_signatures']['Region_based'] = {x:list(set(sum([value for key, value in md.items() if key.startswith(x)], []))) for x in names}
        scplus_obj.uns['eRegulon_signatures']['Region_based'] = {x+'_('+str(len(scplus_obj.uns['eRegulon_signatures']['Region_based'][x]))+'r)': scplus_obj.uns['eRegulon_signatures']['Region_based'][x] for x in scplus_obj.uns['eRegulon_signatures']['Region_based'].keys()}

    
    if 'eRegulon_AUC' not in scplus_obj.uns.keys():
        log.info('Calculating eGRNs AUC')
        if region_ranking is None:
            log.info('Calculating region ranking')
            region_ranking = make_rankings(scplus_obj, target='region')
            with open(os.path.join(save_path,'region_ranking.pkl'), 'wb') as f:
                dill.dump(region_ranking, f, protocol = -1)
        log.info('Calculating eGRNs region based AUC')
        score_eRegulons(scplus_obj,
                ranking = region_ranking,
                eRegulon_signatures_key = 'eRegulon_signatures',
                key_added = 'eRegulon_AUC', 
                enrichment_type= 'region',
                auc_threshold = 0.05,
                normalize = False,
                n_cpu = n_cpu)
        if gene_ranking is None:
            log.info('Calculating gene ranking')
            gene_ranking = make_rankings(scplus_obj, target='gene')
            with open(os.path.join(save_path,'gene_ranking.pkl'), 'wb') as f:
                dill.dump(gene_ranking, f, protocol = -1)
        log.info('Calculating eGRNs gene based AUC')
        score_eRegulons(scplus_obj,
                gene_ranking,
                eRegulon_signatures_key = 'eRegulon_signatures',
                key_added = 'eRegulon_AUC', 
                enrichment_type = 'gene',
                auc_threshold = 0.05,
                normalize= False,
                n_cpu = n_cpu)
                
    if calculate_TF_eGRN_correlation is True:
        log.info('Calculating TF-eGRNs AUC correlation')
        for var in variable:
            generate_pseudobulks(scplus_obj, 
                             variable = var,
                             auc_key = 'eRegulon_AUC',
                             signature_key = 'Gene_based',
                             nr_cells = 5,
                             nr_pseudobulks = 100,
                             seed=555)
            generate_pseudobulks(scplus_obj, 
                                     variable = var,
                                     auc_key = 'eRegulon_AUC',
                                     signature_key = 'Region_based',
                                     nr_cells = 5,
                                     nr_pseudobulks = 100,
                                     seed=555)
            TF_cistrome_correlation(scplus_obj,
                            variable = var, 
                            auc_key = 'eRegulon_AUC',
                            signature_key = 'Gene_based',
                            out_key = var+'_eGRN_gene_based')
            TF_cistrome_correlation(scplus_obj,
                                    variable = var, 
                                    auc_key = 'eRegulon_AUC',
                                    signature_key = 'Region_based',
                                    out_key = var+'_eGRN_region_based')
                                
    if 'eRegulon_AUC_thresholds' not in scplus_obj.uns.keys():
        log.info('Binarizing eGRNs AUC')
        binarize_AUC(scplus_obj, 
             auc_key='eRegulon_AUC',
             out_key='eRegulon_AUC_thresholds',
             signature_keys=['Gene_based', 'Region_based'],
             n_cpu=n_cpu)
             
    if not hasattr(scplus_obj, 'dr_cell'):
        scplus_obj.dr_cell = {}         
    if 'eRegulons_UMAP' not in scplus_obj.dr_cell.keys():
        log.info('Making eGRNs AUC UMAP')
        run_eRegulons_umap(scplus_obj,
                   scale=True, signature_keys=['Gene_based', 'Region_based'])
    if 'eRegulons_tSNE' not in scplus_obj.dr_cell.keys():
        log.info('Making eGRNs AUC tSNE')
        run_eRegulons_tsne(scplus_obj,
                   scale=True, signature_keys=['Gene_based', 'Region_based'])
                   
    if 'RSS' not in scplus_obj.uns.keys():
        log.info('Calculating eRSS')
        for var in variable:
            regulon_specificity_scores(scplus_obj, 
                         var,
                         signature_keys=['Gene_based'],
                         out_key_suffix='_gene_based',
                         scale=False)
            regulon_specificity_scores(scplus_obj, 
                         var,
                         signature_keys=['Region_based'],
                         out_key_suffix='_region_based',
                         scale=False)
                         
    if calculate_DEGs_DARs is True:
        log.info('Calculating DEGs/DARs')
        for var in variable:
            get_differential_features(scplus_obj, var, use_hvg = True, contrast_type = ['DEGs', 'DARs'])
            
    if export_to_loom_file is True:
        log.info('Exporting to loom file')
        export_to_loom(scplus_obj, 
               signature_key = 'Gene_based',
               tree_structure = tree_structure,
               title =  'Gene based eGRN',
               nomenclature = assembly,
               out_fname=os.path.join(save_path,'SCENIC+_gene_based.loom'))
        export_to_loom(scplus_obj, 
               signature_key = 'Region_based',
               tree_structure = tree_structure,
               title =  'Region based eGRN',
               nomenclature = assembly,
               out_fname=os.path.join(save_path,'SCENIC+_region_based.loom'))
               
    if export_to_UCSC_file is True:
        log.info('Exporting to UCSC')
        r2g_data = export_to_UCSC_interact(scplus_obj,
                            species,
                            os.path.join(save_path,'r2g.rho.bed'),
                            path_bedToBigBed=path_bedToBigBed,
                            bigbed_outfile=os.path.join(save_path,'r2g.rho.bb'),
                            region_to_gene_key='region_to_gene',
                            pbm_host=biomart_host,
                            assembly=assembly,
                            ucsc_track_name='R2G',
                            ucsc_description='SCENIC+ region to gene links',
                            cmap_neg='Reds',
                            cmap_pos='Greens',
                            key_for_color='rho',
                            scale_by_gene=False,
                            subset_for_eRegulons_regions=True,
                            eRegulons_key='eRegulons')
        r2g_data = export_to_UCSC_interact(scplus_obj,
                            species,
                            os.path.join(save_path,'r2g.importance.bed'),
                            path_bedToBigBed=path_bedToBigBed,
                            bigbed_outfile=os.path.join(save_path,'r2g.importance.bb'),
                            region_to_gene_key='region_to_gene',
                            pbm_host=biomart_host,
                            assembly=assembly,
                            ucsc_track_name='R2G',
                            ucsc_description='SCENIC+ region to gene links',
                            cmap_neg='Reds',
                            cmap_pos='Greens',
                            key_for_color='importance',
                            scale_by_gene=True,
                            subset_for_eRegulons_regions=True,
                            eRegulons_key='eRegulons')
        regions = export_eRegulons(scplus_obj,
                os.path.join(save_path,'eRegulons.bed'),
                assembly,
                bigbed_outfile = os.path.join(save_path,'eRegulons.bb'),
                eRegulon_metadata_key = 'eRegulon_metadata',
                eRegulon_signature_key = 'eRegulon_signatures',
                path_bedToBigBed=path_bedToBigBed)
        
    log.info('Saving object')         
    with open(os.path.join(save_path,'scplus_obj.pkl'), 'wb') as f:
        dill.dump(scplus_obj, f, protocol = -1)

    log.info('Finished! Took {} minutes'.format((time.time() - start_time)/60))       
    
    
    
    
