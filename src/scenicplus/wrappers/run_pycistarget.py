"""
Wrapper functions to run motif enrichment analysis using pycistarget

After sets of regions have been defined (e.g. topics or DARs). The complete pycistarget workflo can be run using a single function.

this function will run cistarget based and DEM based motif enrichment analysis with or without promoter regions.
"""

from typing import Dict
import pandas as pd
import dill
import pyranges as pr
from pycistarget.motif_enrichment_cistarget import *
from pycistarget.motif_enrichment_dem import *
from pycistarget.utils import *
import pybiomart as pbm
import time

def run_pycistarget(region_sets: Dict[str, pr.PyRanges],
                 species: str,
                 save_path: str,
                 custom_annot: pd.DataFrame = None,
                 save_partial: bool = False,
                 ctx_db_path: str = None,
                 dem_db_path: str = None,
                 run_without_promoters: bool = False,
                 biomart_host: str = 'http://www.ensembl.org',
                 promoter_space: int = 500,
                 ctx_auc_threshold: float = 0.005,
                 ctx_nes_threshold: float = 3.0,
                 ctx_rank_threshold: float = 0.05,
                 dem_log2fc_thr: float = 0.5,
                 dem_motif_hit_thr: float = 3.0,
                 dem_max_bg_regions: int = 500,
                 annotation : List[str] = ['Direct_annot', 'Orthology_annot'],
                 motif_similarity_fdr: float = 0.000001,
                 path_to_motif_annotations: str = None,
                 annotation_version: str = 'v9',
                 n_cpu : int = 1,
                 _temp_dir: str = None,
                 exclude_motifs: str = None,
                 exclude_collection: List[str] = None,
                 **kwargs):
    """
    Wrapper function for pycistarget
    
    Parameters
    ---------
    region_sets: Mapping[str, pr.PyRanges]
         A dictionary of PyRanges containing region coordinates for the region sets to be analyzed.
    species: str
        Species from which genomic coordinates come from, options are: homo_sapiens, mus_musculus, drosophila_melanogaster and gallus_gallus.
    save_path: str
        Directory in which to save outputs.
    custom_annot: pd.DataFrame
        pandas DataFrame with genome annotation for custom species (i.e. for a species other than homo_sapiens, mus_musculus, drosophila_melanogaster or gallus_gallus).
        This DataFrame should (minimally) look like the example below, and only contains protein coding genes:
        >>> custom_annot
                Chromosome      Start  Strand     Gene Transcript_type
            8053         chrY   22490397       1      PRY  protein_coding
            8153         chrY   12662368       1    USP9Y  protein_coding
            8155         chrY   12701231       1    USP9Y  protein_coding
            8158         chrY   12847045       1    USP9Y  protein_coding
            8328         chrY   22096007      -1     PRY2  protein_coding
            ...           ...        ...     ...      ...             ...
            246958       chr1  181483738       1  CACNA1E  protein_coding
            246960       chr1  181732466       1  CACNA1E  protein_coding
            246962       chr1  181776101       1  CACNA1E  protein_coding
            246963       chr1  181793668       1  CACNA1E  protein_coding
            246965       chr1  203305519       1     BTG2  protein_coding

            [78812 rows x 5 columns]

    save_partial: bool=False
        Whether to save the individual analyses as pkl. Useful to run analyses in chunks or add new settings.
    ctx_db_path: str = None
        Path to cistarget database containing rankings of motif scores
    dem_db_path: str = None
        Path to dem database containing motif scores
    run_without_promoters: bool = False
        Boolean specifying wether the analysis should also be run without including promoter regions.
    biomart_host: str = 'http://www.ensembl.org'
        url to biomart host, make sure this host matches your assembly
    promoter_space: int = 500
        integer defining space around the TSS to consider as promoter
    ctx_auc_threshold: float = 0.005
          The fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve
    ctx_nes_threshold: float = 3.0
        The Normalized Enrichment Score (NES) threshold to select enriched features.
    ctx_rank_threshold: float = 0.05
        The total number of ranked genes to take into account when creating a recovery curve.
    dem_log2fc_thr: float = 0.5
        Log2 Fold-change threshold to consider a motif enriched.
    dem_motif_hit_thr: float = 3.0
        Minimul mean signal in the foreground to consider a motif enriched.
    dem_max_bg_regions: int = 500
        Maximum number of regions to use as background. When set to None, all regions are used
    annotation : List[str] = ['Direct_annot', 'Orthology_annot']
        Annotation to use for forming cistromes. It can be 'Direct_annot' (direct evidence that the motif is 
        linked to that TF), 'Motif_similarity_annot' (based on tomtom motif similarity), 'Orthology_annot'
        (based on orthology with a TF that is directly linked to that motif) or 'Motif_similarity_and_Orthology_annot'.
    path_to_motif_annotations: str = None
        Path to motif annotations. If not provided, they will be downloaded from 
        https://resources.aertslab.org based on the specie name provided (only possible for mus_musculus,
        homo_sapiens and drosophila_melanogaster).
    annotation_version: str = 'v9'
         Motif collection version.
    n_cpu : int = 1
        Number of cores to use.
    _temp_dir: str = None
        temp_dir to use for ray.
    exclude_motifs: str = None
        Path to csv file containing motif to exclude from the analysis.
    exclude_collection: List[str] = None
        List of strings identifying which motif collections to exclude from analysis.
    """
    
    # Create logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('pycisTarget_wrapper')
    
    start_time = time.time()
    
    check_folder = os.path.isdir(save_path)
    if not check_folder:
        os.makedirs(save_path)
        log.info("Created folder : " + save_path)
    else:
        log.info(save_path + " folder already exists.")
        
    def get_species_annotation(species: str):
        dataset = pbm.Dataset(name=species,  host=biomart_host)
        annot = dataset.query(attributes=['chromosome_name', 'transcription_start_site', 'strand', 'external_gene_name', 'transcript_biotype'])
        annot.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'Transcript_type']
        annot['Chromosome'] = annot['Chromosome'].astype('str')
        filterf = annot['Chromosome'].str.contains('CHR|GL|JH|MT|KI')
        annot = annot[~filterf]
        annot['Chromosome'] = annot['Chromosome'].replace(r'(\b\S)', r'chr\1')
        annot = annot[annot.Transcript_type == 'protein_coding']
        annot = annot.dropna(subset = ['Chromosome', 'Start'])
        # Check if chromosomes have chr
        check = region_sets[list(region_sets.keys())[0]]
        if not any(['chr' in c for c in check[list(check.keys())[0]].df['Chromosome']]):
            annot.Chromosome = annot.Chromosome.str.replace('chr', '')
        if not any(['chr' in x for x in annot.Chromosome]):
            annot.Chromosome = [f'chr{x}' for x in annot.Chromosome]
        annot_dem=annot.copy()
        # Define promoter space
        annot['End'] = annot['Start'].astype(int)+promoter_space
        annot['Start'] = annot['Start'].astype(int)-promoter_space
        annot = pr.PyRanges(annot[['Chromosome', 'Start', 'End']])
        return annot, annot_dem
        
    # Prepare annotation
    if species == 'homo_sapiens':
        annot, annot_dem = get_species_annotation('hsapiens_gene_ensembl')
    elif species == 'mus_musculus':
        annot, annot_dem = get_species_annotation('mmusculus_gene_ensembl')
    elif species == 'drosophila_melanogaster':
        annot, annot_dem = get_species_annotation('dmelanogaster_gene_ensembl')
    elif species == 'gallus_gallus':
        annot, annot_dem = get_species_annotation('ggallus_gene_ensembl')
    elif species == 'custom':
        annot_dem = custom_annot
        annot = annot_dem.copy()
        # Define promoter space
        annot['End'] = annot['Start'].astype(int)+promoter_space
        annot['Start'] = annot['Start'].astype(int)-promoter_space
        annot = pr.PyRanges(annot[['Chromosome', 'Start', 'End']])
    else:
        raise TypeError("Species not recognized")

    menr = {}
    for key in region_sets.keys():
        if ctx_db_path is not None:
            log.info('Loading cisTarget database for ' + key)
            ## CISTARGET
            regions = region_sets[key]
            ctx_db = cisTargetDatabase(ctx_db_path, regions)  
            if exclude_motifs is not None:
                out = pd.read_csv(exclude_motifs, header=None).iloc[:,0].tolist()
                ctx_db.db_rankings = ctx_db.db_rankings.drop(out)
            if exclude_collection is not None:
                for col in exclude_collection:
                    ctx_db.db_rankings = ctx_db.db_rankings[~ctx_db.db_rankings.index.str.contains(col)]
            ## DEFAULT
            log.info('Running cisTarget for '+key)
            menr['CTX_'+key+'_All'] = run_cistarget(ctx_db = ctx_db,
                                   region_sets = regions,
                                   specie = species,
                                   auc_threshold = ctx_auc_threshold,
                                   nes_threshold = ctx_nes_threshold,
                                   rank_threshold = ctx_rank_threshold,
                                   annotation = annotation,
                                   motif_similarity_fdr = motif_similarity_fdr,
                                   path_to_motif_annotations = path_to_motif_annotations,
                                   n_cpu = n_cpu,
                                   _temp_dir= _temp_dir,
                                   annotation_version = annotation_version,
                                   **kwargs)
            out_folder = os.path.join(save_path,'CTX_'+key+'_All')
            check_folder = os.path.isdir(out_folder)
            if not check_folder:
                os.makedirs(out_folder)
                log.info("Created folder : " + out_folder)
            else:
                log.info(out_folder + " folder already exists.")
            for x in menr['CTX_'+key+'_All'].keys():
                out_file = os.path.join(out_folder, str(x) +'.html')
                menr['CTX_'+key+'_All'][str(x)].motif_enrichment.to_html(open(out_file, 'w'), escape=False, col_space=80)
            if(save_partial):
                with open(os.path.join(save_path,'CTX_'+key+'_All' + '.pkl'), 'wb') as f:
                    dill.dump(menr['CTX_'+key+'_All'], f, protocol=4)

            if run_without_promoters is True:
                ## REMOVE PROMOTERS
                log.info('Running cisTarget without promoters for '+key)
                regions_overlaps = {key: regions[key].count_overlaps(annot) for key in regions.keys()}
                regions_np = {key: regions_overlaps[key][regions_overlaps[key].NumberOverlaps == 0][['Chromosome', 'Start', 'End']] for key in regions.keys()}
                db_regions = set(pd.concat([ctx_db.regions_to_db[x] for x in ctx_db.regions_to_db.keys()])['Query'])
                ctx_db.regions_to_db = {x: target_to_query(regions_np[x], list(db_regions), fraction_overlap = 0.4) for x in regions_np.keys()}
                menr['CTX_'+key+'_No_promoters'] = run_cistarget(ctx_db = ctx_db,
                                   region_sets = regions_np,
                                   specie = species,
                                   auc_threshold = ctx_auc_threshold,
                                   nes_threshold = ctx_nes_threshold,
                                   rank_threshold = ctx_rank_threshold,
                                   annotation = annotation,
                                   motif_similarity_fdr = motif_similarity_fdr, 
                                   path_to_motif_annotations = path_to_motif_annotations,
                                   n_cpu = n_cpu,
                                   _temp_dir= _temp_dir,
                                   annotation_version = annotation_version,
                                   **kwargs)
                out_folder = os.path.join(save_path,'CTX_'+key+'_No_promoters')
                check_folder = os.path.isdir(out_folder)
                if not check_folder:
                    os.makedirs(out_folder)
                    log.info("Created folder:" + out_folder)
                else:
                    log.info(out_folder + " folder already exists.")
                for x in menr['CTX_'+key+'_No_promoters'].keys():
                    out_file = os.path.join(out_folder, str(x) +'.html')
                    menr['CTX_'+key+'_No_promoters'][str(x)].motif_enrichment.to_html(open(out_file, 'w'), escape=False, col_space=80)
                
                if(save_partial):
                    with open(os.path.join(save_path,'CTX_'+key+'_No_promoters' + '.pkl'), 'wb') as f:
                      dill.dump(menr['CTX_'+key+'_No_promoters'], f, protocol=4)
        ## DEM
        if dem_db_path is not None:
            log.info('Running DEM for '+key)
            regions = region_sets[key]
            dem_db = DEMDatabase(dem_db_path, regions)  
            if exclude_motifs is not None:
                out = pd.read_csv(exclude_motifs, header=None).iloc[:,0].tolist()
                dem_db.db_scores = dem_db.db_scores.drop(out)
            if exclude_collection is not None:
                for col in exclude_collection:
                    dem_db.db_scores = dem_db.db_scores[~dem_db.db_scores.index.str.contains(col)]
            menr['DEM_'+key+'_All'] = DEM(dem_db = dem_db,
                               region_sets = regions,
                               log2fc_thr = dem_log2fc_thr,
                               motif_hit_thr = dem_motif_hit_thr,
                               max_bg_regions = dem_max_bg_regions,
                               specie = species,
                               genome_annotation = annot_dem,
                               promoter_space = promoter_space,
                               motif_annotation =   annotation,
                               motif_similarity_fdr = motif_similarity_fdr, 
                               path_to_motif_annotations = path_to_motif_annotations,
                               n_cpu = n_cpu,
                               annotation_version = annotation_version,
                               tmp_dir = save_path,
                               _temp_dir= _temp_dir,
                               **kwargs)
            out_folder = os.path.join(save_path,'DEM_'+key+'_All')
            check_folder = os.path.isdir(out_folder)
            if not check_folder:
                os.makedirs(out_folder)
                log.info("Created folder : "+ out_folder)
            else:
                log.info(out_folder + " folder already exists.")
            for x in menr['DEM_'+key+'_All'].motif_enrichment.keys():
                out_file = os.path.join(out_folder, str(x) +'.html')
                menr['DEM_'+key+'_All'].motif_enrichment[str(x)].to_html(open(out_file, 'w'), escape=False, col_space=80)
            if(save_partial):
                with open(os.path.join(save_path, 'DEM_'+key+'_All'+'.pkl'), 'wb') as f:
                  dill.dump(menr['DEM_'+key+'_All'], f, protocol=4)
                
            if run_without_promoters is True:
                log.info('Running DEM without promoters for '+key)
                ## REMOVE PROMOTERS
                regions_overlaps = {key: regions[key].count_overlaps(annot) for key in regions.keys()}
                regions_np = {key: regions_overlaps[key][regions_overlaps[key].NumberOverlaps == 0][['Chromosome', 'Start', 'End']] for key in regions.keys()}
                db_regions = set(pd.concat([dem_db.regions_to_db[x] for x in dem_db.regions_to_db.keys()])['Query'])
                dem_db.regions_to_db = {x: target_to_query(regions_np[x], list(db_regions), fraction_overlap = 0.4) for x in regions_np.keys()}
                menr['DEM_'+key+'_No_promoters'] = DEM(dem_db = dem_db,
                               region_sets = regions_np,
                               log2fc_thr = dem_log2fc_thr,
                               motif_hit_thr = dem_motif_hit_thr,
                               max_bg_regions = dem_max_bg_regions,
                               specie = species,
                               promoter_space = promoter_space,
                               motif_annotation = annotation,
                               motif_similarity_fdr = motif_similarity_fdr, 
                               path_to_motif_annotations = path_to_motif_annotations,
                               n_cpu = n_cpu,
                               annotation_version = annotation_version,
                               tmp_dir = save_path,
                               _temp_dir= _temp_dir,
                               **kwargs)
                out_folder = os.path.join(save_path,'DEM_'+key+'_No_promoters')
                check_folder = os.path.isdir(out_folder)
                if not check_folder:
                    os.makedirs(out_folder)
                    log.info("Created folder : "+ out_folder)
                else:
                    log.info(out_folder + " folder already exists.")
                for x in menr['DEM_'+key+'_No_promoters'].motif_enrichment.keys():
                    out_file = os.path.join(out_folder, str(x) +'.html')
                    menr['DEM_'+key+'_No_promoters'].motif_enrichment[str(x)].to_html(open(out_file, 'w'), escape=False, col_space=80)
                if(save_partial):
                    with open(os.path.join(save_path, 'DEM_'+key+'_No_promoters'+'.pkl'), 'wb') as f:
                      dill.dump(menr['DEM_'+key+'_All'], f, protocol=4)
                    
    log.info('Saving object')         
    with open(os.path.join(save_path,'menr.pkl'), 'wb') as f:
        dill.dump(menr, f, protocol=4)

    log.info('Finished! Took {} minutes'.format((time.time() - start_time)/60))  
