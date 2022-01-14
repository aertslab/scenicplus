import pandas as pd
import dill
from pycistarget.motif_enrichment_cistarget import *
from pycistarget.motif_enrichment_dem import *
from pycistarget.utils import *
import pybiomart as pbm
import time

def run_pycistarget(region_sets,
                 species,
                 save_path,
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
                 path_to_motif_annotations: str = None,
                 annotation_version: str = 'v9',
                 n_cpu : int = 1,
                 _temp_dir: str = '/scratch/leuven/313/vsc31305/ray_spill',
                 exclude_motifs: str = None,
                 exclude_collection: List[str] = None):
    
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
        log.info("Created folder : ", save_path)
    else:
        log.info(save_path + " folder already exists.")
        
    # Prepare annotation
    if species == 'homo_sapiens':
        name = 'hsapiens_gene_ensembl'
    elif species == 'mus_musculus':
        name = 'mmusculus_gene_ensembl'
    elif species == 'drosophila_melanogaster':
        name = 'dmelanogaster_gene_ensembl'
    dataset = pbm.Dataset(name=name,  host=biomart_host)
    annot = dataset.query(attributes=['chromosome_name', 'transcription_start_site', 'strand', 'external_gene_name', 'transcript_biotype'])
    annot['Chromosome/scaffold name'] = annot['Chromosome/scaffold name'].astype('str')
    filterf = annot['Chromosome/scaffold name'].str.contains('CHR|GL|JH|MT')
    annot = annot[~filterf]
    annot['Chromosome/scaffold name'] = annot['Chromosome/scaffold name'].str.replace(r'(\b\S)', r'chr\1')
    annot.columns=['Chromosome', 'Start', 'Strand', 'Gene', 'Transcript_type']
    annot = annot[annot.Transcript_type == 'protein_coding']
    annot = annot.dropna(subset = ['Chromosome', 'Start'])
    annot_dem=annot.copy()
    # Define promoter space
    annot['End'] = annot['Start'].astype(int)+promoter_space
    annot['Start'] = annot['Start'].astype(int)-promoter_space
    annot = pr.PyRanges(annot[['Chromosome', 'Start', 'End']])
        

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
                                   path_to_motif_annotations = path_to_motif_annotations,
                                   n_cpu = n_cpu,
                                   _temp_dir= _temp_dir,
                                   annotation_version = annotation_version)
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

            if run_without_promoters is True:
                ## REMOVE PROMOTERS
                log.info('Running cisTarget without promoters for '+key)
                regions_overlaps = {key: regions[key].count_overlaps(annot) for key in regions.keys()}
                regions_np = {key: regions_overlaps[key][regions_overlaps[key].NumberOverlaps == 0][['Chromosome', 'Start', 'End']] for key in regions.keys()}
                db_regions = set(pd.concat([ctx_db.regions_to_db[x] for x in ctx_db.regions_to_db.keys()])['Target'])
                ctx_db.regions_to_db = {x: target_to_query(regions_np[x], list(db_regions), fraction_overlap = 0.4) for x in regions_np.keys()}
                menr['CTX_'+key+'_No_promoters'] = run_cistarget(ctx_db = ctx_db,
                                   region_sets = regions_np,
                                   specie = species,
                                   auc_threshold = ctx_auc_threshold,
                                   nes_threshold = ctx_nes_threshold,
                                   rank_threshold = ctx_rank_threshold,
                                   annotation = annotation,
                                   path_to_motif_annotations = path_to_motif_annotations,
                                   n_cpu = n_cpu,
                                   _temp_dir= _temp_dir,
                                   annotation_version = annotation_version)
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
                               path_to_motif_annotations = path_to_motif_annotations,
                               n_cpu = n_cpu,
                               tmp_dir= _temp_dir,
                               annotation_version = annotation_version)
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
            if run_without_promoters is True:
                log.info('Running DEM without promoters for '+key)
                ## REMOVE PROMOTERS
                regions_overlaps = {key: regions[key].count_overlaps(annot) for key in regions.keys()}
                regions_np = {key: regions_overlaps[key][regions_overlaps[key].NumberOverlaps == 0][['Chromosome', 'Start', 'End']] for key in regions.keys()}
                db_regions = set(pd.concat([dem_db.regions_to_db[x] for x in dem_db.regions_to_db.keys()])['Target'])
                dem_db.regions_to_db = {x: target_to_query(regions_np[x], list(db_regions), fraction_overlap = 0.4) for x in regions_np.keys()}
                menr['DEM_'+key+'_No_promoters'] = DEM(dem_db = dem_db,
                               region_sets = regions_np,
                               log2fc_thr = dem_log2fc_thr,
                               motif_hit_thr = dem_motif_hit_thr,
                               max_bg_regions = dem_max_bg_regions,
                               specie = species,
                               promoter_space = promoter_space,
                               motif_annotation = annotation,
                               path_to_motif_annotations = path_to_motif_annotations,
                               n_cpu = n_cpu,
                               tmp_dir= _temp_dir,
                               annotation_version = annotation_version)
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
                    
    log.info('Saving object')         
    with open(os.path.join(save_path,'menr.pkl'), 'wb') as f:
        dill.dump(menr, f)

    log.info('Finished! Took {} minutes'.format((time.time() - start_time)/60))  