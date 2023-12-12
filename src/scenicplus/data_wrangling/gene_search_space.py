import pyranges as pr
import logging
from scenicplus.utils import (calculate_distance_join, calculate_distance_with_limits_join, 
                              extend_pyranges,
                              extend_pyranges_with_limits,
                              reduce_pyranges_b, reduce_pyranges_with_limits_b,
                              region_names_to_coordinates)
import pandas as pd
import numpy as np
from typing import Set, Tuple, Union
import pybiomart as pbm
import xml.etree.ElementTree as xml_tree
import re
import requests
import time
import sys

_NCBI_MAX_RETRIES = 3

class MaxNCBIRetriesReached(Exception):
    def __init__(self):
        super().__init__(f"Data not found after {_NCBI_MAX_RETRIES} retries.")

class NCBISearchNotFound(Exception):
    def __init__(self, search_term, url):
        super().__init__(f"Could not find {search_term} on {url}")

def download_gene_annotation_and_chromsizes(
        species: str,
        biomart_host: str,
        use_ucsc_chromosome_style: bool = True
        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Download gene annotation for specific species.
    This function needs access to the internet.

    Parameters
    ----------
    species: str
        species for which to get the gene annotation, e.g. "hsapiens"
    biomart_host: str
        url for the biomart host to use. Please make sure that the host
        used matches with the annotation used for running SCENIC+.

    Returns
    -------
    A pandas DataFrame with following columns:
        Chromosome, Start, End, Strand, Gene, Transcription_Start_Site, Transcript_type
    The dataframe is subsetted for protein coding genes. 
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('Download gene annotation')
    dataset_name = f"{species}_gene_ensembl"
    server = pbm.Server(host=biomart_host, use_cache=False)
    mart = server["ENSEMBL_MART_ENSEMBL"]
    if dataset_name not in mart.list_datasets()['name'].to_numpy():
        raise ValueError(
            f"The dataset name: {dataset_name} could not be found in biomart. "
            + "Check species name or consider manually providing gene annotations!")
    dataset = mart[dataset_name]
    external_gene_name_query = "external_gene_name" \
        if "external_gene_name" in dataset.attributes.keys() else "hgnc_symbol"
    transcription_start_site_query = "transcription_start_site" \
        if "transcription_start_site" in dataset.attributes.keys() else "transcript_start"
    annot = dataset.query(attributes=["chromosome_name", "start_position", "end_position",
                    "strand", external_gene_name_query, transcription_start_site_query, "transcript_biotype"])
    annot.columns = ["Chromosome", "Start", "End", "Strand",
                        "Gene", "Transcription_Start_Site", "Transcript_type"]
    annot = annot[annot.Transcript_type == "protein_coding"]
    annot['Strand'] = ['+' if strand == 1 else '-' for strand in annot['Strand']]
    # get assembly information
    try:
        _regex_display_name = re.search(r'\((.*?)\)',dataset.display_name)
        if _regex_display_name is None:
            raise ValueError("Could not find assembly from biomart query display name.")
        ncbi_search_term = _regex_display_name.group(1)
        log.info(f"Using genome: {ncbi_search_term}")
        # Look for genome id
        ncbi_tries = 0
        ncbi_search_genome_id_response = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=genome&term={ncbi_search_term}")
        while not ncbi_search_genome_id_response.ok and ncbi_tries < _NCBI_MAX_RETRIES:
            time.sleep(0.5) #sleep to not get blocked by NCBI (max 3 requests per second)
            ncbi_search_genome_id_response = requests.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=genome&term={ncbi_search_term}")
            ncbi_tries = ncbi_tries + 1
        if (ncbi_tries == _NCBI_MAX_RETRIES) and not ncbi_search_genome_id_response.ok:
            raise MaxNCBIRetriesReached
        _IdList_element = xml_tree.fromstring(ncbi_search_genome_id_response.content) \
            .find('IdList')
        if _IdList_element is None:
            raise NCBISearchNotFound(
                search_term = "IdList",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=genome&term={ncbi_search_term}")
        _Id_element = _IdList_element.find('Id')
        if _Id_element is None:
            raise NCBISearchNotFound(
                search_term = "Id",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=genome&term={ncbi_search_term}")
        ncbi_genome_id = _Id_element.text
        log.info(f"Found corresponding genome Id {ncbi_genome_id} on NCBI")
        ncbi_tries = 0
        time.sleep(0.5)
        ncbi_search_assembly_id_response = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
        while not ncbi_search_assembly_id_response.ok and ncbi_tries < _NCBI_MAX_RETRIES:
            time.sleep(0.5)
            ncbi_search_assembly_id_response = requests.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
            ncbi_tries = ncbi_tries + 1
        if (ncbi_tries == _NCBI_MAX_RETRIES) and not ncbi_search_genome_id_response.ok:
            raise MaxNCBIRetriesReached
        _DocSum_element = xml_tree.fromstring(ncbi_search_assembly_id_response.content) \
            .find('DocSum')
        if _DocSum_element is None:
            raise NCBISearchNotFound(
                search_term = 'DocSum',
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
        _DocSumItem_elements = _DocSum_element.findall('Item')
        if len(_DocSumItem_elements) == 0:
            raise NCBISearchNotFound(
                search_term = 'Item',
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
        if not all(["Name" in x.attrib for x in _DocSumItem_elements]):
            raise NCBISearchNotFound(
                search_term = "Name",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
        _DocSumItem_dict = {
            x.attrib["Name"]: x.text
            for x in _DocSumItem_elements}
        if "AssemblyID" not in _DocSumItem_dict.keys():
            raise NCBISearchNotFound(
                search_term = "AssemblyID",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=genome&id={ncbi_genome_id}")
        ncbi_assembly_id = _DocSumItem_dict["AssemblyID"]
        log.info(f"Found corresponding assembly Id {ncbi_assembly_id} on NCBI")
        ncbi_tries = 0
        time.sleep(0.5)
        ncbi_search_assembly_report_url = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
        while not ncbi_search_assembly_report_url.ok and ncbi_tries < _NCBI_MAX_RETRIES:
            time.sleep(0.5)
            ncbi_search_assembly_report_url = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
            ncbi_tries = ncbi_tries + 1
        if (ncbi_tries == _NCBI_MAX_RETRIES) and not ncbi_search_genome_id_response.ok:
            raise MaxNCBIRetriesReached
        _DocumentSummarySet_element = xml_tree.fromstring(ncbi_search_assembly_report_url.content) \
            .find('DocumentSummarySet')
        if _DocumentSummarySet_element is None:
            raise NCBISearchNotFound(
                search_term = "DocumentSummarySet",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
        _DocumentSummary_element = _DocumentSummarySet_element.find("DocumentSummary")
        if _DocumentSummary_element is None:
            raise NCBISearchNotFound(
                search_term = "DocumentSummary",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
        _FtpPath_Assembly_rpt_element = _DocumentSummary_element \
            .find("FtpPath_Assembly_rpt")
        if _FtpPath_Assembly_rpt_element is None:
            raise NCBISearchNotFound(
                search_term = "FtpPath_Assembly_rpt",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
        ncbi_assembly_report_url = _FtpPath_Assembly_rpt_element.text
        if ncbi_assembly_report_url is None:
            raise NCBISearchNotFound(
                search_term = "text",
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={ncbi_assembly_id}")
        log.info(f"Downloading assembly information from: {ncbi_assembly_report_url}")
        assembly_report = pd.read_csv(
            ncbi_assembly_report_url,
            comment = '#',
            names = [
            "Sequence-Name", "Sequence-Role", "Assigned-Molecule",
            "Assigned-Molecule-Location/Type", "GenBank-Accn", "Relationship",
            "RefSeq-Accn", "Assembly-Unit", "Sequence-Length", "UCSC-style-name"],
            sep = '\t')
        assembly_report = assembly_report.loc[
            assembly_report['Sequence-Role'] == 'assembled-molecule']
        assembled_molecules = assembly_report['Sequence-Name'].to_list()
        str_assembled_molecules = "\n\t".join(assembled_molecules)
        log.info("Found following assembled molecules (chromosomes): \n\t" + str_assembled_molecules)
    except (MaxNCBIRetriesReached, NCBISearchNotFound) as e:
        print(e)
        print(
             "Returning gene annotation without subestting for assembled chromosomes" \
            +"and converting to UCSC style. Please make sure that the chromosome names" \
            +"in the returned object match with the chromosome names in the scplus_obj." \
            + "Chromosome sizes will not be returned")
        return annot
    except Exception as e:
        print("Unhandeled exception occured")
        print(e)
        print(
             "Returning gene annotation without subestting for assembled chromosomes" \
            +"and converting to UCSC style. Please make sure that the chromosome names" \
            +"in the returned object match with the chromosome names in the scplus_obj." \
            + "Chromosome sizes will not be returned")
        return annot
    else:
        annot = annot.query('Chromosome in @assembled_molecules').copy()
        chromsizes = pd.DataFrame(columns = ["Chromosome", "Start", "End"])
        chromsizes["Start"] = np.repeat(0, len(assembly_report))
        chromsizes["End"] = assembly_report["Sequence-Length"].to_list()
        if use_ucsc_chromosome_style:
            ensembl_to_ucsc_chrom = assembly_report[['Sequence-Name', 'UCSC-style-name']] \
                .set_index('Sequence-Name')['UCSC-style-name'].to_dict()
            str_ensembl_to_ucsc_chrom = "\n\tOriginal\tUCSC"
            for ens in ensembl_to_ucsc_chrom.keys():
                str_ensembl_to_ucsc_chrom += "\n\t" + ens + "\t" + ensembl_to_ucsc_chrom[ens]
            log.info("Converting chromosomes names to UCSC style as follows: " + str_ensembl_to_ucsc_chrom)
            annot['Chromosome'] = [
                ensembl_to_ucsc_chrom[chrom] 
                if chrom in ensembl_to_ucsc_chrom.keys() 
                else chrom for chrom in annot['Chromosome']]
            chromsizes["Chromosome"] = assembly_report["UCSC-style-name"].to_list()
        else:
            chromsizes["Chromosome"] = assembly_report["Sequence-Name"].to_list()
        return annot, chromsizes

def get_search_space(
    scplus_region: Set[str],
    scplus_genes: Set[str],
    gene_annotation: pd.DataFrame,
    chromsizes: pd.DataFrame,
    use_gene_boundaries: bool = False,
    upstream: Tuple[int, int] = (1000, 150000),
    downstream: Tuple[int, int] = (1000, 150000),
    extend_tss: Tuple[int, int] = (10, 10),
    remove_promoters: bool = False) -> pd.DataFrame:
    """
    Get search space surrounding genes to calculate enhancer to gene links

    Parameters
    ----------
    scplus_regions: Set[str]
        Regions to consider in the analysis.
    scplus_genes: Set[str]
        Genes to consider in the analysis.
    gene_annotation: pd.DataFrame
        A Data Frame with transcription starting site annotations for genes
        Must have the following columns:
        "Chromosome", "Start", "end", "Strand", "Gene", "Transcription_Start_Site"
    chromsizes: pd.DataFrame
        A Data Frame with chromosome sizes.
        Must have the following columns:
        "Chromosome", "Start", "end"
        With Start equal to 0 and end equal to the chromosome length.
    use_gene_boundaries: bool, optional
        Whether to use the whole search space or stop when encountering another gene. Default: False
    upstream: List, optional
        Search space upstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: (1000, 100000)
    downstream: List, optional
        Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: (1000, 100000)
    extend_tss: list, optional
        Space around the TSS consider as promoter. Default: (10,10)
    remove_promoters: bool, optional
        Whether to remove promoters from the search space or not. Default: False
   
    Returns
    -------
    pd.DataFrame
        A data frame containing regions in the search space for each gene
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('Get search space')

    #check column names
    _gene_annotation_required_cols = [
        "Chromosome", "Start", "End", "Strand", "Gene", "Transcription_Start_Site"]
    _chromsizes_required_cols = [
        "Chromosome", "Start", "End"]
    if not all([col in gene_annotation.columns for col in _gene_annotation_required_cols]):
        raise ValueError(
            f"gene_annotation should have the following columns: {', '.join(_gene_annotation_required_cols)}")
    if not all([col in chromsizes.columns for col in _chromsizes_required_cols]):
        raise ValueError(
            f"chromsizes should have the following columns: {', '.join(_chromsizes_required_cols)}")
    
    #convert to pyranges
    pr_gene_annotation = pr.PyRanges(gene_annotation.query("Gene in @scplus_genes").copy())
    pr_chromsizes = pr.PyRanges(chromsizes)
    pr_regions = pr.PyRanges(region_names_to_coordinates(scplus_region))

    # Add gene width
    if pr_gene_annotation.df['Gene'].isnull().to_numpy().any():
        pr_gene_annotation = pr.PyRanges(
            pr_gene_annotation.df.fillna(value={'Gene': 'na'}))
    pr_gene_annotation.Gene_width = abs(
        pr_gene_annotation.End - pr_gene_annotation.Start).astype(np.int32)

    # Prepare promoter annotation
    pd_promoters = pr_gene_annotation.df.loc[:, ['Chromosome',
                                    'Transcription_Start_Site', 'Strand', 'Gene']]
    pd_promoters['Transcription_Start_Site'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']
    ).astype(np.int32)
    pd_promoters['End'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']).astype(np.int32)
    pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'End']
    pd_promoters = pd_promoters.loc[:, [
        'Chromosome', 'Start', 'End', 'Strand', 'Gene']]
    pr_promoters = pr.PyRanges(pd_promoters)
    log.info('Extending promoter annotation to {} bp upstream and {} downstream'.format(
        str(extend_tss[0]), str(extend_tss[1])))
    pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

    if use_gene_boundaries:
        log.info('Calculating gene boundaries')
        # add chromosome limits
        chromsizes_begin_pos = pr_chromsizes.df.copy()
        chromsizes_begin_pos['End'] = 1
        chromsizes_begin_pos['Strand'] = '+'
        chromsizes_begin_pos['Gene'] = 'Chrom_Begin'
        chromsizes_begin_neg = chromsizes_begin_pos.copy()
        chromsizes_begin_neg['Strand'] = '-'
        chromsizes_end_pos = pr_chromsizes.df.copy()
        chromsizes_end_pos['Start'] = chromsizes_end_pos['End'] - 1
        chromsizes_end_pos['Strand'] = '+'
        chromsizes_end_pos['Gene'] = 'Chrom_End'
        chromsizes_end_neg = chromsizes_end_pos.copy()
        chromsizes_end_neg['Strand'] = '-'
        gene_bound = pr.PyRanges(
            pd.concat(
                [
                    pr_promoters.df,
                    chromsizes_begin_pos,
                    chromsizes_begin_neg,
                    chromsizes_end_pos,
                    chromsizes_end_neg
                ]
            )
        )

        # Get distance to nearest promoter (of a differrent gene)
        annot_nodup = pr_gene_annotation[['Chromosome',
                             'Start',
                             'End',
                             'Strand',
                             'Gene',
                             'Gene_width']].drop_duplicate_positions().copy()
        annot_nodup = pr.PyRanges(
            annot_nodup.df.drop_duplicates(subset="Gene", keep="first"))
        closest_promoter_upstream = annot_nodup.nearest(
            gene_bound, overlap=False, how='upstream')
        closest_promoter_upstream = closest_promoter_upstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
        closest_promoter_downstream = annot_nodup.nearest(
            gene_bound, overlap=False, how='downstream')
        closest_promoter_downstream = closest_promoter_downstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
        # Add distance information and limit if above/below thresholds
        annot_df = annot_nodup.df
        annot_df = annot_df.set_index('Gene')
        closest_promoter_upstream_df = closest_promoter_upstream.df.set_index(
            'Gene').Distance
        closest_promoter_upstream_df.name = 'Distance_upstream'
        annot_df = pd.concat(
            [annot_df, closest_promoter_upstream_df], axis=1, sort=False)
        closest_promoter_downstream_df = closest_promoter_downstream.df.set_index(
            'Gene').Distance
        closest_promoter_downstream_df.name = 'Distance_downstream'
        annot_df = pd.concat(
            [annot_df, closest_promoter_downstream_df], axis=1, sort=False).reset_index()
        annot_df.loc[annot_df.Distance_upstream <
                        upstream[0], 'Distance_upstream'] = upstream[0]
        annot_df.loc[annot_df.Distance_upstream >
                        upstream[1], 'Distance_upstream'] = upstream[1]
        annot_df.loc[annot_df.Distance_downstream <
                        downstream[0], 'Distance_downstream'] = downstream[0]
        annot_df.loc[annot_df.Distance_downstream >
                        downstream[1], 'Distance_downstream'] = downstream[1]
        annot_nodup = pr.PyRanges(annot_df.dropna(axis=0))
        # Extend to search space
        log.info(
            """Extending search space to: 
            \t\t\t\t\t\tA minimum of {} bp downstream of the end of the gene.
            \t\t\t\t\t\tA minimum of {} bp upstream of the start of the gene.
            \t\t\t\t\t\tA maximum of {} bp downstream of the end of the gene or the promoter of the nearest downstream gene.
            \t\t\t\t\t\tA maximum of {} bp upstream of the start of the gene or the promoter of the nearest upstream gene""".format(str(downstream[0]), str(upstream[0]), str(downstream[1]), str(upstream[1])))
        extended_annot = extend_pyranges_with_limits(annot_nodup)
        extended_annot = extended_annot[['Chromosome',
                                            'Start',
                                            'End',
                                            'Strand',
                                            'Gene',
                                            'Gene_width',
                                            'Distance_upstream',
                                            'Distance_downstream']]
    else:
        log.info(
            """Extending search space to:
            \t\t\t\t\t\t{} bp downstream of the end of the gene.
            \t\t\t\t\t\t{} bp upstream of the start of the gene.""".format(str(downstream[1]), str(upstream[1])))
        annot_nodup = pr_gene_annotation[['Chromosome',
                             'Start',
                             'End',
                             'Strand',
                             'Gene',
                             'Gene_width']].drop_duplicate_positions().copy()
        annot_nodup = pr.PyRanges(
            annot_nodup.df.drop_duplicates(subset="Gene", keep="first"))
        extended_annot = extend_pyranges(annot_nodup, upstream[1], downstream[1])
        extended_annot = extended_annot[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width']]

    # Format search space
    extended_annot = extended_annot.drop_duplicate_positions()

    log.info('Intersecting with regions.')
    regions_per_gene = pr_regions.join(extended_annot, apply_strand_suffix=False)
    regions_per_gene.Name = [
        f"{chrom}:{start}-{end}" 
        for chrom, start, end in regions_per_gene.df[["Chromosome", "Start", "End"]].to_numpy()]
    regions_per_gene.Width = abs(
        regions_per_gene.End - regions_per_gene.Start).astype(np.int32)
    regions_per_gene.Start = round(
        regions_per_gene.Start + regions_per_gene.Width / 2).astype(np.int32)
    regions_per_gene.End = (regions_per_gene.Start + 1).astype(np.int32)
    # Calculate distance
    log.info('Calculating distances from region to gene')
    if use_gene_boundaries:
        regions_per_gene = reduce_pyranges_with_limits_b(regions_per_gene)
        regions_per_gene = calculate_distance_with_limits_join(regions_per_gene)
    else:
        regions_per_gene = reduce_pyranges_b(
            regions_per_gene, upstream[1], downstream[1])
        regions_per_gene = calculate_distance_join(regions_per_gene)

    # Remove DISTAL regions overlapping with promoters
    if remove_promoters:
        log.info('Removing DISTAL regions overlapping promoters')
        regions_per_gene_overlapping_genes = regions_per_gene[regions_per_gene.Distance == 0]
        regions_per_gene_distal = regions_per_gene[regions_per_gene.Distance != 0]
        regions_per_gene_distal_wo_promoters = regions_per_gene_distal.overlap(
            pr_promoters, invert=True)
        regions_per_gene = pr.PyRanges(pd.concat(
            [regions_per_gene_overlapping_genes.df, regions_per_gene_distal_wo_promoters.df]))

    regions_per_gene = pr.PyRanges(regions_per_gene.df.drop_duplicates())

    log.info('Imploding multiple entries per region and gene')
    df = regions_per_gene.df
    default_columns = ['Chromosome', 'Start',
                        'End', 'Name', 'Strand', 'Gene']
    agg_dict_func1 = {column: lambda x: x.tolist()[0]
                        for column in default_columns}
    agg_dict_func2 = {column: lambda x: x.tolist()
                        for column in set(list(df.columns)) - set(default_columns)}
    agg_dict_func = {**agg_dict_func1, **agg_dict_func2}
    df = df.groupby(['Gene', 'Name'], as_index=False).agg(agg_dict_func)
    regions_per_gene = pr.PyRanges(df)
    return regions_per_gene.df[['Name', 'Gene', 'Distance']]