"""Write SCENIC+ object to hdf5 file.

!depricated!
"""

import h5py
import pandas as pd
import pyranges as pr

import logging
import sys

from copy import deepcopy

from scipy.sparse import csr_matrix, issparse

from anndata._io.h5ad import write_attribute, read_attribute

from pycistarget.hdf5 import write_hdf5 as pycistarget_write_hdf5
from pycistarget.hdf5 import read_h5ad as pycistarget_read_hdf5

from .scenicplus_class import SCENICPLUS
from .grn_builder.modules import eRegulon, REGIONS2GENES_HEADER
from .utils import annotate_eregulon_by_influence

from itertools import chain

EREGULON_ATTRIBUTES = [
    'transcription_factor',
    'cistrome_name',
    'is_extended',
    'context',
    'gsea_enrichment_score',
    'gsea_pval',
    'gsea_adj_pval'
]

def _write_eRegulon_hdf5(eRegulon: eRegulon, hdf5_grp: h5py.Group):

    for attribute_name in EREGULON_ATTRIBUTES:
        if hasattr(eRegulon, attribute_name):
            attribute = getattr(eRegulon, attribute_name)
            if attribute is not None:
                if type(attribute) == frozenset:
                    attribute = list(attribute)
                hdf5_grp.attrs[attribute_name] = attribute
    
    if hasattr(eRegulon, 'regions2genes'):
        attribute = getattr(eRegulon, 'regions2genes')
        if attribute is not None:
            write_attribute(hdf5_grp, 'regions2genes', pd.DataFrame(attribute))
    
    if hasattr(eRegulon, 'in_leading_edge'):
        attribute = getattr(eRegulon, 'in_leading_edge')
        if attribute is not None:
            write_attribute(hdf5_grp, 'in_leading_edge', attribute)

#helper functions for nested dicts
def _flatten_dict(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _flatten_dict(v)
        else:
            yield v

def _traverse_dict_and_set_pyranges_to_pandas(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            _traverse_dict_and_set_pyranges_to_pandas(d[key])
        else:
            d[key] = d[key].df

def _travserse_dict_and_set_pandas_to_pyranges(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            _travserse_dict_and_set_pandas_to_pyranges(d[key])
        else:
            d[key] = pr.PyRanges(d[key])

def _traverse_dict_and_set_series_to_dataframe(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            _traverse_dict_and_set_series_to_dataframe(d[key])
        else:
            d[key] = pd.DataFrame(d[key], columns = ['a'])

def _traverse_dict_and_set_dataframe_to_series(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            _traverse_dict_and_set_dataframe_to_series(d[key])
        else:
            d[key] = pd.Series(d[key]['a'])
            d[key].name = None

#helper functions for nested lists
def _flatten_list(l):
    for v in l:
        if isinstance(v, list):
            yield from _flatten_list(v)
        else:
            yield v

def _traverse_list_and_set_pyranges_to_pandas(l):
    for v in l:
        if isinstance(v, list):
            _traverse_list_and_set_pyranges_to_pandas(v)
        else:
            v = v.df

def _traverse_list_and_set_series_to_dataframe(l):
    for v in l:
        if isinstance(v, list):
            _traverse_list_and_set_series_to_dataframe(v)
        else:
            v = pd.DataFrame(v, columns = ['a'])

def _traverse_list_and_set_pandas_to_pyranges(l):
    for v in l:
        if isinstance(v, list):
            _traverse_list_and_set_pandas_to_pyranges(v)
        else:
            v = pr.PyRanges(v)

def _traverse_list_and_set_dataframe_to_series(l):
    for v in l:
        if isinstance(v, list):
            _traverse_list_and_set_dataframe_to_series(v)
        else:
            v = pd.Series(v['a'])
            v.name = None



def write_hdf5(
    scplus_obj: SCENICPLUS,
    f_name: str,
    force_sparse: bool = True,
    verbose = False):
    """
    Write SCENIC+ object to hdf5

    Parameters
    ----------
    scplus_obj: SCENICPLUS
        An instance of scenicplus_class.SCENICPLUS
    fname: str
        Path to out file.
    force_sparse: bool = True
        Wether or not to force sparsity of expression and chromatin accessibility matrix.
    verbose = False
        Wether or not to print debug messages.
    """


    #TODO: add version!!

    if verbose:
        level = logging.INFO
        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=format, handlers=handlers)
        log = logging.getLogger('hdf5')

    hdf5_file = h5py.File(f_name, 'w')
    try:
        #save expression and accessibility matrices
        if force_sparse and not issparse(scplus_obj.X_EXP):
            if verbose:
                log.info('Making .X_EXP sparse')
            X_EXP = csr_matrix(scplus_obj.X_EXP)
        else:
            X_EXP = scplus_obj.X_EXP
        if verbose:
            log.info('Writing .X_EXP')
        write_attribute(hdf5_file, 'X_EXP', X_EXP, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 6})

        if force_sparse and not issparse(scplus_obj.X_ACC):
            if verbose:
                log.info('Making .X_ACC sparse')
            X_ACC = csr_matrix(scplus_obj.X_ACC)
        else:
            X_ACC = scplus_obj.X_ACC
        if verbose:
            log.info('Writing .X_ACC')
        write_attribute(hdf5_file, 'X_ACC', X_ACC, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 6})

        #write metadata
        if verbose:
            log.info('Writing .metadata_regions')
        write_attribute(hdf5_file, 'metadata_regions', scplus_obj.metadata_regions, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
        if verbose:
            log.info('Writing .metadata_genes')
        write_attribute(hdf5_file, 'metadata_genes', scplus_obj.metadata_genes, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
        if verbose:
            log.info('Writing .metadata_cell')
        write_attribute(hdf5_file, 'metadata_cell', scplus_obj.metadata_cell, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})

        #write motif enrichment results
        grp_menr = hdf5_file.create_group(name = 'menr')
        for menr_name in scplus_obj.menr.keys():
            grp_sub_menr = grp_menr.create_group(name = menr_name)
            if verbose:
                log.info(f'Writing .menr: {menr_name}')
            pycistarget_write_hdf5(result = scplus_obj.menr[menr_name], f_name_or_grp = grp_sub_menr)
        
        #save dimensionality reductions
        if hasattr(scplus_obj, 'dr_cell'):
            if verbose:
                log.info('Writing .dr_cell')
            write_attribute(hdf5_file, 'dr_cell', scplus_obj.dr_cell, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
        
        if hasattr(scplus_obj, 'dr_region'):
            if verbose:
                log.info('Writing .dr_region:')
            write_attribute(hdf5_file, 'dr_region', scplus_obj.dr_region, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})

        #save uns
        if hasattr(scplus_obj, 'uns'):
            grp_uns = hdf5_file.create_group(name = 'uns')
            for key in scplus_obj.uns.keys():
                if type(scplus_obj.uns[key]) == pd.DataFrame:
                    if verbose:
                        log.info(f'Writing .uns[{key}]')
                    write_attribute(grp_uns, key, scplus_obj.uns[key],  dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                    grp_uns[key].attrs['orig-type'] = 'pd.DataFrame'

                if type(scplus_obj.uns[key]) == dict:
                    dict_iter = _flatten_dict(scplus_obj.uns[key])
                    type_dict = set([type(x) for x in dict_iter])
                    if len(type_dict) > 1:
                        raise ValueError(f"dict of multiple types {', '.join([str(x) for x in type_dict])} for {key}")
                    else:
                        type_dict = list(type_dict)[0]
                        if type_dict == pr.PyRanges:
                            attribute = deepcopy(scplus_obj.uns[key])
                            if verbose:
                                log.info(f'Setting .uns[{key}] to pandas dataframe')
                            _traverse_dict_and_set_pyranges_to_pandas(attribute)
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, attribute, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = 'pr.PyRanges'

                        elif type_dict == eRegulon:
                            grp_eRegulons = grp_uns.create_group(name = key)
                            for eRegulon_key in scplus_obj.uns[key].keys():
                                grp_eRegulon = grp_eRegulons.create_group(name = eRegulon_key)
                                if verbose:
                                    log.info(f'Writing .uns[{key}][{eRegulon_key}]')
                                _write_eRegulon_hdf5(scplus_obj.uns[key][eRegulon_key], grp_eRegulon)
                                grp_eRegulon.attrs['orig-type'] = 'eRegulon'
                            grp_eRegulons.attrs['orig-type'] = 'eRegulon-dict'
                        
                        elif type_dict == pd.Series:
                            attribute = deepcopy(scplus_obj.uns[key])
                            if verbose:
                                log.info(f'Setting .uns[{key}] to pandas dataframe')
                            _traverse_dict_and_set_series_to_dataframe(attribute)
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, attribute, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = 'pd.Series'

                        else:
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, scplus_obj.uns[key], dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = str(type(scplus_obj.uns[key]))
                       
                if type(scplus_obj.uns[key]) == list:
                    list_iter = _flatten_list(scplus_obj.uns[key])
                    type_list = set([type(x) for x in list_iter])
                    if len(type_list) > 1:
                        raise ValueError(f"list of multiple types {', '.join([str(x) for x in type_list])} for {key}")
                    else:
                        type_list = list(type_list)[0]
                        
                        if type_list == pr.PyRanges:
                            attribute = deepcopy(scplus_obj.uns[key])
                            if verbose:
                                log.info(f'Setting .uns[{key}] to pandas dataframe')
                            _traverse_list_and_set_pyranges_to_pandas(attribute)
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, attribute, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = 'pr.PyRanges'
                        
                        elif type_list == eRegulon:
                            grp_eRegulons = grp_uns.create_group(name = key)
                            for eregulon in scplus_obj.uns[key]:
                                name = annotate_eregulon_by_influence(eregulon)
                                grp_eRegulon = grp_eRegulons.create_group(name = name)
                                if verbose:
                                    log.info(f'Writing .uns[{key}], Regulon: {name}')
                                _write_eRegulon_hdf5(eregulon, grp_eRegulon)
                                grp_eRegulon.attrs['orig-type'] = 'eRegulon'
                            grp_eRegulons.attrs['orig-type'] = 'eRegulon-list'
                        
                        elif type_list == pd.Series:
                            attribute = deepcopy(scplus_obj.uns[key])
                            if verbose:
                                log.info(f'Setting .uns[{key}] to pandas dataframe')
                            _traverse_list_and_set_series_to_dataframe(attribute)
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, attribute, dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = 'pd.Series'

                        else:
                            if verbose:
                                log.info(f'Writing .uns[{key}]')
                            write_attribute(grp_uns, key, scplus_obj.uns[key], dataset_kwargs = {'compression': 'gzip', 'compression_opts': 9})
                            grp_uns[key].attrs['orig-type'] = str(type(scplus_obj.uns[key]))

    except Exception as e:
        hdf5_file.close()
        raise(e)
    finally:
        hdf5_file.close()

def _read_eRegulon(hdf5_grp: h5py.Group) -> eRegulon:

    attributes_dict = dict(hdf5_grp.attrs)
    regions2genes = read_attribute(hdf5_grp['regions2genes'])

    eRegulon_obj = eRegulon(
        transcription_factor    = attributes_dict['transcription_factor']   if 'transcription_factor'   in attributes_dict.keys() else None,
        cistrome_name           = attributes_dict['cistrome_name']          if 'cistrome_name'          in attributes_dict.keys() else None,
        is_extended             = attributes_dict['is_extended']            if 'is_extended'            in attributes_dict.keys() else None,
        regions2genes           = list(regions2genes[list(REGIONS2GENES_HEADER)].itertuples(index=False, name='r2g'))
    )

    if 'context' in attributes_dict.keys():
        context = frozenset(attributes_dict['context'])
        setattr(eRegulon_obj, 'context', context)
    
    for attribute_name in set(EREGULON_ATTRIBUTES) - set(['transcription_factor', 'cistrome_name', 'is_extended', 'context']):
        if attribute_name in attributes_dict.keys():
            setattr(eRegulon_obj, attribute_name,  attributes_dict[attribute_name])
        elif attribute_name in hdf5_grp.keys():
            setattr(eRegulon_obj, attribute_name,  read_attribute(hdf5_grp[attribute_name]))
    
    return eRegulon_obj
    

def read_h5ad(f_name: str) -> SCENICPLUS:
    """
    Read SCENIC+ object from hdf5 file.

    fname:str
        Path to hdf5 file.
    """
    hdf5_file = h5py.File(f_name, 'r')
    try:
        X_ACC = read_attribute(hdf5_file['X_ACC'])
        X_EXP = read_attribute(hdf5_file['X_EXP'])
        metadata_regions = read_attribute(hdf5_file['metadata_regions'])
        metadata_genes = read_attribute(hdf5_file['metadata_genes'])
        metadata_cell = read_attribute(hdf5_file['metadata_cell'])
        
        grp_menr = hdf5_file['menr']
        menr = {}
        for menr_name in grp_menr.keys():
            menr[menr_name] = pycistarget_read_hdf5(grp_menr[menr_name])

        if 'dr_cell' in hdf5_file.keys():
            dr_cell = read_attribute(hdf5_file['dr_cell'])
        else:
            dr_cell = None

        if 'dr_region' in hdf5_file.keys():
            dr_region = read_attribute(hdf5_file['dr_region'])
        else:
            dr_region = None

        if 'uns' in hdf5_file.keys():
            uns = {}
            for key in hdf5_file['uns'].keys():
                attribute = hdf5_file['uns'][key]
                orig_type = attribute.attrs['orig-type']
                if orig_type == 'eRegulon-list':
                    uns[key] = [_read_eRegulon(attribute[eRegulon_name]) for eRegulon_name in attribute.keys()]
                elif orig_type == 'eRegulon-dict':
                    uns[key] = {eRegulon_name: _read_eRegulon(attribute[eRegulon_name]) for eRegulon_name in attribute.keys()}
                else:
                    sub_uns = read_attribute(attribute)
                    if orig_type == 'pr.PyRanges':
                        if type(sub_uns) == dict:
                            _travserse_dict_and_set_pandas_to_pyranges(sub_uns)
                            uns[key] = sub_uns
                        elif type(sub_uns) == list:
                            _traverse_list_and_set_pandas_to_pyranges(sub_uns)
                            uns[key] = sub_uns
                    elif orig_type == 'pd.Series':
                        if type(sub_uns) == dict:
                            _traverse_dict_and_set_dataframe_to_series(sub_uns)
                            uns[key] = sub_uns
                        elif type(sub_uns) == list:
                            _traverse_list_and_set_dataframe_to_series(sub_uns)
                            uns[key] = sub_uns
                    else:
                        uns[key] = sub_uns
        else:
            uns = None

    except Exception as e:
        hdf5_file.close()
        raise(e)
    finally:
        hdf5_file.close()
    
    SCENICPLUS_obj = SCENICPLUS(
        X_ACC = X_ACC,
        X_EXP = X_EXP,
        metadata_regions = metadata_regions,
        metadata_genes = metadata_genes,
        metadata_cell = metadata_cell,
        menr = menr
    )

    if dr_cell is not None:
        setattr(SCENICPLUS_obj, 'dr_cell', dr_cell)
    
    if dr_region is not None:
        setattr(SCENICPLUS_obj, 'dr_region', dr_region)
    
    if uns is not None:
        setattr(SCENICPLUS_obj, 'uns', uns)
    
    return SCENICPLUS_obj