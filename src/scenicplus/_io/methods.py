from anndata._io.specs.methods import *
from anndata._io.specs.methods import _REGISTRY
from anndata._io.specs.registry import (
    IOSpec,
    write_elem
)

import pyranges as pr
import scenicplus.grn_builder.modules

def _traverse_dict_and_get_type(d):
    for key in d.keys():
        if isinstance(d[key], dict):
            t = _traverse_dict_and_get_type(d[key])
        else:
            return type(d[key])
    return t

def _traverse_list_and_get_type(l):
    for i in l:
        if isinstance(i, list):
            t = _traverse_list_and_get_type(i)
        else:
            return type(i)
    return t

def get_adata_compatible_uns(scplus_obj):
    for key in scplus_obj.uns.keys():
        if isinstance(scplus_obj.uns[key], dict):
            t = _traverse_dict_and_get_type(scplus_obj.uns[key])
        elif isinstance(scplus_obj.uns[key], list):
            t = _traverse_list_and_get_type(scplus_obj.uns[key])
        else:
            t = type(scplus_obj.uns[key])
        if _REGISTRY.has_writer(H5Group, t, frozenset()):
            yield key
        else:
            print(t)

@_REGISTRY.register_write(H5Group, pr.PyRanges, IOSpec("pyranges", "0.2.0"))
@_REGISTRY.register_write(ZarrGroup, pr.PyRanges, IOSpec("pyranges", "0.2.0"))
def write_pyranges(f, key, pr, dataset_kwargs=MappingProxyType({})):
    write_dataframe(f, key, pr.df, dataset_kwargs)

@_REGISTRY.register_write(H5Group, pd.Series, IOSpec("series", "0.2.0"))
@_REGISTRY.register_write(ZarrGroup, pd.Series, IOSpec("series", "0.2.0"))
def write_series(f, key, s, dataset_kwargs=MappingProxyType({})):
    group = f.create_group(key)
    if s.index.name is not None:
        index_name = s.index.name
    else:
        index_name = "_index"
    group.attrs["_index"] = check_key(index_name)
    write_elem(group, index_name, s.index._values, dataset_kwargs=dataset_kwargs)
    write_elem(group, '_values', s.values, dataset_kwargs=dataset_kwargs)

@_REGISTRY.register_write(H5Group, scenicplus.grn_builder.modules.eRegulon, IOSpec("eregulon", "0.2.0"))
@_REGISTRY.register_write(ZarrGroup, scenicplus.grn_builder.modules.eRegulon, IOSpec("eregulon", "0.2.0"))
def write_eregulons(f, key, eRegulon, dataset_kwargs=MappingProxyType({})):
    group = f.create_group(key)
    for attribute_name in scenicplus.grn_builder.modules.EREGULON_ATTRIBUTES:
        if hasattr(eRegulon, attribute_name):
            attribute = getattr(eRegulon, attribute_name)
            if attribute is not None:
                if type(attribute) == frozenset:
                    attribute = list(attribute)
                write_elem(group, check_key(attribute_name), attribute, dataset_kwargs=dataset_kwargs)

@_REGISTRY.register_write(H5Group, pd.core.arrays.string_.StringArray, IOSpec("string", "0.2.0"))
@_REGISTRY.register_write(ZarrGroup, pd.core.arrays.string_.StringArray, IOSpec("string", "0.2.0"))
def write_categorical(f, k, s, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    write_elem(g, 'values', s.tolist(), dataset_kwargs=dataset_kwargs)

@_REGISTRY.register_write(H5Group, pd.core.arrays.floating.FloatingArray, IOSpec("float", "0.2.0"))
@_REGISTRY.register_write(ZarrGroup, pd.core.arrays.floating.FloatingArray, IOSpec("float", "0.2.0"))
def write_categorical(f, k, s, dataset_kwargs=MappingProxyType({})):
    g = f.create_group(k)
    write_elem(g, 'values', s.astype(np.float32), dataset_kwargs=dataset_kwargs)


