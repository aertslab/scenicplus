import mudata
from mudata import MuData
import pandas as pd
from typing import Optional

class ScenicPlusMuData(MuData):
    def __init__(
            self,
            acc_gex_mdata: MuData,
            e_regulon_auc_direct: Optional[MuData] = None,
            e_regulon_auc_extended: Optional[MuData] = None,
            e_regulon_metadata_direct: Optional[pd.DataFrame] = None,
            e_regulon_metadata_extended: Optional[pd.DataFrame] = None):
        # Type checking
        # AUC values (direct / extended) always have to be paired with metadata
        if (e_regulon_auc_direct is None and e_regulon_metadata_direct is not None) or \
           (e_regulon_auc_direct is not None and e_regulon_metadata_direct is None):
            raise ValueError(
                "Both AUC values and eRegulon metadata have to be provided for direct eRegulons!")
        if (e_regulon_auc_extended is None and e_regulon_metadata_extended is not None) or \
           (e_regulon_auc_extended is not None and e_regulon_metadata_extended is None):
            raise ValueError(
                "Both AUC values and eRegulon metadata have to be provided for extended eRegulons!")
        # both auc values can not be None
        if (e_regulon_auc_direct is None and e_regulon_auc_extended is None):
            raise ValueError(
                "Some AUC values have to be provided")
        
        # Set properties
        self.has_direct_e_regulons = e_regulon_auc_direct is not None
        self.has_extended_e_regulons = e_regulon_auc_extended is not None

        # Generate constructor
        _constructor = {}
        _constructor["scRNA_counts"] = acc_gex_mdata["scRNA"]
        _constructor["scATAC_counts"] = acc_gex_mdata["scATAC"]
        if e_regulon_auc_direct is not None:
            _constructor["direct_gene_based_AUC"] = e_regulon_auc_direct["Gene_based"]
            _constructor["direct_region_based_AUC"] = e_regulon_auc_direct["Region_based"]
           
        if e_regulon_auc_extended is not None:
            _constructor["extended_gene_based_AUC"] = e_regulon_auc_extended["Gene_based"]
            _constructor["extended_region_based_AUC"] = e_regulon_auc_extended["Region_based"]

        # construct MuData
        super().__init__(_constructor)

        # Add eRegulon metadata to uns
        if self.has_direct_e_regulons:
             self.uns["direct_e_regulon_metadata"] = e_regulon_metadata_direct
        if self.has_extended_e_regulons:
            self.uns["extended_e_regulon_metadata"] = e_regulon_metadata_extended

def read(path: str):
    mdata = mudata.read(path)
    acc_gex_mdata = MuData(
        {"scRNA": mdata["scRNA_counts"], "scATAC": mdata["scATAC_counts"]})
    if "direct_gene_based_AUC" in mdata.mod.keys():
        e_regulon_auc_direct = MuData(
            {"Gene_based": mdata["direct_gene_based_AUC"], "Region_based": mdata["direct_region_based_AUC"]})
        e_regulon_metadata_direct = mdata.uns["direct_e_regulon_metadata"]
    else:
        e_regulon_auc_direct = None
        e_regulon_metadata_direct = None
    if "extended_gene_based_AUC" in mdata.mod.keys():
        e_regulon_auc_extended = MuData(
            {"Gene_based": mdata["extended_gene_based_AUC"], "Region_based": mdata["extended_region_based_AUC"]})
        e_regulon_metadata_extended = mdata.uns["extended_e_regulon_metadata"]
    else:
        e_regulon_auc_extended = None
        e_regulon_metadata_extended = None
    scplusmdata = ScenicPlusMuData(
        acc_gex_mdata=acc_gex_mdata,
        e_regulon_auc_direct=e_regulon_auc_direct,
        e_regulon_auc_extended=e_regulon_auc_extended,
        e_regulon_metadata_direct=e_regulon_metadata_direct,
        e_regulon_metadata_extended=e_regulon_metadata_extended)
    scplusmdata.uns = mdata.uns
    return scplusmdata