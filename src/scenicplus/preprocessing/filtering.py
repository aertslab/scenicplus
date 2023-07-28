"""Filter outlier genes and regions.

"""

import logging
import sys
from typing import Union
from scenicplus.scenicplus_mudata import ScenicPlusMuData
from mudata import MuData
import pandas as pd

level = logging.INFO
format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
handlers = [logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format, handlers=handlers)
log = logging.getLogger('Preprocessing')

def apply_std_filtering_to_eRegulons(
        scplus_mudata: Union[ScenicPlusMuData, MuData]):
        # Only keep positive R2G
        log.info("Only keeping positive region-to-gene links")
        if "direct_e_regulon_metadata" in scplus_mudata.uns.keys():
                direct_e_regulon_metadata_filtered = scplus_mudata.uns[
                        "direct_e_regulon_metadata"].query("rho_R2G > 0")
        else:
                direct_e_regulon_metadata_filtered = None
        if "extended_e_regulon_metadata" in scplus_mudata.uns.keys():
                extended_e_regulon_metadata_filtered = scplus_mudata.uns[
                        "extended_e_regulon_metadata"].query("rho_R2G > 0") 
        else:
                extended_e_regulon_metadata_filtered = None
        if direct_e_regulon_metadata_filtered is not None and extended_e_regulon_metadata_filtered is not None:
                log.info("Only keep extended inf not direct")
                eRegulons_direct = set(direct_e_regulon_metadata_filtered["eRegulon_name"])
                eRegulons_extended = set(extended_e_regulon_metadata_filtered["eRegulon_name"])
                eRegulons_extended_not_direct = [
                        eRegulon for eRegulon in eRegulons_extended
                        if eRegulon.replace("extended", "direct") not in eRegulons_direct]
                extended_e_regulon_metadata_filtered = extended_e_regulon_metadata_filtered.query(
                        "eRegulon_name in @eRegulons_extended_not_direct")
                e_regulon_metadata_filtered = pd.concat([direct_e_regulon_metadata_filtered, extended_e_regulon_metadata_filtered])
                scplus_mudata.uns["e_regulon_metadata_filtered"] = e_regulon_metadata_filtered
        elif direct_e_regulon_metadata_filtered is not None:
                scplus_mudata.uns["e_regulon_metadata_filtered"] = direct_e_regulon_metadata_filtered
        elif extended_e_regulon_metadata_filtered is not None:
                scplus_mudata.uns["e_regulon_metadata_filtered"] = extended_e_regulon_metadata_filtered