import os
import sys

from preprocess.annotation.entity_item import EntityItem

def get_entity_defines(parameters):
    
    entity_items = [
            EntityItem("cytokine", "Cytokine", 0, "cytokine.dict", parameters),
            EntityItem("transcription-factor", "Transcription_factor", 1, "transcription-factor.dict", parameters),
            EntityItem("t-lymphocyte", "T_lymphocyte", 2, "t-lymphocyte.dict", parameters)
            ]

    return entity_items



