
ABSTAIN = -1
# cancer immunology
CYTOKINE = 0
TRANSCRIPTION_FACTOR = 1
T_LYMPHOCYTE = 2


global dist_dict

# snorkel Labeling functions
@labeling_function()
def lf_cytokine_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cytokine.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CYTOKINE
    return ABSTAIN


@labeling_function()
def lf_cytokine_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cytokine.dict']:
        if is_substr(phrase, ent):
            return CYTOKINE
    return ABSTAIN


@labeling_function()
def lf_transcription_factor_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['transcription_factor.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return TRANSCRIPTION_FACTOR
    return ABSTAIN


@labeling_function()
def lf_transcription_factor_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['transcription_factor.dict']:
        if is_substr(phrase, ent):
            return TRANSCRIPTION_FACTOR
    return ABSTAIN


@labeling_function()
def lf_t_lymphocyte_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['t_lymphocyte.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return T_LYMPHOCYTE
    return ABSTAIN


@labeling_function()
def lf_t_lymphocyte_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['t_lymphocyte.dict']:
        if is_substr(phrase, ent):
            return T_LYMPHOCYTE
    return ABSTAIN

