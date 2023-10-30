
ABSTAIN = -1
# BC5CDR
CHEMICAL = 0
DISEASE = 1


@labeling_function()
def lf_chemicals_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['chemicals.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CHEMICAL
    return ABSTAIN


@labeling_function()
def lf_chemicals_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['chemicals.dict']:
        if is_substr(phrase, ent):
            return CHEMICAL
    return ABSTAIN


@labeling_function()
def lf_disease_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['disease.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return DISEASE
    return ABSTAIN


@labeling_function()
def lf_disease_substr(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['disease.dict']:
        if is_substr(phrase, ent):
            return DISEASE
    return ABSTAIN
