

ABSTAIN = -1
# jnlpba
PROTEIN = 0
CELL_LINE = 1
CELL = 2
DNA = 3
RNA = 4


@labeling_function()
def lf_protein_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['protein.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return PROTEIN
    return ABSTAIN


@labeling_function()
def lf_cell_line_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cell_line.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL_LINE
    return ABSTAIN


@labeling_function()
def lf_cell_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['cell.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return CELL
    return ABSTAIN


@labeling_function()
def lf_dna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['dna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return DNA
    return ABSTAIN


@labeling_function()
def lf_rna_distsv(x):
    # Returns a label of rating if pattern of digit star's found in the phrase
    ent = x.lower()
    for phrase in dist_dict['rna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            return RNA
    return ABSTAIN


@labeling_function()
def lf_debug(x):
    ent = x.lower()
    for phrase in dist_dict['rna.dict']:
        # if ed.eval(ent,phrase.lower()) <= max_dist:
        if min_edit_distance(phrase, ent) <= max_dist:
            # l = ed.eval(ent, phrase)
            # if l == 0:
            return RNA
    return ABSTAIN
