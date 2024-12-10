#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:14:21 2024

@author: hernan

Generate consensus taxonomy for collapsed alignment arrays
"""

import numpy as np

def get_consensus_taxonomy(lineages):
    """
    Given a table containing a set of instance lineages, retrieve the lowest,
    non-conflictive (only one known value) TaxId

    Parameters
    ----------
    lineages : TYPE
        DESCRIPTION.

    Returns
    -------
    consensus_tax : TYPE
        DESCRIPTION.

    """
    # return the consensus taxon at the highest possible level
    consensus_tax = 0
    for lvl in lineages.to_numpy().T:
        lvl_taxa = np.unique(lvl)
        lvl_taxa = lvl_taxa[lvl_taxa != 0]
        if len(lvl_taxa) != 1:
            # none of the sequences have a known tax for this level
            # or
            # multiple taxa found at this level, conflict
            return consensus_tax
        # there is consensus at the current level
        consensus_tax = lvl_taxa[0]
    return consensus_tax

def collapse_taxonomies(branches, taxids, lineages_tab):
    """
    Retrieve the consensus taxonomy for each branch of collapsed sequences.
    Generates an array containign the consensus TaxId for each branch

    Parameters
    ----------
    branches : list
        List of arrays containing the sequence IDs clustered in each branch.
    taxids : numpy.array
        Array containing the TaxIds of the original alignment matrix.
    lineages_tab : pandas.DataFrame
        Lineage dataframe.

    Returns
    -------
    collapsed_taxa : numpy.array
        Array containing the lowest, non conflicting TaxId for each branch
        (shape = # branches).

    """
    
    collapsed_taxa = []
    for branch in branches:
        branch_taxids = taxids[branch]
        branch_lineages = lineages_tab.loc[branch_taxids]
        collapsed_taxa.append(get_consensus_taxonomy(branch_lineages))
    collapsed_taxa = np.array(collapsed_taxa)
    return collapsed_taxa