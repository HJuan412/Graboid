#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:43:13 2024

@author: hernan
"""

#%% modules
import numpy as np
import pandas as pd

from concept_learner.ranks import Rank

#%% functions
def one_hot_encode(matrix):
    encoded = np.stack([matrix == 0,
                        matrix == 1,
                        matrix == 2,
                        matrix == 3,
                        matrix == 4], axis=2)
    return encoded

def flatten_lineage(R_lineage):
    # flatten lineage table, generate dataframe with columns: [idx, TaxId], filter out instances with unknown taxon
    lineage_flat = pd.concat(R_lineage[rk] for rk in R_lineage.columns).to_frame(name='TaxId').reset_index(names='idx')
    lineage_flat.set_index('TaxId', inplace=True)
    try:
        lineage_flat.drop(0, axis=0, inplace=True)
    except:
        pass
    return lineage_flat

# Call cleanup
def get_orphaned(lineage):
    """
    Identify orphaned taxa (taxa that are missing calls for one or more
    ancestor taxa)

    Parameters
    ----------
    lineage : pandas.DataFrame
        Lineage calls table for a single query.

    Returns
    -------
    orphaned : pandas.Series
        Boolean series indicating orphaned taxa.

    """
    orphaned = pd.Series(False, index=lineage.index)
    lineage = pd.concat([lineage.iloc[:,0], lineage], axis=1)
    listed_parents = pd.Series({tax:row[np.argmax(row == 0) - 2] for tax, row in lineage.iterrows()})
    for tax, par in listed_parents.items():
        orphaned[tax] = par not in listed_parents.index.droplevel(0)
    return orphaned
    
def get_truncated(lineage, level=-1):
    """
    Identify truncated lineages (taxa that are incomplete before the level
    specified in argument level).

    Parameters
    ----------
    lineage : pandas.DataFrame
        Lineage calls table for a single query.
    level : int, optional
        Negative integer indicating the last level for which to check
        truncation. The default is -1 (second to last).

    Returns
    -------
    pandas.Series
        Boolean series indicating truncated taxa.

    """
    def has_children(parents, children):
        has = pd.Series(False, index=parents.index)
        has.loc[np.intersect1d(children.unique(), parents.index)] = True
        return has
    
    truncated = pd.Series(True, index=lineage.index.get_level_values(1))

    # add rank to index
    l = lineage.copy()
    l.index = pd.MultiIndex.from_arrays([l.columns[np.argmax(l==0, axis=1)-1], l.index.get_level_values(1)])
    ranks = l.columns[:level][::-1]

    # iterate parent/child ranks from lowest to highest
    for rk, child_rk in zip(ranks[1:], ranks[:-1]):
        try:
            rk_taxa = l.loc[rk]
            children = l.loc[child_rk, rk]
            non_trunc = has_children(rk_taxa, children)
            trunc = non_trunc[~non_trunc]
            l.drop(index=trunc.index, level=1, inplace=True)
        except KeyError:
            break
    truncated.loc[l.index.get_level_values(1)] = False
    truncated.index = lineage.index
    return truncated
    
def get_lineage_calls(calls_tab, lineage_tab):
    """
    Generates table of index: (query, tax) and columns: ranks. Indicates the
    called lineages for each rank, will be used to filter out orphaned &
    truncated taxa

    Parameters
    ----------
    calls_tab : pandas.DataFrame
        Calls tab returned by the classify method. Index: query, columns: (rank, taxa)
    lineage_tab : pandas.DataFrame
        Lineage table of reference dataset.

    Returns
    -------
    lineage_calls : pandas.DataFrame
        Table detailing called taxa for each taxon/rank/query. Index: (query, tax) and columns: ranks.

    """
    # get array of called taxa
    taxa = calls_tab.columns.get_level_values(1)

    # get called lineage for each query
    lineage_calls = {}
    for idx, q in enumerate(calls_tab.values):
        lineage_calls[idx] = lineage_tab.loc[taxa[q]]
    lineage_calls = pd.concat(lineage_calls)
    return lineage_calls

def evaluate_lineage_calls(lineage_calls, level=-1):
    """
    Identify orphaned and/or truncated lineages

    Parameters
    ----------
    lineage_calls : pandas.DataFrame
        Table returned by function get_lineage_calls.
    level : int, optional
        Negative integer indicating the last level to check for truncated lineages.
        This is because low recall at lower level is more likely and this could
        cause the entire lineage to be marked as truncated.
        Level value indicates position from the last called rank.
        The default is -1.

    Returns
    -------
    lin_eval : pandas.DataFrame
        Table indicating the orphaned and truncated status for each called
        taxon for each query. Index: (query, taxon), columns: Orphaned, Truncated

    """
    lin_eval = pd.DataFrame(False, index=lineage_calls.index, columns='Orphaned Truncated'.split())
    for idx, subtab in lineage_calls.groupby(level=0):
        lin_eval.loc[subtab.index, 'Orphaned'] = get_orphaned(subtab)
        lin_eval.loc[subtab.index, 'Truncated'] = get_truncated(subtab, level)
    return lin_eval
#%% classes
class Result:
    def __init__(self, signals, calls, lineage_tab, ranks, level=-1):
        self.signals = signals
        self.calls = calls
        self.sum_calls = calls.T.groupby(level=0).agg('sum').T[ranks]
        lineage_calls = get_lineage_calls(calls, lineage_tab)
        lineage_evals = evaluate_lineage_calls(lineage_calls, level)
        self.lineage_calls = pd.concat([lineage_calls, lineage_evals], axis=1)
    
    def clean_calls(self):
        clean_calls = self.lineage_calls.query('~Orphaned & ~Truncated')
        # build new calls tab with cleaned calls
        new_call_tab = {}
        for col in clean_calls.columns:
            rk_calls = pd.DataFrame(False, index=clean_calls.index.get_level_values(0).unique(), columns=clean_calls[col].unique())
            for q, subtab in clean_calls.groupby(level=0):
                rk_calls.loc[q, subtab[col].unique()] = True
            new_call_tab[col] = rk_calls
            
        new_call_tab = pd.concat(new_call_tab, axis=1)
        new_call_tab.drop(columns=0, level=1, inplace=True)
        return new_call_tab
        
class ConceptLearner:
    def __getitem__(self, rank):
        return self.ranks[rank]
    
    def load_data(self, matrix, lineage_tab, lineage_collapsed, names_tab):
        self.matrix = one_hot_encode(matrix)
        self.lineage_tab = lineage_tab
        self.lineage_collapsed = lineage_collapsed
        self.lineage_flat = flatten_lineage(lineage_collapsed)
        self.names_tab = names_tab
        self.ranks = {rk:Rank(rk) for rk in lineage_tab.columns}
    
    def learn(self, threads=1):
        for rank in self.ranks.values():
            rank.learn(self.matrix, self.lineage_collapsed, self.lineage_flat, threads=threads)
    
    def classify(self, query, clear_multi, *ranks, **kwargs):
        if len(ranks) == 0:
            ranks = self.ranks.keys()
        
        signals = {}
        called_taxa = {}
        for rk in ranks:
            rank = self.ranks[rk]
            rk_calls, rk_signals = rank.classify(query, clear_multi)
            called_taxa[rk] = rk_calls
            signals[rk] = rk_signals
        signals = pd.concat(signals, axis=1)
        called_taxa = pd.concat(called_taxa, axis=1)
        self.result = Result(signals, called_taxa, self.lineage_tab, ranks)
