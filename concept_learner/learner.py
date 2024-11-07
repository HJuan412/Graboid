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

#%% classes
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
    
    def classify(self, query, *ranks):
        if len(ranks) == 0:
            ranks = self.ranks.keys()
        
        signals = {}
        called_taxa = {}
        for rk in ranks:
            rank = self.ranks[rk]
            rk_calls, rk_signals = rank.classify(query)
            
            # filter out taxa with no calls (signal value of 1)
            rk_calls['Sum'] = rk_calls.sum(axis=1)
            called_taxa[rk] = rk_calls
            signals[rk] = rk_signals
        signals = pd.concat(signals, axis=1)
        called_taxa = pd.concat(called_taxa, axis=1)
        return called_taxa, signals
