#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:20:26 2021

@author: hernan
"""
#%% lib
import pandas as pd
#%% variables
ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum']
#%% functions
def build_node_tab(tab, ranks, last_known_parent = 0):
    # ranks must be DESCENDING
    # recursively connects nodes, generates table with index taxID and columns [parent_tax_ID and rank]
    nodes = pd.DataFrame(columns = ['parent_tax_ID', 'rank'], dtype = int)
    rk = ranks[0]
    colname = f'{rk}_taxID'
    for tax in list(tab.loc[tab[colname].notna(), colname].unique()):
        nodes.at[tax] = [last_known_parent, rk]
    
    if len(ranks) > 1:
        for parent, subtab in tab.groupby(colname):
            subnodes = build_node_tab(subtab, ranks[1:], parent)
            nodes = pd.concat([nodes, subnodes], axis = 0)
        
        nantab = tab.loc[tab[colname].isna()]
        nannodes = build_node_tab(nantab, ranks[1:], last_known_parent)
        nodes = pd.concat([nodes, nannodes], axis = 0)
    
    nodes.set_index(nodes.index.astype(int), inplace = True)
    nodes['parent_tax_ID'] = nodes['parent_tax_ID'].astype(int)
    return nodes

#%% classes
class Reconstructor():
    def __init__(self, out_dir, in_file, ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum']):
        self.out_dir = out_dir
        self.ranks = ranks
        self.load_data(in_file)
        self.build_name_tab()
        self.node_tab = build_node_tab(self.tax_tab, ranks[::-1])

    def load_data(self, file):
        # open bold tab and extract the relevant columns
        bold_tab = pd.read_csv(file, sep = '\t', encoding = 'latin-1', index_col = 1, low_memory = False) # latin-1 to parse BOLD files
        
        self.tax_tab = bold_tab[['phylum_taxID', 'phylum_name',
                                      'class_taxID', 'class_name',
                                      'order_taxID', 'order_name',
                                      'family_taxID', 'family_name',
                                      'subfamily_taxID', 'subfamily_name',
                                      'genus_taxID', 'genus_name',
                                      'species_taxID', 'species_name',
                                      'subspecies_taxID', 'subspecies_name']]
    
    def build_name_tab(self):
        # build a taxID:Name table
        names = pd.Series(name='tax_name', dtype = str)
        for rk in self.ranks:
            idxcol = f'{rk}_taxID'
            namecol = f'{rk}_name'
            
            rows = self.tax_tab.loc[self.tax_tab[idxcol].notna()].index.tolist()
            taxids = self.tax_tab.loc[rows, idxcol].values.astype(int).tolist()
            taxnames = self.tax_tab.loc[rows, namecol].values.tolist()
            
            for tid, tnm in zip(taxids, taxnames):
                names.at[tid] = tnm
        self.name_tab = names
    
    def build_acc2tax_tab(self):
        # generate an acc:taxID table
        acc2taxID = pd.Series(name = 'TaxID', dtype = int)
            
        unid_recs = self.tax_tab.index
        for rnk in self.ranks:
            rank_col = f'{rnk}_taxID'
            subtab = self.tax_tab.loc[unid_recs, rank_col]
            identified_recs = subtab.loc[subtab.notna()]
            acc2taxID = pd.concat([acc2taxID, identified_recs])
            unid_recs = subtab.loc[subtab.isna()].index
            if len(unid_recs) == 0:
                break
        self.acc2taxID = acc2taxID.astype(int)
