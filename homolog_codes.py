#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:33:47 2021

@author: hernan
Este script se usa para homologar los ID taxonomicos de BOLD y NCBI y generar una tabla taxonómica única
"""

import pandas as pd

#%% classes
class Homologuer():
    def __init__(self, bold_tax, bold_names, bold_nodes, ncbi_tax, ncbi_names, ncbi_nodes, ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum']):
        self.bold_tax_file = bold_tax
        self.bold_name_file = bold_names
        self.bold_node_file = bold_nodes
        self.ncbi_tax_file = ncbi_tax
        self.ncbi_name_file = ncbi_names
        self.ncbi_node_file = ncbi_nodes
        self.ranks = ranks
        self.load_files()
        self.set_modded_tab()
        self.get_max_ncbi()
    
    def load_files(self):
        self.bold_tax = pd.read_csv(self.bold_tax_file, sep = '\t', index_col=0)
        self.bold_names = pd.read_csv(self.bold_name_file, sep = '\t', index_col=0)['tax_name']
        self.bold_nodes = pd.read_csv(self.bold_node_file, sep = '\t', index_col=0)
        self.ncbi_tax = pd.read_csv(self.ncbi_tax_file, sep = '\t', index_col = 0)
        self.ncbi_names = pd.read_csv(self.ncbi_name_file, sep = '\t', index_col=0, header = None, names = ['tax_name', 'who', 'cares'])['tax_name']
        self.ncbi_nodes = pd.read_csv(self.ncbi_node_file, sep = '\t', index_col=0, header = None)
        
        self.ncbi_tax['db'] = 'NCBI'
        self.bold_tax['db'] = 'BOLD'
    
    def set_modded_tab(self):
        # creates (or resets) the modified bold tax tab as a copy of the original
        self.bold_tax_modded = self.bold_tax.copy()
        self.bold_names_modded = self.bold_names.copy()
        self.bold_nodes_modded = self.bold_nodes.copy()
    
    def get_max_ncbi(self):
        ncbi_ids = self.ncbi_tax.loc[:,self.ranks].to_numpy()
        self.max_ncbi = ncbi_ids.max() + 1
    
    def fix_homologs(self):
        # locate homologs (and non-homologs) present at each rank
        for rk in self.ranks:
            # find the nodes present in the current rank for BOLD and NCBI
            # the following var are arrays containing taxIDs
            bold_nodes_in_rank = self.bold_nodes.loc[self.bold_nodes['rank'] == rk].index.astype(int)
            ncbi_nodes_in_rank = self.ncbi_nodes.loc[self.ncbi_nodes[2] == rk].index.astype(int)
            
            # get the taxID and tax name of every node present in the current rank
            # following vars are series with index=taxIDs and values=tax names
            bold_codes = self.bold_names.loc[bold_nodes_in_rank]
            ncbi_codes = self.ncbi_names.loc[ncbi_nodes_in_rank]
            
            # flip the values of the code arrays (index=tax names, values=taxIDs)
            flipped_bold = pd.Series(data = bold_codes.index, index = bold_codes.values)
            flipped_ncbi = pd.Series(data = ncbi_codes.index, index = ncbi_codes.values)
            
            # get the tax names for BOLD and NCBI
            bold_set = set(bold_codes.tolist())
            ncbi_set = set(ncbi_codes.tolist())
            # get taxons present in both databases in the current rank
            intersect = bold_set.intersection(ncbi_set)
            # get the taxons only present in BOLD and their taxIDs
            only_bold = bold_set.difference(ncbi_set)
            only_bold_codes = flipped_bold.loc[only_bold].tolist()
        
            # set replacement values for the homolog nodes
            homo_dict = {flipped_bold.loc[h]:flipped_ncbi.loc[h] for h in intersect}
            
            # set replacement values for the non homolog nodes
            # shift all values by the maximum ID present in the NCBI table
            # this step prevents repeated ids between bold and ncbi
            shift_dict = {o: o+self.max_ncbi for o in only_bold_codes}
            
            fix_dict = homo_dict | shift_dict # merge both dicts

            # replace values in bold_tax_tab
            self.bold_tax_modded[rk].replace(fix_dict, inplace = True)
            
            # replace fixed values in bold_names and bold_nodes
            self.bold_names_modded.rename(index = fix_dict, inplace = True)
            self.bold_nodes_modded.rename(index = fix_dict, inplace = True)
            self.bold_nodes_modded['parent tax_id'].replace(fix_dict, inplace = True)
    
    def get_homologued(self):
        # get a homologued taxonomy table and names series
        self.homolog_tax = pd.concat([self.ncbi_tax, self.bold_tax_modded])
        self.homolog_names = pd.concat([self.ncbi_names, self.bold_names_modded])