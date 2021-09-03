#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:22:35 2021

@author: hernan

TaxIDgenerator, generates a list of the numeric IDs of all taxons(and ranks) of nematoda and platyhelminthes, used to crop the accession2taxid tables
nodes.dmp and names.dmp must have been preprocessed into tab separated tables and named nodes.tsv and names.tsv
"""

#%% Libraries
from glob import glob
import numpy as np
import pandas as pd

#%% main
if __name__ == '__main__':
    # locate tsv files
    while True:
        tax_dir = input('Enter location of nodes.tsv and names.tsv\n')
        nodesfile = glob(f'{tax_dir}/nodes.tsv')
        namesfile = glob(f'{tax_dir}/names.tsv')
        if len(nodesfile) == 0:
            print(f'{tax_dir}/nodes.tsv not found.')
        if len(namesfile) == 0:
            print(f'{tax_dir}/names.tsv not found.')
        if len(nodesfile) == 1 and len(namesfile) == 1:
            break
    # load files
    nodes = pd.read_csv(f'{tax_dir}/nodes.tsv', sep = '\t', header = None)
    names = pd.read_csv(f'{tax_dir}/names.tsv', sep = '\t', header = None)
    
    # prepare columns
    nodes_labs = {0:'tax_id',
                  1:'parent tax_id',
                  2:'rank',
                  3:'embl code',
                  4:'division id',
                  5:'inherited div flag',
                  6:'genetic code id',
                  7:'inherited GC flag',
                  8:'mitochondrial genetic code id',
                  9:'inherited MGC flag',
                  10:'GenBank hidden flag',
                  11:'hidden subtree root flag',
                  12:'comments'}
    names_labs = {0:'tax_id',
                  1:'name_txt',
                  2:'unique_name',
                  3:'name_class'}
    nodes.rename(columns = nodes_labs, inplace = True)
    names.rename(columns = names_labs, inplace = True)
    
    nodes.set_index('tax_id', inplace=True)
    names.set_index('tax_id', inplace = True)

    # locate interest taxes
    nemID = names.loc[names['name_txt'] == 'Nematoda'].index[0]
    pltID = names.loc[names['name_txt'] == 'Platyhelminthes'].index[0]

    out_list = []
    
    for taxID in [nemID, pltID]:
        taxIDs = [taxID]
        curr_parent = [taxID]
        while len(curr_parent) > 0:
            child_taxes = nodes.loc[nodes['parent tax_id'].isin(curr_parent)].index.tolist()
            taxIDs += child_taxes
            curr_parent = child_taxes
        
        out_list += taxIDs

    # convert IDs to strings
    out_list = np.array(out_list).astype(str)
    
    # write output
    out_file = input('Name output file\n')
    with open(out_file, 'w') as handle:
        for item in out_list:
            handle.writelines(f'{item}\n')