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
# input handling
def load_data(file, level):
    bold_tab = pd.read_csv(file, sep = '\t', encoding = 'latin-1', index_col = 1) # latin-1 to parse BOLD files
    
    tax_tab = bold_tab[['phylum_taxID',
           'phylum_name', 'class_taxID', 'class_name', 'order_taxID', 'order_name',
           'family_taxID', 'family_name', 'subfamily_taxID', 'subfamily_name',
           'genus_taxID', 'genus_name', 'species_taxID', 'species_name',
           'subspecies_taxID', 'subspecies_name']]
    
    filtered_tab = tax_tab.loc[tax_tab[f'{ranks[level]}_name'].notna()]
    
    return filtered_tab

# table foundations
def get_names_nodes(filtered_tab):
    names_dict = {'tax_ID':[], 'tax_name':[]}
    nodes_dict = {'tax_ID':[], 'parent_tax_ID':[], 'rank':[]}
    for idx, lev in enumerate(ranks):
        name_col = f'{lev}_name'
        id_col = f'{lev}_taxID'
        parent_level = idx + 1
        
        for tax, subtab in filtered_tab.groupby(name_col):
            taxid = subtab[id_col].values[0]
            names_dict['tax_ID'].append(taxid)
            names_dict['tax_name'].append(tax)
    
            if parent_level < len(ranks):
                parent_rank = ranks[parent_level]
                parent_id_col = f'{parent_rank}_taxID'
                parent_taxid = subtab[parent_id_col].values[0]
                
                nodes_dict['tax_ID'].append(taxid)
                nodes_dict['parent_tax_ID'].append(parent_taxid)
                nodes_dict['rank'].append(lev)
    return names_dict, nodes_dict

def build_names_tab(names_dict):
    names_tab = pd.DataFrame.from_dict(names_dict)
    names_tab['tax_ID'] = names_tab['tax_ID'].astype(int)
    names_tab.set_index('tax_ID', inplace = True)
    return names_tab

def build_nodes_tab(nodes_dict):
    nodes_tab = pd.DataFrame.from_dict(nodes_dict)
    nodes_tab['tax_ID'] = nodes_tab['tax_ID'].astype(int)
    nodes_tab['parent_tax_ID'] = nodes_tab['parent_tax_ID'].astype(int)
    nodes_tab.set_index('tax_ID', inplace = True)
    return nodes_tab

def build_acc2taxid_tab(filtered_tab, last_level):
    acc2taxID = pd.Series(index = filtered_tab.index, name = 'TaxID', dtype = int)

    subtab = filtered_tab.copy()
    for l in range(last_level + 1):
        level = ranks[l]
        name_col = f'{level}_name'
        id_col = f'{level}_taxID'
        
        sub_subtab = subtab.loc[subtab[name_col].notna()]
        taxids = sub_subtab[id_col].astype(int)
        acc2taxID.update(taxids)
        subtab.drop(sub_subtab.index, inplace = True)
    
    return acc2taxID

# director
def process_file(file, level : int, to_file = False, out_file = None):
    filtered_tab = load_data(file, level)
    names_dict, nodes_dict = get_names_nodes(filtered_tab)
    names_tab = build_names_tab(names_dict)
    nodes_tab = build_nodes_tab(nodes_dict)
    acc2taxid = build_acc2taxid_tab(filtered_tab, level)
    
    if to_file:
        if out_file is None:
            out_file = file.split('_')[0]
        
        names_tab.to_csv(f'{out_file}/names.csv')
        nodes_tab.to_csv(f'{out_file}/nodes.csv')
        acc2taxid.to_csv(f'{out_file}/acc2taxid.csv')