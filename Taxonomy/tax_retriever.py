#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:55:29 2021

@author: hernan

This script generates the taxonomic information for each entry in the test files
"""

import pandas as pd
import toolkit as tools

def make_taxid_tab(seqfile, acc2taxtab1, acc2taxtab2, outfile):
    seqdict = tools.make_seqdict(seqfile)
    acc_list = list(seqdict.keys())
    
    tab1 = pd.read_csv(acc2taxtab1, sep = '\t', header = None)
    tab2 = pd.read_csv(acc2taxtab2, sep = '\t', header = None)
    
    taxid_tab = pd.Series(index = acc_list)
    
    for tab in [tab1, tab2]:
        sub_tab = tab.loc[tab[0].isin(acc_list), [0, 2]]
        for acc, taxID in zip(sub_tab[0].to_list(), sub_tab[2].to_list()):
            taxid_tab.at[acc] = taxID
    
    taxid_tab.to_csv(outfile)