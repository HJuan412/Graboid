#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:55:29 2021

@author: hernan

This script generates the taxonomic information for each entry in the test files
"""

import pandas as pd
import sys
sys.path.append('../') # do this to import toolkit
import toolkit as tools

def make_taxid_tab(seqfile, acc2taxtab1, acc2taxtab2, outfile):
    # get accessions present in fasta file
    seqdict = tools.make_seqdict(seqfile)
    acc_list = list(seqdict.keys())

    # get acc2axid tables, merge them
    tab1 = pd.read_csv(acc2taxtab1, sep = '\t', header = None)
    tab2 = pd.read_csv(acc2taxtab2, sep = '\t', header = None)
    merged_tab = pd.concat((tab1, tab2))

    # get matches
    sub_tab = merged_tab.loc[(merged_tab[0].isin(acc_list)) | (merged_tab[1].isin(acc_list)), [1, 2]]

    # generate and store table
    taxid_tab = sub_tab.loc[:, [1,2]].set_index(1).astype(int)
    taxid_tab.to_csv(outfile)
