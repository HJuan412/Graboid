#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:06:22 2021

@author: hernan
Direct taxonomy reconstruction
"""
#%% libraries
from glob import glob
from tax_NCBI_fetcher import TaxFetcher
from tax_cropper import Cropper
from tax_reconstructor import reconstruct_all
from tax_BOLD_reconstructor import Reconstructor

import pandas as pd

#%% functions
#%%
out_dir = '13_10_2021-20_15_58/Taxonomy_files'
acc_dir = '13_10_2021-20_15_58/Acc_lists'
summ_dir = '13_10_2021-20_15_58/Summaries'
# NCBI
# fetch
fetcher = TaxFetcher(out_dir)
fetcher.fetch()
# crop
cropper = Cropper(out_dir, acc_dir)
cropper.crop()
cropper.split()
# reconstruct
reconstruct_all(out_dir)
# BOLD
# reconstruct for each file
reconstructors = [Reconstructor(file) for file in glob(f'{summ_dir}/*BOLD*')]
# merge names nodes
merged_names = pd.concat([rec.name_tab for rec in reconstructors])
merged_nodes = pd.concat([rec.node_tab for rec in reconstructors])
# save names, nodes and taxonomy_dfs tables
merged_names.to_csv(f'{out_dir}/BOLD_names.tsv', sep = '\t')
merged_nodes.to_csv(f'{out_dir}/BOLD_nodes.tsv', sep = '\t')
for rec in reconstructors:
    rec.taxonomy_df.to_csv(f'{rec.out_prefix}_BOLD_acc2taxid.tsv', sep = '\tsv')