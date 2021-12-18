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
from tax_reconstructor import ReconstructionDirector
from tax_BOLD_reconstructor import Reconstructor

import pandas as pd

#%% functions
#%% classes
class TaxDirector():
    def __init__(self, out_dir, summ_dir, acc_dir):
        self.out_dir = out_dir
        self.summ_dir = summ_dir
        self.acc_dir = acc_dir
        self.set_workers()
    
    def set_workers(self):
        self.fetcher = TaxFetcher(self.out_dir)
        self.cropper = Cropper(self.out_dir, self.acc_dir)
        self.rec_director = ReconstructionDirector(self.out_dir, ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species'])
    
    def direct_fetching(self):
        self.fetcher.fetch()
    
    def direct_cropping(self):
        self.cropper.merge_accs(self.acc_dir, self.out_dir)
        self.cropper.crop()
        self.cropper.split()
    
    def direct_reconstruction(self):
        self.rec_director.locate_acc2tax()
        self.rec_director.reconstruct()
    
    def direct_BOLD_reconstruction(self):
        # reconstruct for each file
        self.bold_reconstructors = [Reconstructor(self.out_dir, file) for file in glob(f'{self.summ_dir}/*BOLD*')]
        # merge names nodes
        merged_names = pd.concat([rec.name_tab for rec in self.bold_reconstructors])
        merged_nodes = pd.concat([rec.node_tab for rec in self.bold_reconstructors])
        # save names, nodes and taxonomy_dfs tables
        merged_names.to_csv(f'{self.out_dir}/BOLD_names.tsv', sep = '\t')
        merged_nodes.to_csv(f'{self.out_dir}/BOLD_nodes.tsv', sep = '\t')
        for rec in self.bold_reconstructors:
            rec.taxonomy_df.to_csv(f'{rec.out_prefix}_BOLD_acc2taxid.tsv', sep = '\t')
    
    def direct(self):
        print('Fetching taxonomy files...')
        self.direct_fetching()
        print('Cropping taxonomy files...')
        self.direct_cropping()
        print('Reconstructing taxonomies...')
        self.direct_reconstruction()
        print('Reconstructing BOLD taxonomies...')
        self.direct_BOLD_reconstruction()

    # TODO: implement cleanup
    # TODO: remove this method
    def tax_reconstruct(self):
        # NCBI
        # fetch
        fetcher = TaxFetcher(self.out_dir)
        fetcher.fetch()
        # crop
        cropper = Cropper(self.out_dir, self.acc_dir)
        cropper.crop()
        cropper.split()
        # reconstruct
        # reconstruct_all(self.out_dir)
        # BOLD
        # reconstruct for each file
        reconstructors = [Reconstructor(file) for file in glob(f'{self.summ_dir}/*BOLD*')]
        # merge names nodes
        merged_names = pd.concat([rec.name_tab for rec in reconstructors])
        merged_nodes = pd.concat([rec.node_tab for rec in reconstructors])
        # save names, nodes and taxonomy_dfs tables
        merged_names.to_csv(f'{self.out_dir}/BOLD_names.tsv', sep = '\t')
        merged_nodes.to_csv(f'{self.out_dir}/BOLD_nodes.tsv', sep = '\t')
        for rec in reconstructors:
            rec.taxonomy_df.to_csv(f'{rec.out_prefix}_BOLD_acc2taxid.tsv', sep = '\tsv')