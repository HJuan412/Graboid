#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:25:41 2024

@author: hernan

This script contains the data holder class used to load graboid databases and query files
"""

#%% libraries
import json
import numpy as np
import os
import pandas as pd

# graboid libraries
from mapping import mapping as mpp
from preprocess import sequence_collapse, consensus_taxonomy
#%% functions
def load_map(map_file, acc_file):
    # map_file: __map.npz file
    # acc_file: __map.acc file
    
    # load a map file and the corresponding accession file
    # from npz file, extract: alignment map, bounds array, coverage array
    # calculate normalized coverage
    # TODO: could incorporate accession list to npz file?
    map_ = np.load(map_file)
    matrix = map_['matrix']
    bounds = map_['bounds']
    coverage = map_['coverage']
    coverage_norm = coverage / coverage.max()
    # retrieve accession list
    with open(acc_file, 'r') as acc_:
        accs = acc_.read().splitlines()
    
    return matrix, accs, bounds, coverage, coverage_norm

#%% classes
class R:
    def load(self, ref_dir):
        try:
            with open(f'{ref_dir}/meta.json', 'r') as handle:
                meta = json.load(handle)
        except FileNotFoundError:
            raise Exception('Meta file not found, verify that the given reference directory is a graboid database.')
        
        # load map files
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(meta['map_mat_file'], meta['map_acc_file'])
        
        # load taxonomy data
        ref_tax = pd.read_csv(meta['tax_file'], names=['Accession', 'TaxId'], skiprows=[0])
        self.y = ref_tax.set_index('Accession').loc[self.accs, 'TaxId'].to_numpy()
        self.lineage_tab = pd.read_csv(meta['lineages_file'], index_col=0)
        self.names_tab = pd.read_csv(meta['names_file'], index_col=0)['SciName']
        
        self.lineage = self.lineage_tab.loc[self.y] # subsection of lineage_tab corresponding to the reference instances
        
        # load guide blast reference
        self.blast_db = meta['guide_db']
    
    def filter_rank(self, rank):
        # filter reference, get indexes of instances with known values at the given rank
        self.rank = rank
        self.valid = self.lineage[rank].to_numpy() != 0
    
    def collapse(self, filtered_sites, max_unk_thresh=.2):
        # collapse reference matrix and taxonomy data
        filtered_map = self.map[self.valid][:, filtered_sites]
        filtered_accs = self.y[self.valid]
        
        self.collapsed, self.branches = sequence_collapse.sequence_collapse(filtered_map, max_unk_thresh)
        self.y_collapsed = consensus_taxonomy.collapse_taxonomies(self.branches, filtered_accs, self.lineage_tab)
        self.lineage_collapsed = self.lineage_tab.loc[self.y_collapsed].reset_index(drop=True)

class Q:
    def load(self, qry_file, qry_dir, blast_db, evalue=0.0005, dropoff=0.05, min_height=0.1, min_width=2, threads=1, qry_name='QUERY'):
        map_prefix = f'{qry_dir}/{qry_name}'
        os.makedirs(qry_dir, exist_ok=True)
        if mpp.check_fasta(qry_file) == 0:
            raise Exception(f'Error: Query file {qry_file} is not a valid fasta file')
        
        qry_map_file, qry_acc_file, nrows, ncols = mpp.build_map(qry_file, blast_db, map_prefix, threads=threads, clip=False)
        
        # load map files
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(qry_map_file, qry_acc_file)
    
    def load_quick(self, qry_map_file, qry_acc_file):
        # shorter version of load query, load pre-generated query map files
        # load query dataset
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(qry_map_file, qry_acc_file)
    
    def collapse(self, filtered_sites):
        # select sites from query & collapse
        filtered_map = self.map[:, filtered_sites]
        self.collapsed, self.branches = sequence_collapse.sequence_collapse(filtered_map)
        
        # build expanded query map, dataframe mapping each query sequence to its branch
        expanded = pd.DataFrame(-1, index=pd.Index(self.accs, name='Query'), columns=['Branch'])
        for br_idx, branch in enumerate(self.branches):
            expanded.iloc[branch, 0] = br_idx
        self.expanded = expanded
        
class DataHolder:
    def load_reference(self, ref_dir):
        self.R = R()
        self.R.load(ref_dir)
        self.map_shape = self.R.map.shape
    
    def load_query(self, qry_file, qry_dir='.', evalue=0.0005, dropoff=0.05, min_height=0.1, min_width=2, threads=1, qry_name='QUERY'):
        self.Q = Q()
        self.Q.load(qry_file,
                    qry_dir,
                    self.R.blast_db,
                    evalue=evalue,
                    dropoff=dropoff,
                    min_height=min_height,
                    min_width=min_width,
                    threads=threads,
                    qry_name=qry_name)
        
    def load_query_short(self, qry_map_file, qry_acc_file):
        self.Q = Q()
        self.Q.load_quick(qry_map_file, qry_acc_file)
    
    def filter_data(self, min_cov=.95, rank='family'):
        # min_cov : minimum coverage (percentage of max coverage) to select the columns by
        
        # filter columns by coverage
        filtered = (self.Q.coverage_norm >= min_cov) & (self.R.coverage_norm > 0)
        self.filtered_sites = np.arange(len(filtered))[filtered]
        
        # filter reference, get indexes of instances with known values at the given rank
        self.R.filter_rank(rank)
    
    def collapse(self, max_unk_thresh=.2):
        self.R.collapse(self.filtered_sites, max_unk_thresh)
        self.Q.collapse(self.filtered_sites)
