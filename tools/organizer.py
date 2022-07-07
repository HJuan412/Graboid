#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:01:40 2022

@author: hernan
Organize file manipulation along the entire graboid process
"""

#%% libraries
import os
import pickle
import sys

#%% classes
# an instance of this class is generated at the beginning of the pipeline
# used to access all files and folders of the process
class graboid_inventory:
    def __init__(self, root):
        self.map = graboid_map(root)
        self._ref_seqs = None
        self._ref_tax = None
        self._mark_ref = None
        self._blastdb = None
        self._ref_map = None
        self._ref_aln = None
        self._query_map = None
        self._query_aln = None
        self._tunning_data = None
        self._tunning_report = None
        self._classif_results = None
    
    # path to reference fasta file
    @property
    def ref_seqs(self):
        return self._ref_seqs
    @ref_seqs.setter
    def ref_seqs(self, ref_seqs):
        self._ref_seqs = ref_seqs
    
    # path to reference taxonomy table
    @property
    def ref_tax(self):
        return self._ref_tax
    @ref_tax.setter
    def ref_tax(self, ref_tax):
        self._ref_tax = ref_tax
    
    # path to the marker reference sequence file
    @property
    def mark_ref(self):
        return self._mark_ref
    @mark_ref.setter
    def mark_ref(self, mark_ref):
        self._mark_ref = mark_ref
    
    # path to the blast database files
    @property
    def blastdb(self):
        return self._blastdb
    @blastdb.setter
    def blastdb(self, blastdb):
        self._blastdb = blastdb
    
    # path to the reference sequence blast result
    @property
    def ref_map(self):
        return self._ref_map
    @ref_map.setter
    def ref_map(self, ref_map):
        self._ref_map = ref_map
    
    # path to the reference alignment matrix
    @property
    def ref_aln(self):
        return self._ref_aln
    @ref_aln.setter
    def ref_aln(self, ref_aln):
        self._ref_aln = ref_aln
    
    # path to the query blast result
    @property
    def query_map(self):
        return self._query_map
    @query_map.setter
    def query_map(self, query_map):
        self._query_map = query_map
    
    # path to the query alignment matrix
    @property
    def query_aln(self):
        return self._query_aln
    @query_aln.setter
    def query_aln(self, query_aln):
        self._query_aln = query_aln
    
    # path to the tunning data file
    @property
    def tunning_data(self):
        return self._tunning_data
    @tunning_data.setter
    def tunning_data(self, tunning_data):
        self._tunning_data = tunning_data
    
    # path to the tunning report
    @property
    def tunning_report(self):
        return self._tunning_report
    @tunning_report.setter
    def tunning_report(self, tunning_report):
       self._tunning_report = tunning_report
    
    # path to the classification results
    @property
    def classif_results(self):
        return self._classif_results
    @classif_results.setter
    def classif_results(self, classif_results):
        self._classif_results = classif_results

# this class is used to quickly access the graboid folders
class graboid_map:
    def __init__(self, root):
        self.root = root
        self.ref_seqs = f'{root}/ref_seqs'
        self.tax = f'{root}/tax'
        self.tmp = f'{root}/tmp'
        self.warn = f'{root}/warn'
        self.logs = f'{root}/logs'
        self.mark_ref = f'{root}/aln/blast/ref'
        self.blastdb = f'{root}/aln/blast/blastdb'
        self.refmap = f'{root}/aln/blast/refmap'
        self.querymap = f'{root}/aln/blast/querymap'
        self.ref_matrix = f'{root}/aln/matrix/ref'
        self.query_matrix = f'{root}/aln/matrix/query'
        self.tunning = f'{root}/tunning'
        self.classif = f'{root}/classif'

#%% functions
# generate directory structure
def get_branches(root):
    branches = [f'{root}/ref_seqs',
                f'{root}/tax',
                f'{root}/tmp',
                f'{root}/warn',
                f'{root}/logs',
                f'{root}/aln/blast/ref',
                f'{root}/aln/blast/blastdb',
                f'{root}/aln/blast/refmap',
                f'{root}/aln/blast/querymap',
                f'{root}/aln/matrix/ref',
                f'{root}/aln/matrix/query',
                f'{root}/tunning',
                f'{root}/classif']
    return branches

# check that the directory structure doesn't already exist (or if it does, check that it's empty)
def check_branches(branches):
    check = True
    for branch in branches:
        if sys.path.isdir(branch):
            if len(os.listdir(branch)) > 0:
                check = False
    return check

# create the graboid directory tree
def build_dir_tree(branches):
    for branch in branches:
        os.makedirs(branch)

# save and retrieve the inventory
def save_inventory(inventory, filename):
    with open(filename, 'wb') as out_handle:
        pickle.dump(inventory, out_handle)

def load_inventory(inventory):
    with open(inventory, 'rb') as in_handle:
        return pickle.load(in_handle)