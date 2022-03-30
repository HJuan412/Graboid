#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:23:54 2022

@author: hernan
Direct dataset_construction
"""

#%% Libraries
import sys
sys.path.append('data_fetch/dataset_construction')
import os
import ref_catalog
import blast
import matrix2

#%% functions
def check_in_dir(dirname, prefix):
    # check that all corresponding files are present
    suffixes = ['.fasta', '.tax', '.taxguide', '.taxid']
    exp_files = {sfx:None for sfx in suffixes}
    warnings = []
    for sfx in suffixes:
        filename = f'{dirname}/{prefix}{sfx}'
        if os.path.isfile(filename):
            exp_files[sfx] = filename
        else:
            warnings.append(f'WARNING: file {filename} not found in {dirname}')
    return exp_files['.fasta'], warnings

def get_ref_dir(marker):
    ref_path = ref_catalog.catalog[marker.upper()]
    return ref_path

#%% classes
class DatasetDirector():
    def __init__(self, taxon, marker, in_dir, out_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.warnings = []
        self.__check_in_dir()
        self.__get_ref_dir()

    def __check_in_dir(self):
        # check that all corresponding files are present
        self.seq_file = None
        suffixes = ['.fasta', '.tax', '.taxguide', '.taxid']
        exp_files = {sfx:None for sfx in suffixes}
        
        for sfx in suffixes:
            filename = f'{self.in_dir}/{self.taxon}_{self.marker}{sfx}'
            if os.path.isfile(filename):
                exp_files[sfx] = filename
            else:
                self.warnings.append(f'WARNING: file {filename} not found in {self.in_dir}')
        
        self.seq_file = exp_files['.fasta']
    
    def __get_ref_dir(self):
        # TODO handle custom references
        self.ref_path = None
        ref_path = ref_catalog.catalog[self.marker.upper()]
        self.ref_path = ref_path
    
    def set_workers(self):
        self.worker_blast = blast.Blaster(self.taxon, self.marker, self.seq_file, self.ref_path, self.out_dir, self.warn_dir)
        #TODO: fix BOLD seq files, blast hates the gaps ('-'). Fix it at the fetch or split step AHHHHHHH
        self.worker_matrix = matrix2.MatBuilder(self.taxon, self.marker, self.worker_blast.out_file, self.seq_file, self.out_dir, self.warn_dir)
    
    def build_dataset(self, threads = 1):
        print(f'Performing BLAST alignment of {self.taxon} {self.marker}')
        self.worker_blast.blast(threads)
        print(f'Building matrix of {self.taxon} {self.marker}')
        self.worker_matrix.build_matrix()
        print('Finished!')
        
#%% set vars
taxons = ['nematoda', 'platyhelminthes']
markers = ['18s', '28s', 'coi']
for taxon in taxons:
    for marker in markers:
        if taxon == 'nematoda' and marker == '18s':
            pass
        in_dir = f'{taxon}_{marker}/out_dir'
        out_dir = f'{taxon}_{marker}/out_dir'
        warn_dir = f'{taxon}_{marker}/warn_dir'
        director = DatasetDirector(taxon, marker, in_dir, out_dir, warn_dir)
        director.set_workers()
        director.build_dataset()

#%%
# worker_blast = blast.Blaster('nematoda', '18s', seqfile, ref_dir, in_dir, warn_dir)
# worker_blast.blast(4)
# #TODO: fix BOLD seq files, blast hates the gaps ('-'). Fix it at the fetch or split step AHHHHHHH
# worker_matrix = matrix2.MatBuilder('nematoda', '18s', worker_blast.out_file, seqfile, in_dir, tmp_dir, warn_dir)
# worker_matrix.build_matrix()
