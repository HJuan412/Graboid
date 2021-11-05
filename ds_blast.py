#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:28:14 2021

@author: hernan
Blast downloaded sequences against references
"""

#%% libraries
from Bio.Blast.Applications import NcbiblastnCommandline as blast_cline
from glob import glob
import pandas as pd

#%% functions
def blast(query, ref, out_file, threads = 1):
    # perform ungapped blast
    cline = blast_cline(cmd = 'blastn', task = 'blastn', db = ref, query = query, out = out_file, outfmt = "6 qseqid pident length qstart qend sstart send evalue", ungapped = True, num_threads = threads)
    cline()

def build_seq_tab(seq_dir):
    # Generate table for the downloaded sequence files
    seq_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'File'])
    files = glob(f'{seq_dir}/*fasta')
    for file in files:
        split_file = file.split('/')[-1].split('.')[0].split('_')
        taxon = split_file[0]
        marker = split_file[1]
        seq_tab = seq_tab.append({'Taxon':taxon, 'Marker':marker, 'File':file}, ignore_index=True)
    return seq_tab

def build_ref_dict(ref_dir):
    # Generate dict with the base directories for each reference
    ref_dict = {}
    bases = glob(f'{ref_dir}/ref_*_base')
    for base in bases:
        marker = base.split('_')[-2]
        ref_dict[marker] = base
    return ref_dict
#%% classes
class Blaster():
    def __init__(self, in_dir, out_dir, ref_dir, warn_dir):
        # in_dir: directory containing fasta sequences
        # out_dir: directory containing the blast report
        # ref_dir: directory containing the reference database directories
        # warn_dir: directory containing the generated warnings
        self.in_dir = in_dir
        self.seq_tab = build_seq_tab(in_dir)
        self.out_dir = out_dir
        self.ref_dict = build_ref_dict(ref_dir)
    
    def check_seq_files(self):
        if len(self.seq_tab) == 0:
            with open(f'{self.warn_dir}/blast.warn', 'a') as handle:
                handle.write(f'No sequence files found in the {self.in_dir} directory.\n')
            return False
        return True

    def check_ref_files(self):
        if len(self.ref_dict) == 0:
            with open(f'{self.warn_dir}/blast.warn', 'a') as handle:
                handle.write(f'No reference database directories found in the {self.ref_dir} directory.\n')
            return False
        return True

    def blast(self, threads = 1):
        if self.check_seq_files() and self.check_ref_files():
            for marker, marktab in self.seq_tab.groupby('Marker'):
                base_dir = self.ref_dict[marker]
                ref = f'{base_dir}/ref_{marker}'
                for _idx, row in marktab.iterrows():
                    taxon = row['Taxon']
                    file = row['File']
                    out_file = f'{self.out_dir}/{taxon}_{marker}.tab'
                    
                    print(f'Blasting {taxon} {marker}')
                    try:
                        blast(file, ref, out_file, threads)
                    except:
                        # TODO develop warning
                        with open(f'{self.warn_dir}/blast.warn', 'a') as handle:
                            handle.write(f'Unable to blast {taxon} {marker}')