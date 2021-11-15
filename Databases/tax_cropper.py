#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:38:24 2021

@author: hernan
Perform grep between accession2taxid files and Acc_lists
"""

#%% libraries
from glob import glob
from tax_NCBI_fetcher import check_outdir
import os
import pandas as pd
import subprocess

#%% functions
def grep_acc2taxid(acc_list, file, out_file):
    # Locate relevant entries in the acc2taxid file, extract them to out_file
    grep_cline = ['grep', '-w', '-f', acc_list, file]
    with open(out_file, 'a') as out_handle:
        process = subprocess.Popen(grep_cline, stdout = out_handle)
        process.wait()
        print('Done!')

#%% classes
class Cropper():
    def __init__(self, out_dir, acc_dir):
        self.out_dir = out_dir
        check_outdir(out_dir)
        self.merge_accs(acc_dir, out_dir)
        self.check_acc2taxid_files()
        self.out_file = f'{out_dir}/acc2taxid_cropped.tsv'
    
    def merge_accs(self, acc_dir, out_dir):
        # merge all the acc_list files into a single list to reduce redundant searches
        self.acc_files = glob(f'{acc_dir}/*NCBI.acc') # list acc_list files to split records later
        self.acc_list = out_file = f'{out_dir}/accs.tmp' # merged accessions file
        accs = set()
        for acc_file in self.acc_files:
            acc_tab = pd.read_csv(acc_file)
            acclist = set(acc_tab.iloc[:,0].tolist())
            accs = accs.union(acclist)
        with open(out_file, 'a') as handle:
            handle.write('\n'.join(accs))
    
    def generate_filename(self, acc_file):
        # Generate out file for a taxon-marker cropped file
        file = acc_file.split('/')[-1][:-8]
        filename = f'{self.out_dir}/{file}acc2taxid.tsv'
        return filename

    def clear_acclist(self):
        # remove acclist
        if os.path.isfile(self.acc_list):
            os.remove(self.acc_list)
    
    def clear_outfile(self):
        # remove out_file
        if os.path.isfile(self.out_file):
            os.remove(self.out_file)

    def check_acc2taxid_files(self):
        # check presence of acc2taxid files
        gb = f'{self.out_dir}/nucl_gb.accession2taxid'
        wgs = f'{self.out_dir}/nucl_wgs.accession2taxid'
        if os.path.isfile(gb):
            self.gb = gb
        else:
            self.gb = None
        if os.path.isfile(wgs):
            self.wgs = wgs
        else:
            self.wgs = None
    
    def crop(self):
        # delete out_file (if present) generate new one with present acc2taxid files
        self.clear_outfile()
        if not self.gb is None:
            grep_acc2taxid(self.acc_list, self.gb, self.out_file)
        if not self.wgs is None:
            grep_acc2taxid(self.acc_list, self.wgs, self.out_file)
    
    def split(self):
        # from the cropped file distribute matches of each taxon/marker pair
        cropped_tab = pd.read_csv(self.out_file, sep = '\t', index_col = 0, header = None)
        for acc_file in self.acc_files:
            accset = set(pd.read_csv(acc_file).iloc[:,0])
            accset = accset.intersection(cropped_tab.index)
            acc2taxid_tab = cropped_tab.loc[accset,:]
            out_file = self.generate_filename(acc_file)
            acc2taxid_tab.to_csv(out_file, sep = '\t')
