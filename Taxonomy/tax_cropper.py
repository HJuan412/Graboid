#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:38:24 2021

@author: hernan
Perform reciprocal grep between accession2taxid files and Acc_lists
"""

#%% libraries
from tax_NCBI_fetcher import check_outdir
import os
import pandas as pd
import subprocess

#%% variables
acc_dir = '/home/hernan/PROYECTOS/Graboid/Databases/13_10_2021-20_15_58/Acc_lists'
gb_file = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Test2/nucl_gb.accession2taxid'
wgs_file = '/home/hernan/PROYECTOS/Graboid/Taxonomy/Test2/nucl_wgs.accession2taxid'

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
    def __init__(self, out_dir, acc_file, tax, mark):
        self.out_dir = out_dir
        check_outdir(out_dir)
        self.make_acclist(acc_file)
        self.check_acc2taxid_files()
        self.out_file = f'{out_dir}/{tax}_{mark}_acc2taxid.tsv'

    def make_acclist(self, acc_file):
        # Generate a temporal accession list to use in grep
        tab = pd.read_csv(acc_file)
        acclist = tab.iloc[:,0].tolist()
        self.acc_list = f'{self.out_dir}/{acc_file.split("/")[-1]}.tmp'
        with open(self.acc_list, 'w') as out_handle:
            out_handle.write('\n'.join(acclist))
        return
    
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
            