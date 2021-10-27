#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:21:34 2021

@author: hernan
Get raw tax files (names.dmp, nodes.dmp and acc2tax) from NCBI repository.
"""

from ftplib import FTP
import os
import pandas as pd
import subprocess
import zipfile

#%% variables
taxdmp = 'pub/taxonomy/taxdmp.zip'
gb_acc2tax = 'pub/taxonomy/accession2taxid/nucl_gb.accession2taxid.gz'
wgs_acc2tax = 'pub/taxonomy/accession2taxid/nucl_wgs.accession2taxid.gz'
#%% functions
def check_outdir(out_dir):
    # create out_dir if it's not already present
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    return

def check_failed_file(file):
    # remove a failed file
    if os.path.isfile(file):
        os.remove(file)
    return

def extract_taxdmp(file, out_dir, verbose = False):
    # extract names.dmp and nodes.dmp from taxdmp.zip
    with zipfile.ZipFile(file, 'r') as zip_handle:
        if verbose:
            print('Extracting names.dmp and nodes.dmp from taxdmp.zip')
        zip_handle.extract('names.dmp', out_dir)
        zip_handle.extract('nodes.dmp', out_dir)
    return

def extract_acc2tax(file, wait = False, verbose = False):
    # extract acc2taxid from gz file
    if verbose:
        print(f'Extracting {file.split("/")[-1]}')
    gzip_cline = ['gunzip', '-k', file]
    process = subprocess.Popen(gzip_cline)
    if wait:
        process.wait()
        if verbose:
            print('Finished')
    return

def dmp2tsv(file, wait = False, verbose = False):
    # convert dmp file into tsv table
    # dmp files use "   |   " as field separators, replace by single tab with sed 's/\t|//g'
    if verbose:
        print(f'Processing {file.split("/")[-1]}')
    out_path = file.replace('.dmp', '.tsv')
    sed_cline = ['sed', 's/\t|//g', file]
    with open(out_path, 'w') as out_handle:
        process = subprocess.Popen(sed_cline, stdout = out_handle)
        if wait:
            process.wait()
            if verbose:
                print('Finished')
    return

#%% classes
class TaxFetcher():
    def __init__(self, out_dir):
        self.initFTP()
        self.out_dir = out_dir
        check_outdir(out_dir)
        self.dl_status = {'taxdmp.zip':(1,),
                          'nucl_gb.accession2taxid.gz':(1,),
                          'nucl_wgs.accession2taxid.gz':(1,)}
        self.dmp = False
    
    def initFTP(self):
        # reconnect to the NCBI FTP server
        self.ftp = FTP('ftp.ncbi.nlm.nih.gov')
        self.ftp.login()
    
    def retrieve(self, file, out_file, tries = 3, verbose = False):
        # retrieve the given file and store it into out_file
        # tries sepcifies number of attempts before quitting
        # return 0 if dl succesful, 1 otherwise
        if verbose:
            print(f'Downloading {out_file}')
        out_path = f'{self.out_dir}/{out_file}'
        t = 0
        while t < tries:
            try:
                self.ftp.retrbinary(f'RETR {file}', open(out_path, 'wb').write)
                return 0, f'Succesfully retrieved {file}'
            except:
                t += 1
        # Failed to download, delete failed file and write waring
        if t == tries:
            check_failed_file(out_path)
            return 1, f'Failed to download {file} after {tries} tries.'
    
    def download(self, files = [taxdmp, gb_acc2tax, wgs_acc2tax], tries = 3, verbose = False):
        # schedule download of multiple files (all 3 by default)
        for file in files:
            out_file = file.split('/')[-1]
            status, msg = self.retrieve(file, out_file, tries, verbose)
            self.dl_status[out_file] = (status, msg, file)
    
    def extract(self, files, wait = True, verbose = False):
        # decompress the downloaded files
        for file in files:
            status = self.dl_status[file][0]
            if status == 0:
                if file.split('.')[0] == 'taxdmp':
                    extract_taxdmp(f'{self.out_dir}/{file}', self.out_dir, verbose)
                    self.dmp = True
                else:
                    extract_acc2tax(f'{self.out_dir}/{file}', wait, verbose)
    
    def fetch(self, wait = False, verbose = True):
        # direct download and processing of all taxonomy files
        self.download(verbose = verbose)
        self.extract(self.dl_status.keys(), wait, verbose)
        if self.dmp:
            dmp2tsv(f'{self.out_dir}/names.dmp', wait, verbose)
            dmp2tsv(f'{self.out_dir}/nodes.dmp', wait, verbose)
    
    def check(self):
        # check if all 4 necesary files were generated correctly
        status = pd.Series(data = 'No', index = ['names.tsv', 'nodes.tsv', 'nucl_gb.accession2taxid', 'nucl_wgs.accession2taxid'])
        for file in status.index:
            if os.path.isfile(f'{self.out_dir}/{file}'):
                status.at[file] = 'Yes'
        return status

    def retry_taxdmp(self, dl = True, ext = True, proc = True, wait = True, verbose = True):
        # retry generation of taxdmp files
        # dl, ext and proc -> set false if some part of the process can be skipped
        self.dmp = False
        if dl:
            self.download([taxdmp], verbose)
        if ext and self.dl_status['taxdmp.zip'][0] == 0:
            self.extract(['taxdmp.zip'], wait, verbose)
        if proc and self.dmp:
            dmp2tsv(f'{self.out_dir}/names.dmp', wait, verbose)
            dmp2tsv(f'{self.out_dir}/nodes.dmp', wait, verbose)
    
    def retry_acc2taxid(self, lab, dl = True, ext = True, wait = True, verbose = True):
        # retry generation of acc2taxid files
        # dl and ext -> set false if some part of the process can be skipped
        file_dl = f'pub/taxonomy/accession2taxid/nucl_{lab}.accession2taxid.gz'
        file = file_dl.split('/')[-1]
        if dl:
            self.download([file_dl], verbose)
        if ext and self.dl_status[file][0] == 0:
            extract_acc2tax(f'{self.out_dir}/{file}', wait, verbose)