#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:33:36 2021

@author: hernan
This script generates accession list tables from the summary files generated by db_survey
"""

#%% libraries
from glob import glob
import os
import pandas as pd

#%% functions
# file managing
def build_summ_tab(summ_dir):
    # generate information table for the summary files
    # get taxon, marker, database and path information for each file
    summ_files = glob(f'{summ_dir}/*summ')
    summ_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'File'])
    
    for file in summ_files:
        split_file = file.split('/')[-1].split('.summ')[0].split('_')
        row = {'Taxon':split_file[0],
               'Marker':split_file[1],
               'Database':split_file[2],
               'File':file}
        summ_tab = summ_tab.append(row, ignore_index=True)
    return summ_tab

def generate_filename(taxon, marker): # delete
    filename = f'acc_{taxon}_{marker}.tab'
    return filename

# data loading
def read_BOLD_summ(summ_file):
    # extract a list of accessions from a BOLD summary
    bold_tab = pd.read_csv(summ_file, sep = '\t', encoding = 'latin-1', dtype = str) # latin-1 to parse BOLD files
    accs = bold_tab['sampleid'].tolist()
    return accs

def read_NCBI_summ(summ_file):
    # extract a list of accessions from an NCBI or ENA summary
    ncbi_tab = pd.read_csv(summ_file, sep = '\t')
    accs = ncbi_tab.iloc[:,0].tolist()
    return accs

def read_summ(summ_file, read_func): # delete
    # extract accessions (and generate accessions w/o version number) from summary file
    # read_func specify what function will be used to read the file (read_BOLD_summ or read_NCBI_summ)
    accs = read_func(summ_file)
    
    shortaccs = get_shortaccs(accs)
    acc_series = pd.Series(accs, index = shortaccs, name = 'Accession')
    return acc_series

# data processing
def get_shortaccs(acclist): # delete
    # remove version number from each accession in acclist
    shortaccs = [acc.split('.')[0] for acc in acclist]
    return shortaccs

def make_acc_subtab(acc_series, dbase, tax, mark = ''): # delete
    # add taxon, marker and database information to the accession series
    acc_subtab = acc_series.to_frame()
    acc_subtab['Database'] = dbase
    acc_subtab['Taxon'] = tax
    acc_subtab['Marker'] = mark
    
    return acc_subtab

def make_dbase_series(tab, dbase): # delete
    # generate accession series for the given dbase
    # tab is a subtab generated from grouping the complete summary tab by taxon and marker, should contain a single entry per database
    
    # select the read_func to use
    if dbase == 'BOLD':
        read_func = read_BOLD_summ
    else:
        read_func = read_NCBI_summ
    
    # select the summary file and generate accsession series for it
    summ_file = tab.loc[tab['Database'] == dbase, 'File'].values[0]
    acc_series = read_summ(summ_file, read_func)
    return acc_series

def merge_subtabs(subtabs): # delete
    # merge the given subtabs into one, mind repeated entries
    # order in which subtabs are presented determines priority for repeated entries. When an entry is repeated between two subtabs, the first one to appear is kept
    # list should start with NCBI
    acc_tab = pd.DataFrame(columns = ['Accession', 'Database', 'Taxon', 'Marker'])
    
    for subtab in subtabs:
        acc_idx = set(acc_tab.index)
        st_idx = set(subtab.index)
        
        # only add entries not already present in acc_tab
        diff = st_idx.difference(acc_idx)
        to_add = subtab.loc[diff]
        
        acc_tab = acc_tab.append(to_add)
    return acc_tab

def make_acc_tab(summ_tab, tax, mark): # delete
    # generate a full accession table for the given tax/mark combo
    subtabs = []
    for dbase in ['NCBI', 'BOLD', 'ENA']:
        # generate subtab for each database (if present)
        if dbase in summ_tab['Database'].values:
            dbase_series = make_dbase_series(summ_tab, dbase)
            subtabs.append(make_acc_subtab(dbase_series, dbase, tax, mark))
    merged = merge_subtabs(subtabs)
    return merged

def acc_list(summ_dir, out_dir): # delete
    # Main function, run to generate accession list files
    # TODO, compare with previous databases, check version changes
    summ_tab = build_summ_tab(summ_dir)
    
    for tax, tax_group in summ_tab.groupby('Taxon'):
        for mark, mark_group in tax_group.groupby('Marker'):
            acc_tab = make_acc_tab(mark_group, tax, mark)
            out_file = generate_filename(tax, mark)
            acc_tab.to_csv(f'{out_dir}/{out_file}')

#%% classes
class Lister():
    def __init__(self, taxon, marker, dbase, summ_file, read_func):
        self.taxon = taxon
        self.marker = marker
        self.dbase = dbase
        self.accs = read_func(summ_file)
        self.out_tab = pd.DataFrame()
    
    def get_shortaccs_ver(self):
        splitaccs = [acc.split('.') for acc in self.accs]
        shortaccs = [acc[0] for acc in splitaccs]
        vers = [acc[-1] for acc in splitaccs]
        return shortaccs, vers

    def make_acc_subtab(self):
        shortaccs, vers = self.get_shortaccs_ver()
        acc_subtab = pd.DataFrame({'Accession': self.accs, 'Version':vers}, index = shortaccs)
        acc_subtab['Entry'] = 1

        self.acc_subtab = acc_subtab
        self.out_tab = acc_subtab.copy()
    
    def generate_filename(self, out_dir):
        filename = f'{out_dir}/{self.taxon}_{self.marker}_{self.dbase}.acc'
        return filename
    
    def get_old_tab(self, old_dir):
        old_file_path = f'{old_dir}/{self.taxon}_{self.marker}_{self.dbase}.acc'
        if os.path.isfile(old_file_path):
            self.old_subtab = pd.read_csv(old_file_path)
            return True
        return False

    def compare_tab(self, old_dir):
        if self.get_old_tab(old_dir):
            long_accs = self.acc_subtab['Accession']
            self_vers = self.acc_subtab['Version']
            old_vers = self.old_subtab['Version']
            mixed_df = pd.DataFrame[{'Accession':long_accs, 'Version':self_vers, 'old':old_vers}].fillna(0)
            mixed_df['Entry'] = 0 # do nothing
            # new entries
            mixed_df.at[mixed_df['old'] == 0, 'Entry'] = 1 # new entry
            mixed_df.at[(mixed_df['old'] > 0) & (mixed_df['old'] != mixed_df['Version']), 'Entry'] = 2 # new version
            mixed_df.at[mixed_df['Version'] == 0] = -1 # missing entry
            self.out_tab = mixed_df[['Accession', 'Version', 'Entry']]
    
    def store_tab(self, out_dir):
        out_file = self.generate_filename(out_dir)
        self.out_tab.to_csv(out_file)

def main(summ_dir, list_dir, old_dir = None):
    summ_tab = build_summ_tab(summ_dir)

    for idx, row in summ_tab.iterrows():
        tax, mark, dbase, file = row['Taxon'], row['Marker'], row['Database'], row['File']
        read_func = read_NCBI_summ
        if dbase == 'BOLD':
            read_func = read_BOLD_summ
        lst = Lister(tax, mark, dbase, file, read_func)
        lst.make_acc_subtab()
        lst.compare_tab(old_dir)
        lst.store_tab(list_dir)