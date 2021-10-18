#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from Bio.SeqRecord import SeqRecord
from glob import glob
import pandas as pd
#%% functions
def make_filetab(seq_dir):
    files = glob(f'{seq_dir}/*tmp')
    filetab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'File'])
    
    for idx, file in enumerate(files):
        split_file = file.split('/')[-1].split('.')[0].split('_')
        tax, mark, dbase = split_file
        
        if mark != '':
            filetab.at[idx] = [tax, mark, dbase, file]
    return filetab

def make_seqdict(seqfile):
    seqdict = {}
    with open(seqfile, 'r') as handle:
        for header, seq in sfp(handle):
            acc = header.split(' ')[0]
            acc_short = acc.split('.')[0]
            seqdict[acc_short] = (acc, header, seq)
    return seqdict

def build_infotab(seqdict, dbase):
    tab = pd.DataFrame.from_dict(seqdict, orient='index', columns = ['Version', '1', '2'])
    info_tab = tab['Version'].to_frame()
    info_tab['Database'] = dbase
    return info_tab

# GONE GONE GONE (Merger methods now)
# def ncbi_vs_bold(ncbi_seqdict, bold_seqdict):
#     ncbi_tab = build_infotab(ncbi_seqdict, 'NCBI')
#     bold_tab = build_infotab(bold_seqdict, 'BOLD')
    
#     bold_accs = set(bold_seqdict.keys()).difference(ncbi_seqdict.keys())
#     bold_subtab = bold_tab.loc[bold_accs]
#     combined_tab = pd.concat([ncbi_tab, bold_subtab])
#     return combined_tab

# def merge_seqfiles(infotab, seqdicts, outfile):
#     records = []
#     for dbase, subtab in infotab.groupby('Database'):
#         seqdict = seqdicts[dbase]
#         for acc in subtab.index:
#             acc, header, seq = seqdict[acc]
#             record = SeqRecord(Seq(seq), header, description = '')
#             records.append(record)
#     with open(outfile, 'w') as out_handle:
#         SeqIO.write(records, out_handle, 'fasta')
#     return
#%% classes
class Merger():
    def __init__(self, taxon, marker, out_dir):
        self.ncbi_dict = {}
        self.bold_dict = {}
        self.ncbi_tab = build_infotab({}, 'NCBI')
        self.bold_tab = build_infotab({}, 'BOLD')
        self.merged_tab = pd.DataFrame(columns = ['Version', 'Database'])
        self.out_file = f'{out_dir}/{taxon}_{marker}'
    
    def add_seqdict(self, seqfile, dbase):
        if dbase == 'NCBI':
            self.ncbi_dict = make_seqdict(seqfile)
            self.ncbi_tab = build_infotab(self.ncbi_dict, dbase)
        if dbase == 'BOLD':
            self.bold_dict = make_seqdict(seqfile)
            self.bold_tab = build_infotab(self.bold_dict, dbase)
        return
    
    def clear_seqdict(self, dbase):
        if dbase == 'NCBI':
            self.ncbi_dict = {}
            self.ncbi_tab = build_infotab({}, dbase)
        if dbase == 'BOLD':
            self.bold_dict = {}
            self.bold_tab = build_infotab({}, dbase)
        
        self.merged_tab = self.merged_tab.iloc[0:0]
        return

    def merge_infotabs(self):
        bold_accs = set(self.bold_dict.keys()).difference(self.ncbi_dict.keys())
        bold_subtab = self.bold_tab.loc[bold_accs]
        self.merged_tab = pd.concat([self.ncbi_tab, bold_subtab])
        return
    
    def save_merged_tab(self):
        out_file = self.out_file + '_info.tab'
        self.merged_tab.to_csv(out_file)

    def merge_seqfiles(self):
        ncbi_accs = self.merged_tab.loc[self.merged_tab['Database'] == 'NCBI'].index
        bold_accs = self.merged_tab.loc[self.merged_tab['Database'] == 'BOLD'].index
        
        records = []
        for db_accs, db_dict in zip([ncbi_accs, bold_accs], [self.ncbi_dict, self.bold_dict]):
            for db_acc in db_accs:
                acc, header, seq = db_dict[db_acc]
                record = SeqRecord(Seq(seq), header, description = '')
                records.append(record)
        
        out_file = self.out_file + '.fasta'
        with open(out_file, 'w') as out_handle:
            SeqIO.write(records, out_handle, 'fasta')
#%%
def merger(seq_dir):
    filetab = make_filetab(seq_dir)
    
    for tax, sub_tab0 in filetab.groupby('Taxon'):
        for mark, sub_tab1 in sub_tab0.groupby('Marker'):
            merge_agent = Merger(tax, mark, seq_dir)
            for _, row in sub_tab1.iterrows():
                dbase = row['Database']
                file = row['File']                
                merge_agent.add_seqdict(file, dbase)
            
            merge_agent.merge_infotabs()
            merge_agent.save_merged_tab()
            merge_agent.merge_seqfiles()    
    return