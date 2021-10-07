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

def ncbi_vs_bold(ncbi_seqdict, bold_seqdict):
    ncbi_tab = build_infotab(ncbi_seqdict, 'NCBI')
    bold_tab = build_infotab(bold_seqdict, 'BOLD')
    
    bold_accs = set(bold_seqdict.keys()).difference(ncbi_seqdict.keys())
    bold_subtab = bold_tab.loc[bold_accs]
    combined_tab = pd.concat([ncbi_tab, bold_subtab])
    return combined_tab

def merge_seqfiles(infotab, seqdicts, outfile):
    records = []
    for dbase, subtab in infotab.groupby('Database'):
        seqdict = seqdicts[dbase]
        for acc in subtab.index:
            acc, header, seq = seqdict[acc]
            record = SeqRecord(Seq(seq), header, description = '')
            records.append(record)
    with open(outfile, 'w') as out_handle:
        SeqIO.write(records, out_handle, 'fasta')
    return

def select_entries(seq_dir):
    filetab = make_filetab(seq_dir)
    
    for tax, sub_tab0 in filetab.groupby('Taxon'):
        for mark, sub_tab1 in sub_tab0.groupby('Marker'):
            seqdicts = {'NCBI':{}, 'BOLD':{}}
            for _, row in sub_tab1.iterrows():
                dbase = row['Database']
                file = row['File']
                seqdicts[dbase] = make_seqdict(file)
            infotab = ncbi_vs_bold(seqdicts['NCBI'], seqdicts['BOLD'])
            tab_filename = f'{tax}_{mark}_info.tab'
            infotab.to_csv(tab_filename)
            merged_filename = f'{tax}_{mark}.fasta'
            merge_seqfiles(infotab, seqdicts, merged_filename)
    
    return

# TODO remove tempfiles
# TODO integrate