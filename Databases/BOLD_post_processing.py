#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:18:26 2021

@author: hernan
Process BOLD data, separate sequences from different markers, recover genbank accessions (when present)
"""

#%% libraries
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#%% functions
def make_BOLD_seqdict(bold_file):
    seqdict = {}
    with open(bold_file, 'r') as handle:
        for title, seq in sfp(handle):
            seqdict[title] = seq
    return seqdict

def get_mark(header):
    split_header = header.split('|')
    mark = split_header[2]
    mark = mark.split('-')[0]
    return mark

def has_alt_acc(header):
    split_header = header.split('|')
    if len(split_header) > 3:
        return True
    return False

def get_acc(header):
    split_header = header.split('|')
    acc = split_header[0].replace('-', '.')
    return acc

def get_alt_acc(header):
    split_header = header.split('|')
    return split_header[3]

#%%
seqdict = make_BOLD_seqdict('Nematoda_COI_BOLD.tmp')

mark_dict = {'18S':[], 'COI':[]}

for header in seqdict.keys():
    mark = get_mark(header)
    if mark in mark_dict.keys():
        mark_dict[mark].append(header)

for mark, headerlist in mark_dict.items():
    outfile = f'Nematoda_{mark}_BOLD2.tmp'
    mark_records = []
    for header in headerlist:
        if has_alt_acc(header):
            acc = get_alt_acc(header)
        else:
            acc = get_acc(header)
        
        record = SeqRecord(Seq(seqdict[header]), id = acc, name = '', description = header)
        mark_records.append(record)
    
    with open(outfile, 'w') as out_handle:
        SeqIO.write(mark_records, out_handle, 'fasta')