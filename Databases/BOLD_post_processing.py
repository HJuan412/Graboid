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

from glob import glob
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

def generate_outfile(bold_file, marker):
    tax = bold_file.split('_')[0]
    outfile = f'{tax}_{marker}_BOLD.tmp'
    return outfile

def locate_BOLD_files(seq_dir):
    files = glob(f'{seq_dir}/*BOLD*tmp')
    return files
#%%
def process_file(bold_file, markers = ['COI', '18S']):
    seqdict = make_BOLD_seqdict(bold_file)
    
    mark_dict = {}
    for mark in markers:
        mark_dict[mark] = []
    
    for header in seqdict.keys():
        mark = get_mark(header)
        if mark in mark_dict.keys():
            mark_dict[mark].append(header)
    
    for mark, headerlist in mark_dict.items():
        mark_records = []
        for header in headerlist:
            if has_alt_acc(header):
                acc = get_alt_acc(header)
            else:
                acc = get_acc(header)
            
            record = SeqRecord(Seq(seqdict[header]), id = acc, name = '', description = header)
            mark_records.append(record)

        outfile = generate_outfile(bold_file, mark)
        with open(outfile, 'w') as out_handle:
            SeqIO.write(mark_records, out_handle, 'fasta')
    return