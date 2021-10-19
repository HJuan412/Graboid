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
def build_BOLD_seqdict(bold_file):
    # builds a {header:seq} dictionary from the given file
    seqdict = {}
    with open(bold_file, 'r') as handle:
        for title, seq in sfp(handle):
            seqdict[title] = seq
    return seqdict

def build_mark_dict(markers):
    # builds a {marker:[]} dictionary, will store the headers for each marker
    mark_dict = {}
    for mark in markers:
        mark_dict[mark] = []
    return mark_dict

def get_mark(header):
    # get the sequence's marker from its header (third field in header)
    split_header = header.split('|')
    mark = split_header[2]
    mark = mark.split('-')[0]
    return mark

def has_alt_acc(header):
    # check if the sequence has an alternate accession (fourth field in header)
    split_header = header.split('|')
    if len(split_header) > 3:
        return True
    return False

def get_acc(header):
    # get the sequence accession and fix version number
    split_header = header.split('|')
    acc = split_header[0].replace('-', '.')
    return acc

def get_alt_acc(header):
    # get the alternative accession from the header (fourth field)
    split_header = header.split('|')
    return split_header[3]

def get_record_acc(header):
    if has_alt_acc(header):
        acc = get_alt_acc(header)
    else:
        acc = get_acc(header)
    return acc

def locate_BOLD_files(seq_dir):
    files = glob(f'{seq_dir}/*BOLDr*tmp')
    return files

#%% classes
class Processor():
    def __init__(self, bold_file, markers = ['COI', '18S']):
        self.bold_file = bold_file
        self.seqs = build_BOLD_seqdict(bold_file)
        self.markers = markers
        self.mark_dict = build_mark_dict(markers)
        self.split_seqs()
    
    def split_seqs(self):
        # distribute sequence headers by marker
        for header in self.seqs.keys():
            mark = get_mark(header)
            if mark in self.markers:
                self.mark_dict[mark].append(header)
    
    def generate_filename(self, marker):
        # generate filename for the processed file
        split_file = self.bold_file.split('/')
        out_dir = '/'.join(split_file[:-1])
        tax = split_file[-1].split('_')[0]
        outfile = f'{out_dir}/{tax}_{marker}_BOLD.tmp' # BOLD means it's been processed, use to differentiate from raw BOLD files (marked as BOLDr)
        return outfile
    
    def get_mark_records(self, headerlist):
        # get all sequences belonging to a given marker (also removes gaps)
        records = []
        for header in headerlist:
            acc = get_record_acc(header)
            seq = self.seqs[header].replace('-', '')
            record = SeqRecord(Seq(seq), id = acc, name = '', description = header)
            records.append(record)
        return records
    
    def process(self):
        # generate processed files for all markers
        for mark, headerlist in self.mark_dict.items():
            mark_records = self.get_mark_records(headerlist)
            mark_outfile = self.generate_filename(mark)
            with open(mark_outfile, 'w') as out_handle:
                SeqIO.write(mark_records, out_handle, 'fasta')
#%% main
def process_files(seq_dir, markers = ['COI', '18S']):
    bold_files = locate_BOLD_files(seq_dir)
    
    for file in bold_files:
        proc = Processor(file, markers)
        proc.process()
    return
