#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:11:34 2021

@author: hernan

Extracts and aligns sequences from blast report
"""
#%% libraries
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment as msa
from ds_blast import build_seq_tab
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import toolkit as tools

#%% functions
def build_repseq_tab(blast_dir, seq_dir):
    seq_tab = build_seq_tab(seq_dir)
    seq_tab['Tab'] = 'seq'

    rep_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'File'])
    blast_reps = glob(f'{blast_dir}/*tab')
    for idx, blast_rep in enumerate(blast_reps):
        split_file = blast_rep.split('/')[-1].split('.tab')[0].split('_')
        taxon = split_file[0]
        marker = split_file[1]
        rep_tab.at[idx] = [taxon, marker, blast_rep]
    
    rep_tab['Tab'] = 'rep'
    
    repseq_tab = pd.concat((rep_tab, seq_tab))
    
    return repseq_tab

def load_report(report_file):
    """
    Load the ungapped BLAST report as a pandas.DataFrame

    Parameters
    ----------
    report_file : str
        Path to the report file.

    Returns
    -------
    report_tab : pandas.DataFrame

    """
    report_tab = pd.read_csv(report_file, sep = '\t', header = None, names = ['qseqid', 'pident', 'length', 'qstart', 'qend', 'sstart', 'send', 'evalue'])
    report_tab.sort_values(['qseqid', 'sstart'], inplace = True)
    return report_tab

def build_matchdict(report):
    # version 5, 1100 times faster than the original
    """
    Generate a dictionary with the coordinates array of all matches in the blast report

    Parameters
    ----------
    report : pandas.DataFrame
        BLAST report to base alignment on

    Returns
    -------
    matchdict : dict
        Dictionary of the form query_id:coord_array.

    """
    match_dict = {}
    qid_array = report['qseqid'].tolist() + [''] # tail makes sure last id is included
    coord_array = report[['sstart', 'send', 'qstart', 'qend']].to_numpy(dtype = int) - 1
    
    curr_qid = qid_array[0]
    idx_0 = 0
    
    for idx_1, qid in enumerate(qid_array):
        if qid != curr_qid:
            coords = coord_array[idx_0:idx_1, :]
            # match_dict[curr_qid] = coords
            match_dict[curr_qid] = Match(curr_qid, coords)
            
            idx_0 = idx_1
            curr_qid = qid
    return match_dict


#%% classes
class Match():
    def __init__(self, seqid, coords):
        self.seqid = seqid
        self.coords = coords
        self.orient = np.ones(len(self.coords), dtype = int)
        self.flip()
    
    def flip(self):
        # flips matches in reverse sense, tracks sense of each match in self.orient
        to_flip = np.where(self.coords[:,0]>self.coords[:,1])[0]
        self.orient[to_flip] = -1
        for i in to_flip:
            row = self.coords[i]
            row[0], row[1] = row[1], row[0]
            self.coords[i] = row

    def build_alignment(self, seq, aln_len, gap_char = '-'):
        # Uses the cropped and flipped coordinates to build the window alignment
        # seq is the sequence for the Matche's coordinates
        # aln_len is the length of the alignment
        # gap_char is the character used to designate gaps
        # returns alignment as a string
        alig = [gap_char] * aln_len
        for co, ori in zip(self.coords, self.orient):
            a_start = co[0]
            a_end = co[1]
            b_start = co[2]
            b_end = co[3]
            alig[a_start:a_end] = seq[b_start:b_end][::ori] # ori is used to flip the query match
        return ''.join(alig)

class Window():
    def __init__(self, aln_dict, start, end, gap_thresh = 0.2):
        self.start = start
        self.end = end
        self.length = end-start
        self.build_window(aln_dict, gap_thresh)
    
    def build_window(self, aln_dict, gap_thresh = 0.1):
        # Extracts the corresponding bases from each sequence in the alignment
        # only includes them in the window if they are below the specified gap threshold
        max_gaps = self.length * gap_thresh
        self.window_dict = {} # will store the extractes subsequences
        
        for k,v in aln_dict.items():
            seq = v[self.start:self.end] # subsequence
            # filter by number of gaps
            if seq.count('-') < max_gaps:
                self.window_dict[k] = seq

        self.n = len(self.window_dict) # count the number of sequences in the window

    def store_alignment(self, prefix = 'alignment'):
        # save the window to file
        if self.n == 0:
            return
        records = []
        for acc, seq in self.window_dict.items():            
            records.append(SeqRecord(Seq(seq), id = acc, name='', description = ''))

        alignment = msa(records)
        filename = f'{prefix}_{self.start + 1}-{self.end}_n{self.n}.fasta' # filename gives window boundaries and amount of seqs
        with open(filename, 'w') as handle:
            AlignIO.write(alignment, handle, 'fasta')
        return

class WindowBuilder():
    # Handles window construction for a single alignment
    def __init__(self, in_file, seq_file, out_dir):
        self.report = load_report(in_file)
        self.matches = build_matchdict(self.report)
        self.seqs = tools.make_seqdict(seq_file)
        self.get_match_bounds()
        self.build_alignment()
        self.get_aln_coverage()
        self.out_dir = out_dir
    
    def get_match_bounds(self):
        self.lower = self.report['sstart'].min() - 1 # should always be 0
        self.upper = self.report['send'].max() - 1
        
    def build_alignment(self):
        # align all sequences using the blast coordinates
        self.aln_dict = {}
        
        for seqid, match in self.matches.items():
            seq = self.seqs[seqid]
            alig = match.build_alignment(seq, self.upper)
            self.aln_dict[seqid] = alig

    def get_aln_coverage(self):
        # count coverage for each base in the alignment
        coverage = np.zeros(self.upper, int)
        for seq in self.aln_dict.values():
            for i in range(self.upper):
                if seq[i] != '-':
                    coverage[i] += 1
        self.coverage = coverage

    def build_window(self, start, end, gap_thresh = 0.1, prefix = 'alignment', store = True):
        # create a window at the specified boundaries (start, end)
        if 0 <= start < end <= self.upper:
            # only proceed if given boundaries are valid
            window = Window(self.aln_dict, start, end, gap_thresh)
            if store:
                # save window to file
                window.store_alignment(f'{self.out_dir}/{prefix}')
            return window
        else:
            print(f'Invalid.\tWindow bounds: start = {start}, end = {end}\n\tMust be within match bounds: lower = {self.lower}, upper = {self.upper}')
            return
    
    def build_all(self, width, step, gap_thresh = 0.1, store = True):
        # create multiple windows along the entire length of the alignment
        last_win = self.upper - width # position of the last possible window
        windows = np.arange(0, last_win, step) # start position for all the windows
        # add a tail window if the last bases are missing
        if windows[-1] < last_win:
            windows = np.concatenate((windows, [last_win]))
        
        # register used variables
        self.width = width
        self.step = step
        self.windows = windows
        self.window_coverage = np.zeros(len(windows), int) # store the n of each generated window

        for idx, start in enumerate(windows):
            print(f'Building window {idx + 1} of {len(windows)}') # TODO generate report of built windows (maybe)
            end = start + width
            window = self.build_window(start, end, gap_thresh, f'{idx}_aln', store)
            self.window_coverage[idx] = window.n

    def plot_coverage(self, start = 0, end = np.inf):
        # plot the alignment coverage (per base and per window)
        # start and end allow to view a specified portion of the alignment
        # TODO: adjust bar width according to window step
        start = max(self.lower, start)
        end = min(self.upper, end)
        view = end - start
        ticklen = int(view/10)
        xticks = np.arange(0, view + 1, ticklen)
        xlabs = np.arange(start, end+1, ticklen)

        # generate plot information
        # per base coverage data
        cov = self.coverage[start:end]
        x = np.arange(len(cov))
        # per window coverage data
        w_cov_idx = np.where(np.logical_and(self.windows >= start, self.windows <= end))
        w_x = self.windows[w_cov_idx] - start + 5
        w_y = self.window_coverage[w_cov_idx]

        # plot
        fig, ax = plt.subplots(figsize = (12,7))
        ax.margins(x=0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs)
        ax.set_xlabel('Position')
        ax.set_ylabel('Coverage')
        ax.set_title(f'Coverage\nWindow width: {self.width}\nWindow step: {self.step}')

        ax.plot(x, cov, color = 'r', label = 'Per base coverage')
        ax.bar(w_x, w_y, width = 10, label = 'Per window coverage')
        ax.legend()
        return ax

class WindowDirector():
    # Handles window construction for multiple alignments
    def __init__(self, blast_dir, seq_dir, out_dir, warn_dir):
        self.blast_dir = blast_dir
        self.seq_dir = seq_dir
        self.repseq_tab = build_repseq_tab(blast_dir, seq_dir)
        self.out_dir = out_dir
        self.make_subdirs()
        self.warn_dir = warn_dir
        self.plot_dict = {}
    
    def make_subdirs(self):
        taxons = self.repseq_tab['Taxon'].unique()
        markers = self.repseq_tab['Marker'].unique()
        self.subdirs = []
        for tax in taxons:
            tax_dir = f'{self.out_dir}/{tax}' 
            if not os.path.isdir(tax_dir):
                os.mkdir(tax_dir)
            for mark in markers:
                mark_dir = f'{self.out_dir}/{tax}/{mark}'
                self.subdirs.append(mark_dir)
                if not os.path.isdir(mark_dir):
                    os.mkdir(mark_dir)

    def check_tabs(self):
        # make sure seq_tab is not empty
        warns = False
        with open(f'{self.warn_dir}/windows.warn', 'w') as warn_handle:
            for taxon, subtab0 in self.repseq_tab.groupby('Taxon'):
                for marker, subtab1 in subtab0.groupby('Marker'):
                    vals = subtab1['Tab'].value_counts()
                    if vals['rep'] == 0:
                        warns = True
                        warn_handle.write('No blast report for {taxon} - {marker} found in directory {self.blast_dir}')
                    if vals['seq'] == 0:
                        warns = True
                        warn_handle.write('No sequence file for {taxon} - {marker} found in directory {self.seq_dir}')
            if warns:
                return False
        return True
    
    def direct(self, width, step, gap_thresh = 0.1, store = True):
        if self.check_tabs():
            for taxon, subtab0 in self.repseq_tab.groupby('Taxon'):
                for marker, subtab1 in subtab0.groupby('Marker'):
                    print(f'Building windows for {taxon}, {marker}')
                    rep_file = subtab1.loc[subtab1['Tab'] == 'rep', 'File'].values[0]
                    seq_file = subtab1.loc[subtab1['Tab'] == 'seq', 'File'].values[0]
                    
                    out_dir = f'{self.out_dir}/{taxon}/{marker}'
                    wb = WindowBuilder(rep_file, seq_file, out_dir)
                    wb.build_all(width, step, gap_thresh, store)
                    
                    self.plot_dict[(taxon, marker)] = wb.plot_coverage()