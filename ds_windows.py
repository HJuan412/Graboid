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
import numpy as np
import os
import pandas as pd
import toolkit as tools

#%% functions
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
            match_dict[curr_qid] = coords
            
            idx_0 = idx_1
            curr_qid = qid
    return match_dict


#%% WB
class Match():
    def __init__(self, seqid, coords, start, end):
        self.seqid = seqid
        self.coords = self.crop(coords, start, end)
        self.flip()
        self.start = start
        self.end = end
        self.len = end - start
    
    def crop(self, coords, start, end):
        # Clips coordinates to include only the portions of the alignments that overlap with the window.
        cropped = coords.copy() # cropped match
        
        to_crop_left = np.where(cropped[:,0] < start)
        to_crop_right = np.where(cropped[:,1] > end)
        
        # # crop left
        # for i in to_crop_left:
        #     diff = start - cropped[i, 0]
        #     cropped[i, [0, 2]] += diff
        # # crop right
        # for j in to_crop_right:
        #     diff = end - cropped[i, 1]
        #     cropped[j, [1, 3]] += diff
        
        for to_crop, idx, position in zip([to_crop_left, to_crop_right], [0, 1], [start, end]):
            for i in to_crop:
                diff = position - cropped[i, idx]
                cropped[i, idx] += diff
                cropped[i, idx + 2] += diff

        return cropped
    
    def flip(self):
        # flips matches in reverse sense, tracks sense of each match
        self.orient = np.ones(len(self.coords), dtype = int)
        to_flip = np.where(self.coords[:,0]>self.coords[:,1])[0]
        self.orient[to_flip] = -1
        for i in to_flip:
            row = self.coords[i]
            row[0], row[1] = row[1], row[0]
            # row[2], row[3] = row[3], row[2]
            self.coords[i] = row
    
    def build_alignment(self, seq, gap_char = '-'):
        # Uses the cropped and flipped coordinates to build the window alignment
        alig = [gap_char] * self.len
        for co, ori in zip(self.coords, self.orient):
            a_start = co[0] - self.start
            a_end = co[1] - self.start
            b_start = co[2]
            b_end = co[3]
            alig[a_start:a_end] = seq[b_start:b_end][::ori] # ori is used to flip the query match
        return ''.join(alig)
        
        
class Window():
    def __init__(self, start, end, matches, seqs):
        self.start = start
        self.end = end
        self.get_in_window(matches)
        self.build_alignment(seqs)
    
    def get_in_window(self, matches):
        # extracts all the matches overlapping with the window (excludes fragments that don't overlap)
        self.matches = {}
        
        for seqid, match in matches.items():
            matches_in = np.where((match[:,0] < self.end) & (match[:,1] > self.start))
            miw = match[matches_in]
            if len(miw) > 0:
                self.matches[seqid] = Match(seqid, miw, self.start, self.end)

    def build_alignment(self, seqs):
        self.aln_dict = {}
        
        for seqid, match in self.matches.items():
            seq = seqs[seqid]
            # alig, sum_gaps = match.build_alignment(seq)
            # self.aln_dict[seqid] = alig, sum_gaps
            alig= match.build_alignment(seq)
            self.aln_dict[seqid] = alig
    
    def store_alignment(self, prefix = 'alignment'):
        records = []
        for acc, seq in self.aln_dict.items():            
            records.append(SeqRecord(Seq(seq), id = acc, name='', description = ''))

        alignment = msa(records)
        filename = f'{prefix}_({self.start}-{self.end}_n{len(records)}).fasta'
        with open(filename, 'w') as handle:
            AlignIO.write(alignment, handle, 'fasta')
        return

# TODO what if I build a giant alignment array and extract windows from there?
class WindowBuilder():
    def __init__(self, in_file, seq_file):
        self.report = load_report(in_file)
        self.matches = build_matchdict(self.report)
        self.seqs = tools.make_seqdict(seq_file)
        self.get_match_bounds()
    
    def get_match_bounds(self):
        self.lower = self.report['sstart'].min() - 1
        self.upper = self.report['send'].max() - 1
    
    def build_window(self, start, end):
        if self.lower <= start < end <= self.upper:
            return Window(start, end, self.matches, self.seqs)
        else:
            print(f'Invalid.\tWindow bounds: start = {start}, end = {end}\n\tMust be within match bounds: lower = {self.lower}, upper = {self.upper}')
            return
#%%
wb = WindowBuilder('Dataset/test2/Nematoda_18S.tab', 'Databases/13_10_2021-20_15_58/Sequence_files/Nematoda_18S.fasta')

width = 100
step = 15
out_dir = 'Dataset/test2'
windows = np.arange(wb.lower, wb.upper + 1 - width, step)

aln_window = wb.build_window(wb.lower, wb.upper)

#%%
accs = list(aln_window.aln_dict.keys())
aln_len = len(aln_window.aln_dict[accs[0]])

cov = np.zeros(aln_len, int)
for i in range(aln_len):
    for seq in aln_window.aln_dict.values():
        if seq[i] != '-':
            cov[i] += 1
#%%
def make_window(aln_dict, start, end, gap_thresh = 0.2):
    length = end-start
    max_gaps = length * gap_thresh
    window_dict = {}
    
    for k,v in aln_dict.items():
        seq = v[start:end]
        if seq.count('-') < max_gaps:
            window_dict[k] = seq
    
    return window_dict

#%%
import matplotlib.pyplot as plt
#%%
x = np.arange(len(windows))
y = []

thresh = 0.01
for wstart in windows:
    wd = make_window(aln_window.aln_dict, wstart, wstart + width, thresh)
    y.append(len(wd))

plt.plot(x, y)
#%%
def process(in_dir, taxons = ['Nematoda', 'Platyhelminthes'], markers = ['18S', '28S', 'COI'], width = 100, step = 15):
    sequence_dir = f'Databases/{in_dir}/Sequence_files'
    report_dir = f'Dataset/{in_dir}/BLAST_reports'
    
    out_dir = f'Dataset/{in_dir}/Windows'
    os.mkdir(out_dir)
    for tax in taxons:
        tax_dir = f'{out_dir}/{tax}'
        os.mkdir(tax_dir)
        for mark in markers:
            print(f'Processing {tax} {mark}')
            mark_dir = f'{tax_dir}/{mark}'
            os.mkdir(mark_dir)
            
            seq_file = f'{sequence_dir}/{tax}_{mark}.fasta'
            rep_file = f'{report_dir}/{tax}_{mark}.tab'
            wn = Windower(rep_file, seq_file)
            reflen = wn.get_max_match()
            
            windows = np.arange(0, reflen + 1 - width, step)
            
            for idx, w in enumerate(windows):
                print(f'Window {idx} of {len(windows)}')
                start = w
                end = w + width
                wn.update_window(start, end)
                wn.make_alignment()
                wn.aln_report()
                wn.store_alignment(f'{mark_dir}/{tax}_{mark}')
            
            
#%% classes
class Windower():
    def __init__(self, report_file, seqfile):
        report = load_report(report_file)
        self.matches = build_matchdict(report)
        self.seqs = tools.make_seqdict(seqfile)
        self.__w_start = 0
        self.__w_end = 0
    
    @property
    def w_start(self):
        return self.__w_start
    
    @w_start.setter
    def w_start(self, start):
        if type(start) is int:
            if start >= 0:
                self.__w_start = start
    
    @property
    def w_end(self):
        return self.__w_end
    
    @w_end.setter
    def w_end(self, end):
        if type(end) is int:
            if end >= self.w_start:
                self.__w_end = end
    
    def get_max_match(self):
        # use this to calculate the length of the reference sequence
        max_match = 0
        
        for match in self.matches.values():
            match_max = match[:,1].max()
            if match_max > max_match:
                max_match = match_max
        
        return max_match

    def update_window(self, start, end):
        # Redefine window boundaries
        # make sure bothe values are over 0 and start < end
        if 0 <= start < end:
            self.w_start = int(start)
            self.w_end = int(end)

    def make_alignment(self):
        # Generate alignment on the current window
        self.aln_dict = {}
        for seqid, match in self.matches.items():
            seq = self.seqs[seqid]
            in_window = match_in_window(match, self.w_start, self.w_end) # get all sequence matches overlapping with the window
            
            if len(in_window) == 0:
                # sequence doesn't overlap with window
                continue
            cropped = crop_match(in_window, self.w_start, self.w_end) # crop non-overlapping segments
            gaps = get_gaps(cropped) # calculate gaps
            self.aln_dict[seqid] = align_match(seq, cropped, gaps[0]) # generate alignment
    
    def aln_report(self):
        total_accs = list(self.aln_dict.keys())
        window_len = self.w_end - self.w_start

        aln_df = pd.DataFrame(index = range(len(total_accs)), columns = ['acc', 'seqlen', 'gaplen', 'seq'])

        for idx, acc in enumerate(total_accs):
            seq, gaps = self.aln_dict[acc]
            seqlen = len(seq)
            aln_df.at[idx] = [acc, seqlen, gaps, seq]
        
        self.normal = aln_df.loc[aln_df['seqlen'] == window_len]
        self.anormal = aln_df.loc[aln_df['seqlen'] != window_len]
    
    def store_alignment(self, prefix = 'alignment', max_gap = 30):
        filtered = self.normal.loc[self.normal['gaplen'] <= max_gap]
        records = []
        for idx, row in filtered.iterrows():
            acc = row['acc']
            seq = row['seq']
            
            records.append(SeqRecord(Seq(seq), id = acc, name='', description = ''))

        alignment = msa(records)
        filename = f'{prefix}_({self.w_start}-{self.w_end}_n{len(records)}).fasta'
        with open(filename, 'w') as handle:
            AlignIO.write(alignment, handle, 'fasta')
        return