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
from glob import glob
# TODO implement numba
from numba import njit
import numpy as np
import os
import pandas as pd
import toolkit as tools

#%% functions
def load_ref(file):
    with open(file, 'r') as handle:
        ref = handle.read().splitlines()[1:]
    ref = ''.join(ref)
    return ref

def load_report(report_file):
    report_tab = pd.read_csv(report_file, sep = '\t', header = None, names = ['qseqid', 'pident', 'length', 'qstart', 'qend', 'sstart', 'send', 'evalue'])
    report_tab.sort_values(['qseqid', 'sstart'], inplace = True)
    return report_tab

def match_in_window(match, w_start, w_end):
    """    
    Lists match records that overlap with the given window.

    Parameters
    ----------
    match : numpy.array
        Coordinate matrix of matches to the reference sequence (sstart, ssend, qstart, qend).
    w_start : int
        Start position for the window.
    w_end : int
        End position for the window.

    Returns
    -------
    miw : numpy.array
        Match records overlapping with window..
    """

    matches_in = np.where((match[:,0] < w_end) & (match[:,1] > w_start))
    miw = match[matches_in]
    miw = miw[miw[:,0].argsort()] # sort by first column
    return miw

def get_gaps(match):
    """
    Lists gaps in the subject alignment and the query alignment

    Parameters
    ----------
    match : numpy.array
        Coordinate matrix extracted from blast report (sstart, send, qstart, qend).

    Returns
    -------
    gaps : numpy.array
        Array of shape (2, # matches) with the lengths of the gaps between matches.
        First row is gaps between subject matches.
        Second row is gaps between query matches.

    """
    nsegs = match.shape[0]
    gaps = np.zeros((nsegs, 2), dtype = int)
    
    for i in range(1, nsegs):
        gaps[i-1, 0] = match[i, 0] - match[i-1, 1]
        gaps[i-1, 1] = match[i, 2] - match[i-1, 3]
    
    gaps = gaps.T
    return gaps

def crop_match(match, w_start, w_end):
    """
    Adjusts the match array to fit the boundaries of the window.

    Parameters
    ----------
    match : numpy.array
        Match coordinates array.
    w_start : int
        Start position for the window.
    w_end : int
        End position for the window.

    Returns
    -------
    cropped : numpy.array
        Adjusted coordinates array.

    """

    start_crop = w_start - match[0,0] # if - crop, if + pad
    end_crop = w_end - match[-1, 1] # if - crop, if + pad
    
    cropped = match.copy()

    if start_crop >= 0:
        # get starting point in query
        q_start = cropped[0, 2] + start_crop
        # adjust starting coords
        cropped[:, 0] = np.where(cropped[:,0] < w_start, w_start, cropped[:,0])
        cropped[:, 2] = np.where(cropped[:,2] < q_start, q_start, cropped[:,2])
    else:
        # add padding at the beginning of the match
        start_pad = match[0, [0, 2]] + start_crop
        cropped = np.concatenate(([[0, start_pad[0], start_pad[1], start_pad[1]]], cropped))

    if end_crop <= 0:
        # get ending point in query
        q_end = cropped[-1, 3] + end_crop
        # adjust ending coords
        cropped[:, 1] = np.where(cropped[:,1] > w_end, w_end, cropped[:,1])
        cropped[:, 3] = np.where(cropped[:,3] > q_end, q_end, cropped[:,3])
    else:
        # add padding at the beginning of the match
        end_pad = match[-1, [1, 3]] + end_crop
        cropped = np.concatenate((cropped, [[end_pad[0], 0, end_pad[1], end_pad[1]]]))
    
    return cropped

def align_match(seq, cropped, gaps):
    """
    Extract the sequence segments given by the cropped match array, add gaps
    where needed.

    Parameters
    ----------
    seq : str
        Queryequence string.
    cropped : numpy.array
        Adjusted coordinates array.

    Returns
    -------
    alig : str
        Aligned sequence.
    sum_gaps : int
        Number of gap positions in sequence.

    """
    #TODO: handle superpositions
    alig = ''
    sum_gaps = sum(np.where(gaps > 0, gaps, 0))

    q_coords = cropped[:,[2,3]]
    
    gap_char = '-'
    
    for gp, coo in zip(gaps, q_coords):
        # start by adding gaps (that's why the gapped array always starts with 0)
        alig += gap_char * max(0, gp) # number of gaps can only be positive
        seg_start = coo[0] - min(0, gp) # this solves match overlap in query
        seg_end = coo[1]
        alig += seq[seg_start:seg_end]
    return alig, sum_gaps

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
    coord_array = report[['sstart', 'send', 'qstart', 'qend']].to_numpy(dtype = int)
    
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
        self.pad_start = 0
        self.pad_end = 0
        self.crop(coords, start, end)
        self.start = start
        self.end = end
        self.get_gaps()
    
    def crop(self, coords, start, end):
        start_crop = start - coords[0,0] # if + crop, if - pad
        end_crop = end - coords[-1, 1] # if - crop, if + pad
        
        cropped = coords.copy() # cropped match
    
        if start_crop >= 0:
            # get starting point in query
            q_start = cropped[0, 2] + start_crop
            # adjust starting coords
            # every place in column 0 to the LEFT of w_start is adjusted to w_start
            # every_place in column 2 to the LEFT of q_start is adjusted to q_start
            cropped[:, 0] = np.where(cropped[:,0] < start, start, cropped[:,0])
            cropped[:, 2] = np.where(cropped[:,2] < q_start, q_start, cropped[:,2])
        else:
            # add padding at the beginning of the match
            self.pad_start = coords[0, [0, 2]] + start_crop
    
        if end_crop <= 0:
            # get ending point in query
            q_end = cropped[-1, 3] + end_crop
            # adjust ending coords
            cropped[:, 1] = np.where(cropped[:,1] > end, end, cropped[:,1])
            cropped[:, 3] = np.where(cropped[:,3] > q_end, q_end, cropped[:,3])
        else:
            # add padding at the beginning of the match
            self.pad_end = coords[-1, [1, 3]] + end_crop
        
        self.coords = cropped
    
    def get_gaps(self):
        nsegs = self.coords.shape[0]
        self.gaps = np.zeros((1,2), dtype = int)
        if len(nsegs) > 1:
            gaps = np.zeros((nsegs, 2), dtype = int) # this array will hold the gaps in the subject and query cols
            # first row in the array always 0, this allows to iterate it along the coord array (same dimentions)
        
            for i in range(1, nsegs):
                gaps[i, 0] = self.coords[i, 0] - self.coords[i-1, 1]
                gaps[i, 1] = self.coords[i, 2] - self.coords[i-1, 3]
            
            gaps = gaps.T
            self.gaps = gaps
    
    def build_alignment(self, seq, gap_char = '-'):
        alig = ''
        sum_gaps = self.gaps[:,1].sum()
        
        alig += gap_char * self.pad_start
        for gp, coo in zip(self.gaps[:,1], self.coords[:,2:]):
            alig += gap_char * gp
            alig += seq[coo[0]:coo[1] + 1]
        alig += gap_char * self.pad_end
        
        return alig, sum_gaps
        
        
class Window():
    def __init__(self, start, end, matches, seqs):
        self.w_start = start
        self.w_end = end
        self.get_in_window(matches)
        self.build_alignment(seqs)
    
    def get_in_window(self, matches):
        # extracts all the matches overlapping with the window (excludes fragments that don't overlap)
        self.matches = {}
        
        for seqid, match in matches:
            matches_in = np.where((match[:,0] < self.w_end) & (match[:,1] > self.w_start))
            miw = match[matches_in]
            if len(miw) > 0:
                self.matches[seqid] = Match(seqid, miw, self.w_start, self.w_end)

    def build_alignment(self, seqs):
        #TODO: handle superpositions
        self.aln_dict = {}
        
        for seqid, match in self.matches.items():
            seq = seqs[seqid]
            alig, sum_gaps = match.build_alignment(seq)
            self.aln_dict[seqid] = alig, sum_gaps

class WindowBuilder():
    def __init__(self, in_file, seq_file):
        self.report = load_report(in_file)
        self.matches = build_matchdict(self.report)
        self.seqs = tools.make_seqdict(seq_file)
        self.get_match_bounds()
    
    def get_match_bounds(self):
        self.lower = self.report['sstart'].min()
        self.upper = self.report['send'].max()
    
    def build_window(self, start, end):
        return Window(start, end, self.matches, self.seqs)
        
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