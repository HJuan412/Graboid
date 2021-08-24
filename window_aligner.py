#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:11:34 2021

@author: hernan
"""

from glob import glob
from numba import njit
import numpy as np
import pandas as pd
import toolkit as tools

#%%
gapped_file = '/home/hernan/PROYECTOS/Maestria/Graboid/Reference_data/blast_reports/18S_nem_gapped.tsv'
ungapped_file = '/home/hernan/PROYECTOS/Maestria/Graboid/Reference_data/blast_reports/18S_nem_ungapped.tsv'

gapped_tab = pd.read_csv(gapped_file, sep = '\t', header = None)
ungapped_tab = pd.read_csv(ungapped_file, sep = '\t', header = None)

gapped_tab.rename(columns = {0:'qseqid',
                             1:'sseqid',
                             2:'pident',
                             3:'length',
                             4:'mismatch',
                             5:'gapopen',
                             6:'qstart',
                             7:'qend',
                             8:'sstart',
                             9:'send',
                             10:'evalue',
                             11:'bitscore'}, inplace = True)
ungapped_tab.rename(columns = {0:'qseqid',
                             1:'sseqid',
                             2:'pident',
                             3:'length',
                             4:'mismatch',
                             5:'gapopen',
                             6:'qstart',
                             7:'qend',
                             8:'sstart',
                             9:'send',
                             10:'evalue',
                             11:'bitscore'}, inplace = True)

#%%
test_qseq = 'AB009317'

full_match = gapped_tab.loc[gapped_tab['qseqid'] == test_qseq, ['sstart', 'send', 'qstart', 'qend']].to_numpy()
partial_match = ungapped_tab.loc[ungapped_tab['qseqid'] == test_qseq, ['sstart', 'send', 'qstart', 'qend']].sort_values(by = 'qstart').to_numpy()

seqfile = '/home/hernan/PROYECTOS/Maestria/Data/18S_Nematoda.fasta'
seqdict = tools.make_seqdict(seqfile)

#%%
# @njit
def match_in_window(match, w_start, w_end):
    matches_in = np.where((match[:,0] < w_end) & (match[:,1] > w_start))
    return match[matches_in]

# @njit
def get_gaps(match):
    if match.shape[0] == 1:
        return np.zeros(1, dtype = int)
    else:
        gaps = [0]
        for i in range(1, len(match)):
            gap1 = match[i,0] - match[i-1, 1]
            gap2 = match[i,2] - match[i-1, 3]
            gaps.append(max(gap1, gap2))
        # gaps = np.array([[match[i-1, 3], match[i, 2]] for i in range(1, len(match))])
        # return np.concatenate(([0], gaps[:,1] - gaps[:,0]))
        return np.array(gaps)

# @njit
def crop_match(match, w_start, w_end):
    start_crop = w_start - match[0,0] # if + crop, if - pad
    end_crop = w_end - match[-1, 1] # if - crop, if + pad
    
    cropped_match = match
    if start_crop >= 0:
        cropped_match[0, [0, 2]] += start_crop
    else:
        start_pad = match[0,2] + start_crop
        cropped_match = np.concatenate(([[0, match[0, 0], start_pad, start_pad]], match))
    if end_crop <= 0:
        cropped_match[-1, [1, 3]] += int(end_crop)
    else:
        end_pad = match[-1, 3] + end_crop
        cropped_match = np.concatenate((match, [[match[-1, 1],0, end_pad, end_pad]]))
    
    return cropped_match

# @njit
def prepare_seq(seq, cropped, max_gap):
    #TODO: handle superpositions
    gaps = get_gaps(cropped)
    if sum(np.where(gaps > 0, gaps, 0)) >= max_gap:
        return 0
    q_coords = cropped[:,[2,3]]
    
    alig = ''
    gap_char = '-'
    for gp, coo in zip(gaps, q_coords):
        alig += gap_char * max(0, gp)
        seg_start = coo[0] - min(0, gp)
        seg_end = coo[1]
        alig += seq[seg_start:seg_end]
    return alig

def make_matchdict(tab):
    matchdict = {}
    uniq_matches = tab['qseqid'].unique()
    
    for match in uniq_matches:
        match_matrix = tab.loc[ungapped_tab['qseqid'] == match, ['sstart', 'send', 'qstart', 'qend']].sort_values(by = 'qstart').to_numpy()
        matchdict[match] = match_matrix
    
    return matchdict

def align_match(seqid, match, seq, w_start, w_end, max_gap):
    in_window = match_in_window(match, w_start, w_end)
    if len(in_window) == 0:
        return 0
    else:
        cropped = crop_match(in_window, w_start, w_end)
        aligned = prepare_seq(seq, cropped, max_gap)
        
        return aligned
#%%
matchdict = make_matchdict(ungapped_tab)
#%%
alig = {}
w_start = 600
w_end = 700

max_gap = 30

for seqid, match in matchdict.items():
    seq = seqdict[seqid]
    aligned = align_match(seqid, match, seq, w_start, w_end, 50)
    if aligned != 0:
        alig[seqid] = aligned
        # break
    else:
        print(0)