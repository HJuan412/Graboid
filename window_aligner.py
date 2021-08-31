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
# gapped_file = '/home/hernan/PROYECTOS/Maestria/Graboid/Reference_data/blast_reports/18S_nem_gapped.tsv'
# ungapped_file = '/home/hernan/PROYECTOS/Maestria/Graboid/Reference_data/blast_reports/18S_nem_ungapped.tsv'
gapped_file = '/home/hernan/PROYECTOS/Tesis_maestria/Graboid/Reference_data/blast_reports/18S_nem_gapped.tsv'
ungapped_file = '/home/hernan/PROYECTOS/Tesis_maestria/Graboid/Reference_data/blast_reports/18S_nem_ungapped.tsv'

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

# seqfile = '/home/hernan/PROYECTOS/Maestria/Data/18S_Nematoda.fasta'
seqfile = '/home/hernan/PROYECTOS/Tesis_maestria/Secuencias/Updated_datasets/18S_Nematoda.fasta'
seqdict = tools.make_seqdict(seqfile)

#%%
# @njit
def match_in_window(match, w_start, w_end):
    """
    Returns match records that overlap with the given window.

    Parameters
    ----------
    match : numpy.array
        Coordinate matrix of matches to the reference seuqnce (sstart, ssend, qstart, qend).
    w_start : int
        Start position for the window.
    w_end : int
        End position for the window.

    Returns
    -------
    numpy.array
        Match records overlapping with window.

    """
    matches_in = np.where((match[:,0] < w_end) & (match[:,1] > w_start))
    miw = match[matches_in]
    miw = miw[miw[:,0].argsort()]
    # return match[matches_in]
    return miw

def get_multimatch(matches):
    """
    Locates multimatches: when different sections of the query sequence overlap with the window

    Parameters
    ----------
    matches : numpy.array
        Matches in window.

    Returns
    -------
    multimatch : list
        List of numpy.array, each element represents a different section of the query overlapping with the reference.

    """

    multimatch = []
    mmstart = 0
    for i in range(1, len(miw)):
        gap = miw[i, 0] - miw[i-1, 1]
        
        # if current match overlaps with previous onw
        if gap < 0:
            # add prevous matches after last overlap to list update mmstart
            multimatch.append(miw[mmstart:i])
            mmstart = i
    # add tailing matches to multimatch
    multimatch.append(miw[mmstart:])
    
    return multimatch

# @njit
def get_gaps(match):
    """
    List all gap lengths between matches. A negative length mean two matches
    overlap. Starts with a gap of length 0 so the lenght of the gap array
    matches the length of the match array.

    Parameters
    ----------
    match : numpy.array
        Matches in window.

    Returns
    -------
    numpy.array
        Array containing the lengths of all gaps found.

    """
    if match.shape[0] == 1:
        # if there is a single match, return an array [0]
        return np.zeros(1, dtype = int)
    else:
        gaps = [0]
        for i in range(1, len(match)):
            # compare the gap between subject match and query match (not always the same),
            # keep the largest
            gap1 = match[i,0] - match[i-1, 1]
            gap2 = match[i,2] - match[i-1, 3] 
            gaps.append(max(gap1, gap2))
        return np.array(gaps)

# @njit
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
        # cropped_match[0, [0, 2]] += start_crop
        # get starting point in query
        q_start = cropped[0, 2] + start_crop
        # adjust starting coords
        cropped[:, 0] = np.where(cropped[:,0] < w_start, w_start, cropped[:,0])
        cropped[:, 2] = np.where(cropped[:,2] < q_start, q_start, cropped[:,2])
    else:
        # add padding at the beginning of the match
        start_pad = match[0,2] + start_crop
        cropped = np.concatenate(([[0, match[0, 0], start_pad, start_pad]], cropped))

    if end_crop <= 0:
        # cropped[-1, [1, 3]] += int(end_crop)
        # get ending point in query
        q_end = cropped[-1, 3] + end_crop
        # adjust ending coords
        cropped[:, 1] = np.where(cropped[:,1] > w_end, w_end, cropped[:,1])
        cropped[:, 3] = np.where(cropped[:,3] > q_end, q_end, cropped[:,3])
    else:
        # add padding at the beginning of the match
        end_pad = match[-1, 3] + end_crop
        cropped = np.concatenate((cropped, [[match[-1, 1],0, end_pad, end_pad]]))
    
    return cropped

# @njit
def prepare_seq(seq, cropped, max_gap):
    """
    Extract the sequence segments given by the cropped match array, add gaps
    where needed.

    Parameters
    ----------
    seq : str
        Queryequence string.
    cropped : numpy.array
        Adjusted coordinates array.
    max_gap : int
        Max number of gap locations to allow in the sequence.

    Returns
    -------
    alig : str
        Aligned sequence.
    sum_gaps : int
        Number of gap positions in sequence.

    """
    #TODO: handle superpositions
    alig = ''
    gaps = get_gaps(cropped)
    sum_gaps = sum(np.where(gaps > 0, gaps, 0))
    # if sum_gaps >= max_gap:
    #     return alig, sum_gaps
    q_coords = cropped[:,[2,3]]
    
    gap_char = '-'
    
    for gp, coo in zip(gaps, q_coords):
        # start by adding gaps (that's why the gapped array always starts with 0)
        alig += gap_char * max(0, gp) # number of gaps can only be positive
        seg_start = coo[0] - min(0, gp) # this solves match overlap in query
        seg_end = coo[1]
        alig += seq[seg_start:seg_end]
    return alig, sum_gaps

def make_matchdict(tab):
    matchdict = {}
    uniq_matches = tab['qseqid'].unique()
    
    for match in uniq_matches:
        # match_matrix = tab.loc[ungapped_tab['qseqid'] == match, ['sstart', 'send', 'qstart', 'qend']].sort_values(by = 'qstart').to_numpy()
        match_matrix = tab.loc[ungapped_tab['qseqid'] == match, ['sstart', 'send', 'qstart', 'qend']].to_numpy()
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
qid_array = ungapped_tab['qseqid'].to_numpy(dtype=str)
coord_array = ungapped_tab[['sstart', 'send', 'qstart', 'qend']].to_numpy(dtype = int)

#%%
# second version of match dict, returns a dict of coordinate matrixes (sstart, send, qstart, qend) for each match ordered by sstart
def make_matchdict2(qid_array, coord_array):
    matchdict = {}

    uniq_qids = np.unique(qid_array)
    
    for qid in uniq_qids:
        idx = np.where(qid_array == qid)
        match_matrix = coord_array[idx]
        match_matrix = match_matrix[match_matrix[:,0].argsort()]
        matchdict[qid] = match_matrix
    return matchdict

matchdict = make_matchdict2(qid_array, coord_array)
#%%
alig = {}
w_start = 200
w_end = 300

max_gap = 50

for seqid, match in matchdict.items():
    seq = seqdict[seqid]
    aligned = align_match(seqid, match, seq, w_start, w_end, 50)
    if aligned != 0:
        alig[seqid] = aligned
        # break

#%% make alig dataframe

total_accs = list(alig.keys())

alig_df = pd.DataFrame(index = range(len(total_accs)), columns = ['acc', 'seqlen', 'gaplen', 'seq'])

for idx, acc in enumerate(total_accs):
    
    seq, gaps = alig[acc]
    seqlen = len(seq)
    alig_df.at[idx] = [acc, seqlen, gaps, seq]
#%% get anomalous matches

anomal = alig_df.loc[alig_df['seqlen'] != 100]

#%% get multimatch
anom_acc = 'CAJEWN010004185.1'
aln = alig[anom_acc]
match = matchdict[anom_acc]
seq = seqdict[anom_acc]

miw = match_in_window(match, w_start, w_end)
cropped = crop_match(miw, w_start, w_end)

align_match(anom_acc, match, seq, w_start, w_end, 50)

#%%
# def detect_multimatch(match):