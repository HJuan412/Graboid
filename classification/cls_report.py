#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:18:14 2023

@author: hernan
Build classification reports
"""

#%% libraries
import numpy as np
import pandas as pd

#%% functions
def build_prereport_V(id_array, data_array, branch_counts):
    # for each rank, generates a dataframe with columns: tax_id, total_neighbours, mean_distances, std_distances, total_support, softmax_support, n_seqs (sequences in the collapsed query branch), with indexes corresponding to the classified query sequence (indexes repeat themselves for multiple potential candidates)
    # supports come sorted in descending order by classify_V
    rank_idxs = np.unique(id_array[:,1])
    rk_tabs = {}
    for rk in rank_idxs:
        rk_locs = id_array[:,1] == rk
        qry_locs = id_array[rk_locs, 0]
        rk_tab = pd.DataFrame(np.concatenate((id_array[rk_locs,[0,2]], data_array[rk_locs], branch_counts[qry_locs]), 1), columns = 'query tax_id total_neighbours mean_distances std_distances total_support softmax_support n_seqs'.split())
        rk_tab.astype({'query':np.int32,
                       'tax_id':np.int32,
                       'total_neighbours':np.int32,
                       'mean_distances':np.float32,
                       'std_distances':np.float32,
                       'total_support':np.float32,
                       'softmax_support':np.float32,
                       'n_seqs':np.int32})
        rk_tabs[rk] = rk_tab.set_index('query').sort_index()
    return rk_tabs

def build_prereport(classifications, branch_counts):
    # for each rank, generates a dataframe with columns taxa, supports, normalized supports and sequences in branch, with indexes corresponding to the classified query sequence (indexes repeat themselves)
    rk_tabs = {}
    for rk, seqs in classifications.items():
        rk_array = []
        for idx, seq in enumerate(seqs):
            seq_array = np.array((np.full(seq[0].shape, idx), seq[0], seq[1], seq[2], np.full(seq[0].shape, branch_counts[idx]))).T
            rk_array.append(seq_array)
        rk_array = np.concatenate(rk_array)
        rk_tab = pd.DataFrame({'seq':rk_array[:,0].astype(np.int32),
                               'tax_id':rk_array[:,1].astype(np.int32),
                               'support':rk_array[:,2].astype(np.float64),
                               'softmax_support':rk_array[:,3].astype(np.float64),
                               'n_seqs':rk_array[:,4].astype(np.int32)})
        rk_tabs[rk] = rk_tab.set_index('seq')
    return rk_tabs

def build_report(pre_report, q_seqs, seqs_per_branch, guide):
    # q_seqs is the number of query sequences
    # seqs_per_branch is the array containing the number of sequences collapsed into each q_seq
    # guide is the classifier's (not extended) taxonomy guide
    # extract the winning classification for each sequence in the prereport
    # report has (multiindex) columns:
        # rk_0                      rk_1                      ...   rk_x                      n_seqs
        # Taxon  LinCode  Support   Taxon  LinCode  Support         Taxon  LinCode  Support   n_seqs
    # LinCode columns are used to group tax matches by taxonomy in the pie chart, remember to remove them before saving the table
    
    guide = guide.copy()
    guide.loc[-1, 'SciName'] = 'Undefined'
    guide.loc[-1, 'LinCode'] = 'Undefined'
    
    abv_reports = []
    for rk, rk_prereport in pre_report.items():
        # for each rank prereport, designate each sequence's classifciation as that with the SINGLE greatest support (if there is multiple top taxa, sequence is left ambiguous)
        # conclusion holds the winning classification (NOT SUPPORT) for each sequence
        conclusion = np.full(q_seqs, -1, dtype=np.int32)
        
        # get index values and starting location of each index group
        seq_indexes, seq_loc, seq_counts = np.unique(rk_prereport.index.values, return_index=True, return_counts=True)
        single = seq_counts == 1 # get sequences with a single classification, those are assigned directly
        single_idxs = seq_indexes[single]
        conclusion[single_idxs] = rk_prereport.tax.loc[single_idxs]
        
        # get winning classifications for each sequence with support for multiple taxa
        tax_array = rk_prereport.tax_id.values
        supp_array = rk_prereport.softmax_support.values
        
        for idx, loc in zip(seq_indexes[~single], seq_loc[~single]):
            if supp_array[loc] > supp_array[loc+1]:
                conclusion[idx] = tax_array[loc]
        
        # get the winning taxa's support
        clear_support = np.zeros(q_seqs)
        clear_support[seq_indexes] = supp_array[seq_loc] # get the highest support for each query sequence
        clear_support[conclusion < 0] = np.nan # delete support from inconclusive sequences
        abv_reports.append(np.array([conclusion, clear_support]))
    
    # merge all abreviation reports
    abv_reports = np.concatenate(abv_reports, axis=0)
    header = pd.MultiIndex.from_product((pre_report.keys(), ['Taxon', 'LinCode', 'support']))
    report = pd.DataFrame(abv_reports.T, columns=header) # columns: (rk, (taxon, support))
    
    # replace tax_ids with tax names
    for rk in report.columns.get_level_values(0):
        tax_codes = report.loc[:, (rk, 'Taxon')].values
        report.loc[:, (rk, 'Taxon')] = guide.loc[tax_codes, 'SciName'].values
        report.loc[:, (rk, 'LinCode')] = guide.loc[tax_codes, 'LinCode'].values
        
    # add sequence counts
    report[('n_seqs', 'n_seqs')] = seqs_per_branch
    return report

def characterize_sample(report):
    # report taxa present in sample, specifying rank, name, number of sequences and of branches, mean support and std support
    counts = []
    for rk in report.columns.get_level_values(0).unique()[:-1]:
        rk_report = report[[rk, 'n_seqs']].droplevel(0,1)
        rk_counts = []
        for tax, tax_report in rk_report.groupby('Taxon'):
            # get number of sequences, branches, mean support and std support for the current taxon
            total_seqs = tax_report.n_seqs.sum()
            n_branches = len(tax_report)
            mean_support = (tax_report.support * tax_report.n_seqs).sum() / total_seqs # account for the number of sequences in these calculations
            std_support = ((tax_report.support - mean_support)**2 * tax_report.n_seqs).sum() / total_seqs
            rk_counts.append((tax, total_seqs, n_branches, mean_support, std_support))
        rk_counts = pd.DataFrame(rk_counts, columns=['taxon', 'n_seqs', 'n_branches', 'mean_support', 'std_support'])
        rk_counts['rank'] = rk
        counts.append(rk_counts.set_index(['rank', 'taxon']))
    
    counts = pd.concat(counts)
    return counts

def designate_branch_seqs(branches, accs):
    br_array = np.concatenate([[br_idx]*len(br) for br_idx, br in enumerate(branches)])
    acc_idxs = np.concatenate(branches)
    acc_array = np.array(accs)[acc_idxs]
    designation = pd.DataFrame({'branch':br_array, 'accession':acc_array})
    return designation