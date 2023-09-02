#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:32:37 2023

@author: hernan

Select the best parameter combinations from a calibration run
"""

import pandas as pd
import numpy as np
from calibration import cal_metrics

#%%
def get_general_APRF(table, window):
    """Retrieve the best average scores (& generating metrics) weighted by tax count for each rank for any of the APRF tables"""
    win_table = table[window]
    
    # get mean and std APRF scores, count taxa in each rank
    results = pd.DataFrame(columns='n k mth Mean Std Count'.split())
    for rk, rk_tab in win_table.groupby(level=0):
        # select taxa with scores above 0
        valid = rk_tab > 0
        mean_scores = pd.DataFrame(index=rk_tab.columns, columns='Mean Std Count'.split())
        for params in rk_tab.columns:
            mean = rk_tab.loc[valid[params], params].mean()
            std = rk_tab.loc[valid[params], params].std()
            count = valid[params].sum()
            mean_scores.loc[params] = [mean, std, count]
        results.loc[rk] = mean_scores.reset_index().sort_values(['Count', 'Mean'], ascending=False).iloc[0]
    
    def autoselect(results, table):
        """Automatically select the best parameter combination as that with the highest mean score for the ranks in the given window"""
        # returns tuple: (n, k, method)
        # build parameters table, index:(n, k, mth), columns:ranks
        param_idxs = pd.MultiIndex.from_frame(results[['n', 'k', 'mth']].astype(int)).unique().sort_values()
        param_tab = pd.DataFrame(index=param_idxs, columns=results.index)
        
        for rk, rk_tab in table.groupby(level=0):
            summed = rk_tab.loc[:, param_idxs].sum(0)
            valid = (rk_tab.loc[:, param_idxs] > 0).sum()
            param_tab.loc[:, rk] = summed / valid
        
        # calculate mean APRF scores, get parameter combination that generates the HIGHEST average score
        mean_scores = param_tab.mean(1).sort_values(ascending=False)
        params = mean_scores.index[0]
        return params
    
    params = autoselect(results, win_table)
    return results, params

def get_general_CE(table, window):
    """Retrieve the best average CE values (& generating metrics) weighted by tax count for each rank for the CROSS ENTROPY table"""
    win_table = table[window]
    
    # get mean and std CE values, count taxa in each rank
    results = pd.DataFrame(columns='n k mth Mean Std Count'.split())
    for rk, rk_tab in win_table.groupby(level=0):
        mean_scores = pd.DataFrame(index=rk_tab.columns, columns='Mean Std Count'.split())
        mean_scores['Mean'] = rk_tab.mean()
        mean_scores['Std'] = rk_tab.std()
        mean_scores['Count'] = (~rk_tab.isna()).sum()
        
        results.loc[rk] = mean_scores.reset_index().sort_values(['Count', 'Mean']).iloc[0]
    results[['n', 'k', 'mth', 'Count']] = results[['n', 'k', 'mth', 'Count']].astype(int) # retype table columns
    
    # select the best parameter combinations
    def autoselect(results, table):
        """Automatically select the best parameter combination as that with the lowest mean CE for the ranks in the given window"""
        # returns tuple: (n, k, method)
        # build parameters table, index:(n, k, mth), columns:ranks
        param_idxs = pd.MultiIndex.from_frame(results[['n', 'k', 'mth']].astype(int)).unique().sort_values()
        param_tab = pd.DataFrame(index=param_idxs, columns=results.index)
        
        for rk, rk_tab in table.groupby(level=0):
            param_tab.loc[:, rk] = rk_tab.loc[:, param_idxs].mean()
        
        # calculate mean CE scores, get parameter combination that generates the LOWEST average CE
        mean_scores = param_tab.mean(1).sort_values()
        params = mean_scores.index[0]
        return params
    
    params = autoselect(results, win_table)
    return results, params

def get_general_params(table, window, CE=False):
    """Execute either of the get_general... functions"""
    # returns:
        # results table: dataframe with index:ranks and columns:(n, k, mth, Mean, Std, Count)
        # parameters: tuple containig automatically selected n, k, method
    if CE:
        # use function for Cross Entropy
        return get_general_CE(table, window)
    return get_general_APRF(table, window)

def get_taxa_params(table, window, taxa, CE=False):
    """Given a set of taxa, identify parameter combinations that maximize scores for them in the set window"""
    # if CE is true, we are using a CROSS ENTROPY table, lower is best
    # returns:
        # results: table with the (COLLAPSED) parameters that generate best scores, columns: n, k, mth, Score, index (Rank, Taxon)
        # collapsed: table with columns (Rank, Taxon) and index (n, k, mth), used to locate equally good parameters
        # diff: list of taxa not present in the table
        # params: automatically selected parameter combination
        
    win_table = table[window]
    
    # verify taxa in window
    taxa_U = set([tx.upper() for tx in taxa])
    index_U = pd.Index([tx.upper() for tx in win_table.index.droplevel(0)])
    
    diff = taxa_U.difference(index_U)
    tax_table = win_table.loc[index_U.isin(taxa_U)]
    
    if len(tax_table) == 0:
        raise Exception(f'None of the given taxa {taxa} are present in the results table for window {window}')
    # build and populate results table
    results = pd.DataFrame(index = pd.MultiIndex.from_arrays([[],[]], names=['Rank', 'Taxon']), columns = 'n k mth Score'.split())
    # extract the best score and generating parameters for each taxon
    # if we are using an APRF table, higher score is better, if we are using a CE table, lower is better
    for (rk, tax), row in tax_table.iterrows():
        sorted_row = row.sort_values(ascending=CE) # sort in ascending or descending order depending on the table
        n, k, mth = sorted_row.index[0]
        score = sorted_row.iloc[0]
        results.loc[(rk, tax),:] = n, k, mth, score
    
    # collapse results
    collapsed = pd.DataFrame(index = pd.MultiIndex.from_frame(results[['n', 'k', 'mth']].astype(int)).unique(), columns=results.index)
    # locate the score of each taxon for all selected parameter combinations
    for params in collapsed.index:
        collapsed.loc[params] = tax_table[params]
    
    # sort parameters and correct results
    # if multiple parameter combinations generate optimum scores for a given taxa, select the one that is shared with another taxa to reduce redundancies
    # for each taxon, check if any of the other candidate parameter combinations generate an optimum (as high) score
    if CE:
        # priorize taxa with the highest minimum score (if using a CE table)
        collapsed.sort_values(collapsed.min(0).sort_values(ascending=False).index.tolist(), inplace=True)
        best_positions = np.argmax((collapsed <= results.Score).to_numpy(), 0) # any parameter combination generates a score lesser or equal
    else:
        # priorize taxa with the lowest maximum score (if using an APRF table)
        collapsed.sort_values(collapsed.max(0).sort_values().index.tolist(), ascending = False, inplace = True)
        best_positions = np.argmax((collapsed >= results.Score).to_numpy(), 0) # any parameter combination generates a score greater or equal
    
    for pos, tax in zip(best_positions, collapsed.columns):
        results.loc[tax, ['n', 'k', 'mth']] = collapsed.index[pos]
    
    def autoselect(results, table, CE):
        """Automaticaly select the best parameter combination as that with the best (lowest if using CE, highest otherwise) average score for the given taxa"""
        
        # pre build parameters table
        # multiindex built from candidate parameter combinations
        # columns built from given taxa
        res_idx = results.index
        param_idx = pd.MultiIndex.from_frame(results[['n', 'k', 'mth']].astype(int)).unique().sort_values()
        param_tab = pd.DataFrame(index = param_idx, columns = res_idx)
        
        # populate parameters table, average scores and select winner parameters
        param_tab.loc[param_idx, res_idx] = table.loc[res_idx, param_idx].T
        mean_scores = param_tab.mean(1).sort_values(ascending=CE)
        params = mean_scores.index.values[0]
        return params
    
    params = autoselect(results, win_table, CE)
    return results, collapsed, diff, params

def get_params_scores(table, params, *taxa):
    """Return the scores generated by the given parameter combination for the given taxa"""
    
    # verify params in table
    if not params in table.columns:
        return
    params_table = table[params]
    
    if len(taxa) > 0:
        # verify taxa in table
        taxa_U = set([tx.upper() for tx in taxa])
        index_U = pd.Index([tx.upper() for tx in table.index.droplevel(0)])
        
        diff = taxa_U.difference(index_U)
        return params_table.loc[index_U.isin(taxa_U)], diff
    return params_table, None

def report_parameters(table, window, CE, *taxa):
    """Get the best parameter combination for a given window and (if given) for a subset of taxa"""
    # table: pandas dataframe containing a full report (index is a multiindex of levels (rank, taxon), columns is a multiindex of levels(window, n, k, method))
    # window: int indicating the window index to get the parameters from
    # CE: boolean indicating if the current table is a Cross Entropy report
    # taxa: set of taxa to get the best parameters for
    
    # returns:
        # general_params: dataframe containing the parameter combinations that generate the best average score for each rank (index: ranks, columns: [n, k, mth, Mean, Std, Count (number of taxa in rank)])
        # general_auto: tuple containing the parameters (n, k, method) that generate the best average metric for all ranks
        # taxa_params: dataframe containing the parameter combinations that generate the best score for each given valid taxon
        # taxa_auto: tuple containing the parameters (n, k, method) that generate the best average metric for the given taxa
        # taxa_collapsed: dataframe containing the score generated by the candidate parameter combinations for all the given taxa
        # taxa_diff: taxa that are not present in the metrics table
    
    # get general parameters
    general_params, general_auto = get_general_params(table, window, CE)
    
    # get parameters for the specified taxa
    taxa_params = None
    taxa_collapsed = None
    taxa_diff = []
    taxa_auto = None
    if len(taxa) > 0:
        taxa_params, taxa_collapsed, taxa_diff, taxa_auto = get_taxa_params(table, window, taxa, CE)
    return general_params, general_auto, taxa_params, taxa_auto, taxa_collapsed, taxa_diff

def get_parameters(work_dir, window, metric, *taxa):
    """Generate a parameter report, print to screen"""
    # work dir is the CALIBRATION DIRECTORY containing the full reports
    # window is an int indicating the window to select the parameters from
    # metric is a single capital letter indicating the metric to base the parameter selection on
    # taxa is an optional list of taxa to select the best parameters for
    # get report files
    reports = {'A': cal_metrics.read_full_report_tab(work_dir + '/full_report_accuracy.csv'),
               'P': cal_metrics.read_full_report_tab(work_dir + '/full_report_precision.csv'),
               'R': cal_metrics.read_full_report_tab(work_dir + '/full_report_recall.csv'),
               'F': cal_metrics.read_full_report_tab(work_dir + '/full_report_f1.csv'),
               'C': cal_metrics.read_full_report_tab(work_dir + '/full_report__cross_entropy.csv')}
    
    try:
        table = reports[metric]
    except KeyError:
        raise Exception(f'Given metric {metric} is not valid. Must be A, P, R, F or C')
    general_params, general_auto, taxa_params, taxa_auto, taxa_collapsed, taxa_diff = report_parameters(table, window, metric == 'C', *taxa)
    
    def report():
        selection_metric = {'A':'accuracy',
                            'P':'precision',
                            'R':'recall',
                            'F':'f1',
                            'C':'cross entropy'}[metric]
        print(f'# Parameter selection for window {window}')
        print(f'# Selection based on metric: {selection_metric}')
        print('\n### General parameters')
        print(general_params.sort_values('Mean', ascending = metric == 'C'))
        
        if not taxa_params is None:
            print('\n### Taxon specific parameters')
            print(taxa_params)
            print('\n### Scores for all parameter combinations')
            print(taxa_collapsed.T)
        
        if len(taxa_diff) > 0:
            print('\nThe following taxa were not present in the given reports:')
            for tax in taxa_diff:
                print(tax)
                
        methods = ['unweighted', 'wknn', 'dwknn']
        print('\nAutomatically selected parameters:')
        print(f'In general:\t n: {general_auto[0]}, k: {general_auto[1]}, method: {methods[general_auto[2]]}')
        if not taxa_auto is None:
            print(f'For the given taxa:\t n: {taxa_auto[0]}, k: {taxa_auto[1]}, method: {methods[taxa_auto[2]]}')
    report()        
    return general_params, general_auto, taxa_params, taxa_auto, taxa_collapsed, taxa_diff