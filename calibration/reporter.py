#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:36:32 2022

@author: hernan
This script is used to extract the best parameter combinations from a calibration result
"""

#%% libraries
import json
import logging
import numpy as np
import pandas as pd
import re

#%% set logger
logger = logging.getLogger('Graboid.reporter')
logger.setLevel(logging.INFO)

#%% functions
def make_summ_tab(report, metric='F1_score'):
    # report : calibration report dataframe
    # metric : Accuracy Precision Recall F1_score
    # generates a short summary table, get the best score for metric for each taxon in each window
    # stores parameter combination that yields best score
    
    # build the column multiindex (window coordinates)
    windows_index = pd.MultiIndex.from_frame(report[['w_start', 'w_end']].loc[~report.w_start.duplicated()])
    rank_tabs = []
    param_tabs = []
    # get best score and parameters for each taxon per rank
    for rk, rk_subtab in report.groupby('Rank'):
        taxa = rk_subtab.Taxon.unique()
        rk_tab = pd.DataFrame(index = taxa, columns = windows_index)
        pr_tab = pd.DataFrame(index = taxa, columns = windows_index)
        for (tax, w_start), win_subtab in rk_subtab.groupby(['Taxon', 'w_start']):
            w_end = win_subtab.w_end.values[0]
            # get the row with the best value
            selected_row = win_subtab.sort_values(metric).iloc[-1]
            rk_tab.at[tax, (w_start, w_end)] = selected_row[metric]
            pr_tab.at[tax, (w_start, w_end)] = '%d %d %s' % tuple(selected_row[['K', 'n_sites', 'mode']].values)
        rk_tab.index = pd.MultiIndex.from_product([[rk], taxa], names=['rank', 'taxa'])
        pr_tab.index = pd.MultiIndex.from_product([[rk], taxa], names=['rank', 'taxa'])
        rank_tabs.append(rk_tab)
        param_tabs.append(pr_tab)
    rank_tab = pd.concat(rank_tabs)
    # count the number of columns each taxon appears in
    param_tab = pd.concat(param_tabs)
    param_tab[('n', 'n')] = rank_tab.notna().sum(axis=1)
    return rank_tab, param_tab

def get_best_params(param_tab, rows, cols):
    best_params = [params.split() for params in param_tab.to_numpy()[rows, cols]]
    params = pd.DataFrame(index = param_tab.index[rows].droplevel(0), columns = 'w_start w_end K n mode'.split())
    params[['w_start', 'w_end']] = [list(coords) for coords in param_tab.columns[cols].values]
    params[['K', 'n', 'mode']] = best_params
    return params.astype({'w_start':int, 'w_end':int, 'K':int, 'n':int})

def get_valid_taxa(taxa, guide_tab, rk_dict):
    # check that all taxa exist
    valid = []
    invalid = []
    for tx in taxa:
        if tx in guide_tab.index:
            valid.append(tx)
        else:
            invalid.append(tx)
    # get the rank for each taxa
    rks = guide_tab.loc[valid, 'rank'].tolist()
    idxs = guide_tab.loc[valid, 'taxID'].tolist()
    rows = [(rk_dict[rk], idx) for rk, idx in zip(rks, idxs)]
    return rows, valid, invalid

def get_params(mesas, score_tab, param_tab, rk_dict, guide_tab, rank=None, *taxa):
    tax_dict = {idx:tax for tax, idx in guide_tab.taxID.iteritems()}
    tax_tab = score_tab
    tax_param_tab = param_tab.iloc[:,:-1]
    if len(taxa) > 0:
        # check for valid taxa
        rows, valid, invalid = get_valid_taxa(taxa, guide_tab, rk_dict)
        if len(valid) == 0:
            logger.warning('No valid taxa found among: {taxa}')
            raise Exception
        for inv in invalid:
            logger.warning(f'Taxon {inv} not found in database')
        # tax tab contains the rows for the specified taxa
        tax_tab = score_tab.loc[rows]
        tax_param_tab = param_tab.iloc[:,:-1].loc[rows]
        # TODO: may need to check for taxa in the score_tab
    elif not rank is None:
        # a taxnomic rank was selected instead of a set of taxa
        rank_id = rk_dict[rank]
        tax_tab = score_tab.loc[rank_id]
        tax_param_tab = param_tab.iloc[:,:-1].loc[rank_id]
    
    start_coords = score_tab.columns.get_level_values(0).to_numpy(dtype=int)
    end_coords = score_tab.columns.get_level_values(1).to_numpy(dtype=int)
    
    params = []
    metrics = []
    
    for mesa in mesas:
        mesa_name = '[%d - %d]' % (mesa[0], mesa[1])
        mesa_len = int(mesa[2])
        # get windows overlapping with mesa
        wins = (start_coords >= mesa[0]) & (end_coords <= mesa[1])
        if wins.sum() == 0:
            logger.warning(f'Mesa {mesa_name} of length {mesa_len} contains no whole calibration windows')
            continue
        
        wins_tab = tax_tab.loc[:, wins]
        wins_param_tab = tax_param_tab.loc[:, wins]
        # check for empty rows
        ne_rows = (wins_tab.notna().sum(axis=1) > 0).values
        if ne_rows.sum() == 0:
            # all rows are empty
            logger.warning(f'Mesa {mesa_name} has no valid calibration results for the specified taxa')
            continue
        # warn empty rows
        for empty_tax in wins_tab.loc[~ne_rows].index.get_level_values(1):
            logger.warning(f'Mesa {mesa_name} has no valid results for taxon {tax_dict[empty_tax]}')
        # warn for rows with all 0 values
        n0_rows = (wins_tab.sum(axis=1) > 0).values
        for all0_tax in wins_tab.loc[~n0_rows & ne_rows].index.get_level_values(1):
            logger.warning(f'Mesa {mesa_name} has no above 0 scores for taxon {tax_dict[all0_tax]}')
        # locate best cell for each taxon
        best_cells = np.argmax(wins_tab.loc[n0_rows].fillna(-np.inf).to_numpy(), axis=1)
        # get best params per taxon
        best_params = get_best_params(wins_param_tab, n0_rows, best_cells)
        best_params['mesa'] = mesa_name
        # get best metrics per taxon
        best_metrics = pd.DataFrame(wins_tab.to_numpy()[n0_rows, best_cells], index=wins_tab.index[n0_rows].droplevel(0), columns=[mesa_name]).T
        params.append(best_params)
        metrics.append(best_metrics)
    
    params = pd.concat(params)
    metrics = pd.concat(metrics)
    return params, metrics

def get_consensus_params(report):
    # this function takes unique parameter combinations for a multi taxon calibration report, used for automatic parameter selection
    # returns a dataframe of the form:
        # win n k modes taxs
        # win contains tuples of window coordinates, used as the dataframe index (values may repeat if there are multiple parameter combinations for each window)
        # n, k & modes, optimum combinations of sites, neighbours and classification modes (modes is a list if multiple modes are applied)
        # taxs, taxons for which a given parameter combination is the most adequate (list)
    
    consensus = pd.DataFrame(columns = 'win n k modes taxs'.split())
    # begin by exploring each region in the report
    for reg, reg_subtab in report.groupby(level=0):
        sub_report = reg_subtab.droplevel(0)
        # merge window coordinates and n & k parameters into single value by adding the second half as decimal numbers
        # get orders of magnitude
        orders = np.floor(np.log10(sub_report.loc[:, (slice(None), ['k', 'e'])].max()))
        ids0 = sub_report.loc[:, (slice(None), ['n', 's'])]
        ids1 = sub_report.loc[:, (slice(None), ['k', 'e'])] * 10 ** (-orders - 1)
        ids = (ids0 + ids1.to_numpy()).rename(columns={'s':'wins', 'n':'params'})
        
        # flatten ids table, turn taxon into another attribute
        flattened = []
        for tax in ids.columns.levels[0]:
            subtab = ids.loc[:, tax].copy()
            subtab['tax'] = tax
            flattened.append(subtab)
        # keep the index location for each parameter combination in the original report
        flattened = pd.concat(flattened).reset_index(drop = False, names = 'idxs')
        
        for (win, par), subtab in flattened.groupby(['wins', 'params']):
            indexes = subtab.idxs.to_list()
            taxes = subtab.tax.to_list()
            modes = reg_subtab.loc[indexes, (taxes, 'md')].to_list()
            w_coords = tuple(reg_subtab.loc[:, (taxes[0], ['ws', 'we'])].iloc[0])
            n = reg_subtab.loc[:, (taxes[0], 'n')].unique()[0]
            k = reg_subtab.loc[:, (taxes[0], 'k')].unique()[0]
            consensus.loc[len(consensus)] = [w_coords, n, k, modes, taxes]
        consensus.set_index('win')
    # consensus table uses: group by index, extract unique parameter combinations, run classifications, register differences for each query seq
    return consensus
    
#%% classes
class Reporter:
    def load_report(self, report_file):
        meta_file = re.sub('.report', '.meta', report_file)
        report = pd.read_csv(report_file)
        self.report = report.loc[report.F1_score > 0].sort_values('F1_score', ascending=False).sort_values('w_start')
        with open(meta_file, 'r') as meta_handle:
            meta = json.load(meta_handle)
            self.k_range = meta['k']
            self.n_range = meta['n']
            self.windows = meta['windows']
            self.db = meta['db']
            self.guide_file = meta['guide']
        # set guide file
        self.tax_guide = pd.read_csv(self.guide_file, index_col=0)
        self.rep_dict = self.tax_guide.reset_index().set_index('taxID').SciName.to_dict()
        
    def get_summary(self, r_starts=0, r_ends=np.inf, metric='F1_score', nwins=3, show=True, *taxa):
        # generate a report for the best parameters for the given region/taxa using the selected metric
        # if only r_start and r_end are given, select the overall best parameters for the region
        # if only taxa is given, select the best parameters for the best nwins windows for each taxon
        # if both are given, select the best parameters for each taxon in the selected region(s)
        # nwins determines the number of calibration windows to be shown
        
        # establish windows
        r_starts = list([r_starts])
        r_ends = list([r_ends])
        if len(r_starts) != len(r_ends):
            raise Exception(f'Given starts and ends lengths do not match: {len(r_starts)} starts, {len(r_ends)} ends')
        regions = np.array([r_starts, r_ends]).T
        
        # check valid taxa
        valid_taxa = list(set(taxa).intersection(self.tax_guide.index))
        missing = set(taxa).difference(valid_taxa)
        if len(valid_taxa) == 0 and len(taxa) > 0:
            logger.warning('No valid taxa presented. Aborting')
            return
        elif len(missing) > 0:
            logger.warning(f'The following taxa are not found in the database: {" ".join(missing)}')
        
        # get corresponding taxIDs
        tax_ids = self.tax_guide.loc[valid_taxa, 'taxID'].values
        # prune report
        report = self.report.loc[self.report[metric] > 0]
        tab_index = pd.MultiIndex.from_product([np.arange(len(r_starts)), np.arange(nwins)])
        
        for r_idx, (start, end) in enumerate(regions):
            sub_report = report.loc[(report.w_start >= start) & (report.w_end <= end)].sort_values(metric, ascending=False)
            if len(sub_report) == 0:
                logger.warning(f'No rows selected from report with scope {start} - {end}')
                continue
            # filter for taxa
            if len(tax_ids) > 0:
                report = report.loc[report.Taxon.isin(tax_ids)]
                # prepare results tab
                tab_columns = pd.MultiIndex.from_product([tax_ids, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]])
                report_tab = pd.DataFrame(index = tab_index, columns = tab_columns)
                # report tab :
                #               tax0                tax1                ...
                #               n k ws we md mt     n k ws we md mt     (md : mode, mt : metric)
                # reg0  win0
                #       win1
                # ...
                for tax, sub_subreport in sub_report.groupby('Taxon'):
                    # get the best combination for every window present in sub_subreport
                    # ~sub_subreport.w_start.duplicated() used to get the first occurrence of each unique w_start
                    win_subreport = sub_subreport.loc[~sub_subreport.w_start.duplicated()].reset_index()
                    tax_rows = np.arange(min(nwins, win_subreport.shape[0])) # determine the number of windows for tax in this window is lower than nwins
                    report_tab.loc[(r_idx, tax_rows), tax] = win_subreport.iloc[tax_rows, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]]
                # translate taxID's back to taxon names
                report_tab.columns = pd.MultiIndex.from_product([valid_taxa, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]])
            else:
                report_tab = pd.DataFrame(index = tab_index, columns = ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric])
                # report tab :
                #               n k ws we md mt    (md : mode, mt : metric)
                # reg0  win0
                #       win1
                # ...
                win_subreport = sub_report.loc[~sub_report.w_start.duplicated()].reset_index()
                tax_rows = np.arange(min(nwins, win_subreport.shape[0])) # determine the number of windows for tax in this window is lower than nwins
                report_tab.loc[(r_idx, tax_rows)] = win_subreport.iloc[tax_rows][['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]]
        
        header = f'{nwins} best parameter combinations given by the mean {metric} values'
        if show:
            print(header)
            if len(missing) > 0:
                print('The following taxa had no matches amongst the specified windows')
                print('\n'.join(missing))
            for r_idx, (r_start, r_end) in enumerate(regions):
                print(f'Region {r_idx} [{r_start} - {r_end}]')
                print(report_tab.loc[r_idx])
