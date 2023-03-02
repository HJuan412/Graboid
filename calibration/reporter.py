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
    rank_tab[('n', 'n')] = rank_tab.notna().sum(axis=1)
    param_tab = pd.concat(param_tabs)
    return rank_tab, param_tab

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
