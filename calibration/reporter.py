#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:36:32 2022

@author: hernan
This script is used to extract the best parameter combinations from a calibration result
"""

#%% libraries
from DATA import DATA
import json
import logging
import numpy as np
import pandas as pd
import re
import time

#%% set logger
logger = logging.getLogger('Graboid.reporter')
logger.setLevel(logging.INFO)

#%% functions
def get_general_params(subreport, metric='F1_score', nrows=3):
    # get the nrows best parameter combinations with the best mean values for the chosen metric
    m_mean = f'{metric} mean'
    m_std = f'{metric} std'
    params = {'K':[], 'n_sites':[], 'mode':[], m_mean:[], m_std:[]}
    for (k, n, m), subtab in subreport.groupby(['K', 'n_sites', 'mode']):
        params['K'].append(k)
        params['n_sites'].append(n)
        params['mode'].append(m)
        params[m_mean].append(subtab[metric].mean())
        params[m_std].append(subtab[metric].std())
    param_tab = pd.DataFrame(params)
    param_tab.sort_values(m_mean, ascending=False, inplace=True)
    return param_tab.iloc[:nrows]

#%% classes
class Reporter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        
    def load_report(self, report_file):
        meta_file = re.sub('.csv', '.meta', report_file) # TODO: change calibration outfiles to share a name
        report = pd.read_csv(report_file)
        self.report = report.loc[report.F1_score > 0].sort_values('F1_score', ascending=False).sort_values(['w_start'])
        with open(meta_file, 'r') as meta_handle:
            meta = json.load(meta_handle)
            self.k_range = meta['k']
            self.n_range = meta['n']
            self.windows = meta['windows']
            self.db = meta['db'] # TODO: add db to calibration meta file
        # set guide file
        self.taxguide = pd.read_csv(self.db + '/data.guide', index_col=0) # TODO: universalize filenames in database creator, fasta name / search params stored in meta file
        self.rep_dict = self.tax_guide.reset_index().set_index('taxID').SciName.to_dict()
    
    def get_summary_scope(self, w_start=0, w_end=None, metric='F1_score', nrows=3):
        # get the nrows best parameters within the scope delimited by w_start and w_end
        if w_end is None:
            # single window, defined by w_start
            # if w_start doesn't match exactly with the values in the w_start column, choose the closest value by left
            w_values = self.report.w_start.unique()
            w_start = max(w_start, w_values.min())
            w_start = w_values[w_values <= w_start].max()
            report = self.report.loc[self.report.w_start == w_start]
        else:
            report = self.report.loc[(self.report.w_start >= w_start) & (self.report.w_end <= w_end)]
            if len(report) == 0:
                logger.warning(f'No rows selected from report with scope {w_start} - {w_end}')
                return
        
        general_tabs = []
        # explore each window in the selection
        for win, win_tab in report.groupby('w_start'):
            # get the parameter combinations with the general best metric
            general_win_params = get_general_params(win_tab, metric, nrows)
            general_win_params.insert(0, 'w_start', win)
            general_win_params.insert(1, 'w_end', win + self.w_size)
            general_tabs.append(general_win_params)
        general_tab = pd.concat(general_tabs)
        general_tab.reset_index(drop=True, inplace=True)
        self.summary_scope = general_tab
    
    def get_summary_taxa(self, taxa, w_start=0, w_end=np.inf, metric='F1_score', nrows=3):
        # get the nrows best parameters for each taxa for each window between w_start and w_end
        report = self.report.loc[(self.report.w_start >= w_start) & (self.report.w_end <= w_end)]
        if len(report) == 0:
            logger.warning(f'No rows selected from report with scope {w_start} - {w_end}')
            return
        # check the provided taxa
        valid_taxa = set(taxa).intersection(self.taxguide.index)
        missing = set(taxa).difference(valid_taxa)
        if len(valid_taxa) == 0 and len(taxa) > 0:
            logger.warning('No valid taxa presented. Aborting')
            return
        elif len(missing) > 0:
            logger.warning(f'The following taxa are not found in the database: {" ".join(missing)}')
        
        taxIDs = self.taxguide.loc[valid_taxa, 'taxID'].values
        singular_indexes = []
        for win, win_tab in report.groupby('w_start'):
            if len(taxIDs) > 0:
                # get the nrows best parameters match for each taxon in the window
                for tax in taxIDs:
                    tax_tab = win_tab.loc[win_tab.Taxon == tax]
                    # no sorting by metric is needed, already done when defining sub_report
                    singular_indexes += tax_tab.index[:nrows].to_list()
        singular_tab = report.loc[singular_indexes, ['Taxon', 'rank', 'w_start', 'w_end', 'K', 'n_sites', 'mode', metric]].sort_values(metric, ascending=False).sort_values(['w_start', 'Taxon'])
        singular_tab.reset_index(drop=True, inplace=True)
        singular_tab.Taxon.replace(self.rep_dict, inplace=True)
        missing_tax = set(valid_taxa).difference(singular_tab.Taxon)
        if len(missing_tax) > 0:
            logger.warning(f'The following taxa had no matches amongst the specified windows: {" ".join(missing_tax)}')
        self.summary_taxa = singular_tab
    
    def get_summary_specific(self, w_start, K, n, mode, metric='F1_score'):
        # get the metric obtained with K and n for each taxa represented in the window starting at w_start
        report = self.report.loc[(self.report.w_start == w_start) & (self.report.K == K) & (self.report.n_sites == n) & (self.report['mode'] == mode), ['Taxon', 'rank', 'w_start', 'w_end', 'K', 'n_sites', 'mode', metric]].sort_values(metric, ascending=False).sort_values(['rank', 'Taxon'])
        report.Taxon.replace(self.rep_dict, inplace=True)
        self.summary_spec = report
    
    def get_summary(self, w_start=0, w_end=np.inf, taxa=[], metric='F1_score', nrows=3, show=True, save=False, **qwargs):
        # generate a report for the best parameters for the given window/taxa using the selected metric
        # if only w_start and w_end are given, select the overall best parameters for the window
        # if only taxa is given, select the best parameters including windows for each taxon
        # if both are given, select the best parameters for each taxon in the selected window
        # nrows determines the number of parameter combinations to be shown
        
        header = [f'{nrows} best parameter combinations given by the mean {metric} values']
        # prune report
        report = self.report.loc[self.report[metric] > 0]
        # set scope
        sub_report = report.loc[(report.w_start >= w_start) & (report.w_end <= w_end)].sort_values(metric, ascending=False).sort_values(['w_start'])
        # if scope was invalid, generate a warning and stop
        if len(sub_report) == 0:
            logger.warning(f'No rows selected from report with scope {w_start} - {w_end}')
            return
        # check valid taxa
        valid_taxa = set(taxa).intersection(self.taxguide.index)
        missing = set(taxa).difference(valid_taxa)
        if len(valid_taxa) == 0 and len(taxa) > 0:
            logger.warning('No valid taxa presented. Aborting')
            return
        elif len(missing) > 0:
            logger.warning(f'The following taxa are not found in the database: {" ".join(missing)}')
        
        taxIDs = self.taxguide.loc[valid_taxa, 'taxID'].values
        general_tabs = []
        singular_indexes = []
        # explore each window in the selection
        for win, win_tab in sub_report.groupby('w_start'):
            # get the parameter combinations with the general best metric
            general_win_params = get_general_params(win_tab, metric, nrows)
            general_win_params.insert(0, 'w_start', win)
            general_win_params.insert(1, 'w_end', win + self.w_size)
            general_tabs.append(general_win_params)
            if len(taxIDs) > 0:
                # get the nrows best parameters match for each taxon in the window
                for tax in taxIDs:
                    tax_tab = win_tab.loc[win_tab.Taxon == tax]
                    # no sorting by metric is needed, already done when defining sub_report
                    singular_indexes += tax_tab.index[:nrows].to_list()
        general_tab = pd.concat(general_tabs)
        general_tab.reset_index(drop=True, inplace=True)
        singular_tab = sub_report.loc[singular_indexes, ['Taxon', 'rank', 'w_start', 'w_end', 'K', 'n_sites', 'mode', metric]].sort_values(metric, ascending=False).sort_values(['w_start', 'Taxon'])
        singular_tab.reset_index(drop=True, inplace=True)
        singular_tab.Taxon.replace(self.rep_dict, inplace=True)
        missing_tax = set(valid_taxa).difference(singular_tab.Taxon)
        if show:
            print(header)
            print('General parameters')
            print(general_tab)
            if len(missing_tax) > 0:
                print('The following taxa had no matches amongst the specified windows')
                print('\n'.join(missing_tax))
            if len(singular_tab) > 0:
                print('Singular parameters')
                for win, win_subtab in singular_tab.groupby(['w_start', 'w_end']):
                    print(f'Window {win[0]} - {win[1]}')
                    for tax, tax_subtab in win_subtab.groupby('Taxon'):
                        print(tax)
                        print(tax_subtab)
        if save:
            try:
                out_file = f'{self.out_dir}/{qwargs["filename"]}.csv'
            except KeyError:
                out_file = f'{self.out_dir}/{time.strftime("report_summ_%d%m%Y-%H%M%S")}.csv'
            merged_tab = pd.concat([general_tab, singular_tab])
            merged_tab.to_csv(out_file)
    
    def get_taxa_summary(self, w_start, K, n, mode, metric):
        report = self.report.loc[self.report[metric] > 0]
        report = report.loc[(report.w_start == w_start) & (report.K == K) & (report.n_sites == n) & (report['mode'] == mode)]
        report.sort_values(metric, ascending=False, inplace = True)
        
    def save_report(self, filename=None):
        if filename is None:
            filename = time.strftime("report_summ_%d%m%Y-%H%M%S")
        self.out_file = f'{self.out_dir}/{filename}.csv'
        with open(self.out_file, 'w') as out_handle:
            out_handle.write('\n'.join(self.report))
        self.params.to_csv(self.out_file, mode='a')
        logger.info(f'Calibration summary saved to {self.out_file}')