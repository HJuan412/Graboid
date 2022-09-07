#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:36:32 2022

@author: hernan
This script is used to extract the best parameter combinations from a calibration result
"""

#%% libraries
import time
import pandas as pd
import pickle

#%% functions
def get_best_params(subreport, metric='F1_score', nrows=3):
    # from a subsection of a calibration report, get the average score for the given metric for each represented combination of parameters
    best_params = []
    
    for params, param_tab in subreport.groupby(['K', 'n_sites', 'mode']):
        # no need to sort the subreport because it was sorted by Reporter.report
        sample_row = param_tab.iloc[[0]].copy()
        sample_row[f'mean {metric}'] = param_tab[metric].mean()
        sample_row[f'std {metric}'] = param_tab[metric].std()
        best_params.append(sample_row)
    best_params.append(pd.DataFrame(index=[nrows], columns = sample_row.columns)) # separator row (empty)
    best_params = pd.concat(best_params, ignore_index=True)
    best_params = best_params.sort_values(f'mean {metric}', ascending = False).iloc[:nrows]
    return best_params

#%% classes
class Reporter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        
    def load_report(self, report_file):
        meta_file = report_file.split('.csv')[-1] + '.meta'
        self.report = pd.read_csv(report_file)
        with open(meta_file, 'rb') as meta_handle:
            meta = pickle.load(meta_handle)
            self.k_range = meta['k']
            self.n_range = meta['n']
            self.w_size = meta['w_size']
            self.w_step = meta['w_step']
    
    def set_guide(self, guide_file):
        self.taxguide = pd.read_csv(guide_file, index_col=0)
        
    def report(self, w_start=None, w_end=None, taxa=[], metric='F1_score', nrows=3, show=True):
        # generate a report for the best parameters for the given window/taxa using the selected metric
        # if only w_start and w_end are given, select the overall best parameters for the window
        # if only taxa is given, select the best parameters including windows for each taxon
        # if both are given, select the best parameters for each taxon in the selected window
        
        header = [f'{nrows} best parameter combinations given by the mean {metric} values']
        if w_end - w_start > self.w_len * 1.5 or w_end - w_start < self.w_len * 0.8:
            header.append(f'**Warning: The provided window has a length of {w_end - w_start}. The window length used in the calibration {self.w_len}**')
            header.append('**Parameter hints may not be reliable. Recommend performing a calibration step for the desired window**')
        
        sub_report = self.report.sort_values(metric, ascending=False)
        if not w_start is None and not w_end is None:
            sub_report = sub_report.loc[(sub_report.w_start >= w_start) & (sub_report.w_end <= w_end)]
        # check valid taxa
        missing = set(taxa).difference(self.taxguide.index)
        if len(missing) > 0:
            header.append(f'The follofing taxa are not found in the database: {" ".join(missing)}')
        elif len(missing) == len(taxa):
            header.append('No valid taxa presented. Aborting')
            return
        
        param_report = []
        # explore each window in the selection
        for win, win_tab in sub_report.groupby('w_start'):
            if len(taxa) > 0:
                # get the best match for each taxon
                for tax, tax_tab in win_tab.loc[win_tab.taxon.isin(taxa)]:
                    param_report.append(get_best_params(tax_tab, metric, nrows))
            else:
                # no taxes are set, get the generic best for each window
                best_params = get_best_params(sub_report, metric, nrows).drop(columns=['rank', 'taxon'])
                param_report.append(best_params)
        self.params = pd.concat(param_report)
        for i in range(7 - len(header)):
            header.append('')
        self.header = header
        
        if show:
            print('\n'.join(header))
            print()
            print(self.paran_report)
    
    def save_report(self, filename=None):
        if filename is None:
            filename = time.strftime("report_summ_%d%m%Y-%H%M%S")
        self.out_file = f'{self.out_dir}/{filename}.csv'
        with open(self.out_file, 'w') as out_handle:
            out_handle.write('\n'.join(self.report))
        self.params.to_csv(self.out_file, mode='a')