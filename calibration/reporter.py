#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:14:55 2022

@author: hernan
read the calibration results, present replies to queries
"""

from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0,1,256))
grey = np.array([128/256, 128/256, 128/256, 1])
newcolors[0] = grey
newcmp = ListedColormap(newcolors)

#%% functions
def build_plot_tab(report_tab, metric='F1_score'):
    # prepare iterables for multiindexes
    w_start = report_tab['w_start'].unique()
    w_end = report_tab['w_end'].unique()
    taxons = report_tab['taxon'].unique()
    params = ['k', 'n', 'mode']
    # prepare result dataframes
    met_tab = pd.DataFrame(index = taxons,
                            columns = pd.MultiIndex.from_arrays((w_start, w_end)))
    param_tab = pd.DataFrame(index = taxons,
                             columns = pd.MultiIndex.from_product((w_start, params)))
    # process report table
    report_tab = report_tab.sort_values(metric, ascending = False).set_index(['w_start', 'taxon'])
    for win in w_start:
        win_tab = report_tab.loc[win]
        for tax in win_tab.index.unique():
            row = win_tab.loc[tax].iloc[0]
            # row contains the parameter combination with the best results for the given parameters
            # store metric value in met_tab and parameter combination in param_tab
            met_tab.at[tax, win] = row[metric]
            param_tab.at[tax, win] = row.loc[['k', 'n', 'mode']].values
    # prepare met_tab to build heatmap
    met_tab.fillna(-1, inplace=True)
    return met_tab, param_tab

def plot_report(report_tab, tax_dict=None, metric='F1_score', rank=None):
    if not rank is None:
        report_tab = report_tab.loc[report_tab['rank'] == rank]
    met_tab, param_tab = build_plot_tab(report_tab, metric)
    
    matrix = met_tab.to_numpy()
    not_null_idx = np.argwhere(matrix >= 0)
    not_null_param = np.argwhere(matrix >= 0) * [1,3]
    
    fig, ax = plt.subplots(figsize = matrix.shape)
    ax.imshow(matrix, cmap=newcmp, vmax=1)
    
    for idx0, idx1 in zip(not_null_idx, not_null_param):
        params = param_tab.iloc[idx1[0], idx1[1]:idx1[1]+3].values
        # vignette = f'{metric}: {matrix[idx[0], idx[1]]:.3f}\n\
        #     K: {params[0]}\n\
        #     n: {params[1]}\n\
        #     {params[2]}'
        vignette = f'K:{params[0]}\nn:{params[1]}\n{params[2]}'
        ax.text(idx0[1], idx0[0], vignette, size=8, ha='center', va='center')
    
    win_labels = [f'{i[0]} - {i[1]}' for i in met_tab.columns]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(win_labels)
    ax.set_yticks(np.arange(matrix.shape[0]))
    if not tax_dict is None:
        tax_labels = [tax_dict[i] for i in met_tab.index]
        ax.set_yticklabels(tax_labels)
    else:
        ax.set_yticklabels(met_tab.index)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title(f'Best {metric}')
    ax.set_xlabel('Windows')
    ax.set_xlabel('Taxons')
    
#%% classes
class ReportLoader:
    def get_report(self, report_file):
        self.report_file = report_file
        self.report = pd.read_csv(report_file)
    
    def get_taxguide(self, taxguide_file):
        self.taxguide_file = taxguide_file
        self.taxguide = pd.read_csv(self.taxguide_file, index_col = 0)
        self.tax_dict = {taxid:tax for tax, taxid in self.taxguide['TaxID'].iteritems()}
    
    def query_tax(self, *taxa, w_start=None, w_end=None, metric='F1_score'):
        # query is case insensitive
        lower_taxa = [tax.lower() for tax in taxa]
        valid_taxa = []
        invalid_taxa = []
        
        # check which queried taxa can be found in the database
        for tax in lower_taxa:
            if tax in self.taxguide.index:
                valid_taxa.append(tax)
            else:
                invalid_taxa.append(tax)
        
        # let user know if there are invalid taxa
        if len(invalid_taxa) > 0:
            print('The following taxa are not found in the database:')
            for it in invalid_taxa:
                print(it)
        
        # locate taxIDs of queried taxa
        taxids = self.taxguide.loc[valid_taxa, 'TaxID'].tolist()
        
        # filter report for the queried taxa
        subtab = self.report.loc[self.report['taxon'].isin(taxids)]
        
        # if a given range is specified, filter subtab
        if not w_start is None:
            subtab = subtab.loc[subtab['w_start'] >= w_start]
        if not w_end is None:
            subtab = subtab.loc[subtab['w_end'] <= w_end]
        
        results = []
        # split the subtab by taxon, then by window, extract the best parameter combination for said taxon - window param
        for tax, tax_subtab in subtab.groupby('taxon'):
            for win, win_subtab in tax_subtab.groupby('w_start'):
                win_subtab.sort_values(by=metric, ascending=False, inplace=True)
                results = results.append(win_subtab.iloc[0])
        results = pd.DataFrame(results)
        results['taxon'].replace(self.tax_dict, inplace=True)
        return results

    def query_window(self, w_start=0, w_end=np.inf, metric='F1_score'):
        subtab = self.report.loc[(self.report['w_start'] >= w_start) & (self.report['w_end'] <= w_end)]
        results = []
        for win, win_subtab in subtab.groupby('w_start'):
            for tax, tax_subtab in win_subtab.groupby('taxon'):
                tax_subtab.sort_values(by=metric, ascending=False, inplace=True)
                results.append(tax_subtab.iloc[0])
        
        results = pd.DataFrame(results)
        results['taxon'].replace(self.tax_dict, inplace=True)
        return results

class Director:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.loader = ReportLoader()
        
    def set_data(self, report_file, taxguide_file):
        self.loader.set_report(report_file)
        self.loader.set_taxguide(taxguide_file)
    
    def query_report(self, metric='F1_score'):
        self.report = self.loader.query_window(metric = metric)
        
    def query_window(self, w_start, w_end, metric='F1_score'):
        self.report = self.loader.query_window(self, w_start, w_end, metric='F1_score')
        
    def query_tax(self, *taxa, w_start=None, w_end=None, metric='F1_score'):
        self.report = self.loader.query_tax(taxa, w_start=None, w_end=None, metric='F1_score')
    
    def plot_report(self, metric='F1_score', rank=None):
        plot_report(self.report, self.loader.tax_dict, metric, rank)
    
    def save_report(self, out_file):
        self.report.to_csv(f'{self.out_dir}/{out_file}')