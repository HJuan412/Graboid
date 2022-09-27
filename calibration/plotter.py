#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:14:55 2022

@author: hernan
read the calibration results, present replies to queries
"""

import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#%% set logger
logger = logging.getLogger('Graboid.plotter')
logger.setLevel(logging.INFO)
#%% variables
viridis = cm.get_cmap('viridis', 256)
grey = np.array([128/256, 128/256, 128/256, 1])
viridis.set_bad(grey)

#%% functions
def generate_title(report_tab):
    # check metric
    title = f'Parameters generating the best {report_tab.columns[-1]} values for'
    # check rank
    if len(report_tab['rank'].unique()) == 1:
        title += f'Rank: {report_tab["rank"].iloc[0]}, '
    # check windows
    first_win = (report_tab.iloc[0].w_start, report_tab.iloc[0].w_end)
    last_win = (report_tab.iloc[-1].w_start, report_tab.iloc[-1].w_end)
    if first_win == last_win:
        title += f'Window: {first_win[0]} - {first_win[1]}, '
    else:
        title += f'Windows: [{first_win[0]} - {first_win[1]}] to [{last_win[0]} - {last_win[1]}], '
    # check n taxa
    title += f'{len(report_tab.Taxon.unique())} taxa'
    return title

def build_plot_tab(report_tab):
    x_values = report_tab.w_start.unique()
    x_values1 = report_tab.w_end.unique()
    y_values = report_tab.Taxon.unique()
    x_labels = [f'{x0} - {x1}' for x0,x1 in zip(x_values, x_values1)]
    
    met_tab = pd.DataFrame(index = y_values, columns = x_values)
    param_tab = pd.DataFrame(index = y_values, columns = (x_values))
    
    for (w_start, taxon), wt_tab in report_tab.groupby(['w_start', 'Taxon']):
        row = wt_tab.iloc[0]
        # row contains the parameter combination with the best results for the given parameters
        # should only be a single row in wt_tab, select first one in case of ties
        # store metric value in met_tab and parameter combination in param_tab
        # metric column is always the last one
        met_tab.at[taxon, w_start] = row.iloc[-1]
        param_tab.at[taxon, w_start] = f'k:{row.K}\nn:{row.n_sites}\nmode:{row["mode"]}'
    
    return met_tab.to_numpy(dtype=np.float), param_tab, x_labels, y_values

def plot_report(report_tab, show=True, **kwargs):
    # kwargs: out_file, use to save generated plot to out_file
    # get values, labels and title
    matrix, params, x_labels, y_labels = build_plot_tab(report_tab)
    title = generate_title(report_tab)
    
    # generate plot
    figure_size = (matrix.shape[0]/1.5, matrix.shape[1]/1.5)
    fig, ax = plt.subplots(figsize=figure_size, dpi=200)
    
    sns.heatmap(data=matrix,
                cmap=viridis,
                annot=params.to_numpy(dtype=str),
                fmt='',
                annot_kws={"size": 20 / np.sqrt(min(matrix.shape)), 'ha':'center', 'va':'center'},
                cbar=True,
                square=True,
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax)
    ax.set_xlabel('Windows')
    ax.set_xlabel('Taxons')
    ax.set_title(title)
    
    if show:
        fig.show()
    try:
        plt.savefig(kwargs['out_file'], format='png')
        logger.info(f'Plot "{title}" saved to {kwargs["out_file"]}')
    except KeyError:
        return

def build_plot_tabOLD(report_tab, metric='F1_score'):
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

def plot_reportOLD(report_tab, tax_dict=None, metric='F1_score', rank=None):
    if not rank is None:
        report_tab = report_tab.loc[report_tab['rank'] == rank]
    met_tab, param_tab = build_plot_tab(report_tab, metric)
    
    matrix = met_tab.to_numpy()
    not_null_idx = np.argwhere(matrix >= 0)
    not_null_param = np.argwhere(matrix >= 0) * [1,3]
    
    fig, ax = plt.subplots(figsize = matrix.shape)
    ax.imshow(matrix, cmap=viridis, vmax=1)
    
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
    def set_report(self, report_file):
        self.report_file = report_file
        self.report = pd.read_csv(report_file)
    
    def set_guide(self, guide_file):
        self.taxguide = pd.read_csv(guide_file, index_col=0)
    
    def query(self, w_start=0, w_end=np.inf, metric='F1_score', *taxa):
        report = self.report.loc[(self.report.w_start >= w_start) & (self.report.w_end <= w_end)]
        # check that taxa were provided
        if len(taxa) > 0:
            # query is case insensitive
            lower_taxa = [tax.lower() for tax in taxa]
            missing_tax = set(lower_taxa).difference(report.taxon.unique)
            # let user know if there are invalid taxa
            if len(missing_tax) > 0:
                print('The following taxa are not found in the window scope:')
                for mt in missing_tax:
                    print(mt)
            report = report.loc[report.taxon.isin(lower_taxa)]
        # locate best combination per window/taxon
        indexes = []
        for (win, tax), subtab in report.groupby(['w_start', 'taxon']):
            indexes.append(subtab.idxmax(metric))
        result_tab = report.loc[indexes]
        result_tab.sort_values('w_start')
        return result_tab
    
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
        
    def set_data(self, report_file):
        self.report = pd.read_csv(report_file)
    
    def query_report(self, rank, w_start=0, w_end=np.inf, metric='F1_score', *taxa):
        self.report = self.loader.query(rank, w_start, w_end, metric, taxa)
    
    def zoom_report(self, rank=None, w_start=0, w_end=np.inf, metric='F1_score', *taxa):
        # zoom onto a specific part of the calibration report
        # rank and taxa aren mutually exclusive but if rank is set only the taxa belonging to said rank will be shown
        report = self.report.loc[(self.report.w_start >= w_start) & (self.report.w_end <= w_end)]
        if not rank is None:
            report = report.loc[report['rank'] == rank]
        if len(taxa) > 0:
            # query is case insensitive
            lower_taxa = [tax.lower() for tax in taxa]
            missing_tax = set(lower_taxa).difference(report.taxon.unique)
            # let user know if there are invalid taxa
            if len(missing_tax) > 0:
                print('The following taxa are not found in the window scope:')
                for mt in missing_tax:
                    print(mt)
            report = report.loc[report.taxon.isin(lower_taxa)]
        # locate best combination per window/taxon
        indexes = []
        for (win, tax), subtab in report.groupby(['w_start', 'Taxon']):
            indexes.append(subtab.index[np.argmax(subtab[metric])])
        zoomed_report = report.loc[indexes, ['rank', 'Taxon', 'w_start', 'w_end', 'K', 'n_sites', 'mode', metric]]
        zoomed_report.sort_values('w_start', inplace=True)
        # zoomed_report.set_index(['w_start', 'taxon'], drop=True, inplace=True)
        self.zoomed = zoomed_report
        
    def plot_report(self,show=True, out_file=None):
        # kwargs: out_file, use to save generated plot to out_file
        try:
            if out_file is None:
                plot_report(self.zoomed, show)
            else:
                self.out_file = self.out_dir + '/' + out_file
                plot_report(self.zoomed, show, out_file=self.out_file)
        except AttributeError:
            print('Set the zoom level before generating a table')
            return
