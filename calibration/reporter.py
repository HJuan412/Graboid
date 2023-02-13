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
        # set guide file
        self.taxguide = pd.read_csv(self.db + '/data.guide', index_col=0) # TODO: universalize filenames in database creator, fasta name / search params stored in meta file
        self.rep_dict = self.tax_guide.reset_index().set_index('taxID').SciName.to_dict()
        
    def get_summary(self, r_starts=0, r_ends=np.inf, metric='F1_score', nwins=3, show=True, *taxa):
        # generate a report for the best parameters for the given region/taxa using the selected metric
        # if only r_start and r_end are given, select the overall best parameters for the region
        # if only taxa is given, select the best parameters for the best nwins windows for each taxon
        # if both are given, select the best parameters for each taxon in the selected region(s)
        # nwins determines the number of calibration windows to be shown
        
        # establish windows
        r_starts = list(r_starts)
        r_ends = list(r_ends)
        if len(r_starts) != len(r_ends):
            raise Exception(f'Given starts and ends lengths do not match: {len(r_starts)} starts, {len(r_ends)} ends')
        regions = np.array([r_starts, r_ends])
        
        # check valid taxa
        valid_taxa = set(taxa).intersection(self.taxguide.index)
        missing = set(taxa).difference(valid_taxa)
        if len(valid_taxa) == 0 and len(taxa) > 0:
            logger.warning('No valid taxa presented. Aborting')
            return
        elif len(missing) > 0:
            logger.warning(f'The following taxa are not found in the database: {" ".join(missing)}')
        
        # prune report
        report = self.report.loc[self.report[metric] > 0]
        tab_index = pd.MultiIndex.from_product([np.arange(len(r_starts)), np.arange(nwins)])
        
        for r_idx, (start, end) in enumerate(regions):
            sub_report = report.loc[(report.w_start >= start) & (report.w_end <= end)].sort_values(metric, ascending=False)
            if len(sub_report) == 0:
                logger.warning(f'No rows selected from report with scope {start} - {end}')
                continue
            # filter for taxa
            if len(valid_taxa) > 0:
                report = report.loc[report.Taxon.isin(valid_taxa)]
                # prepare results tab
                tab_columns = pd.MultiIndex.from_product([valid_taxa, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]])
                report_tab = pd.DataFrame(index = tab_index, columns = tab_columns)
                # report tab :
                #               tax0                tax1                ...
                #               n k ws we md mt     n k ws we md mt     (md : mode, mt : metric)
                # win0  row0
                #       row1
                # ...
                for tax, sub_subreport in sub_report.groupby('Taxon'):
                    # get the best combination for every window present in sub_subreport
                    # ~sub_subreport.w_start.duplicated() used to get the first occurrence of each unique w_start
                    win_subreport = sub_subreport.loc[~sub_subreport.w_start.duplicated()].reset_index()
                    tax_rows = np.arange(min(nwins, win_subreport.shape[0])) # determine the number of windows for tax in this window is lower than nwins
                    report_tab.loc[(r_idx, tax_rows), tax] = win_subreport.iloc[tax_rows, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]]
            else:
                report_tab = pd.DataFrame(index = tab_index, columns = ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric])
                # report tab :
                #               n k ws we md mt    (md : mode, mt : metric)
                # win0  row0
                #       row1
                # ...
                win_subreport = sub_report.loc[~sub_report.w_start.duplicated()].reset_index()
                tax_rows = np.arange(min(nwins, win_subreport.shape[0])) # determine the number of windows for tax in this window is lower than nwins
                report_tab.loc[(r_idx, tax_rows)] = win_subreport.iloc[tax_rows, ['w_start', 'w_end', 'K', 'n_sites', 'mode', metric]]
        
        header = f'{nwins} best parameter combinations given by the mean {metric} values'
        # TODO: translate tax names
        if show:
            print(header)
            if len(missing) > 0:
                print('The following taxa had no matches amongst the specified windows')
                print('\n'.join(missing))
            for r_idx, (r_start, r_end) in enumerate(regions):
                print(f'Region {r_idx} [{r_start} - {r_end}]')
                print(report_tab.loc[r_idx])
