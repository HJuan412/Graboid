#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:14:55 2022

@author: hernan
read the calibration results, present replies to queries
"""

import pandas as pd

#%%
# directories
calib_dir = 'calib_test'

taxa = ['nematoda', 'platyhelminthes']
markers = ['18s', '28s', 'coi']

#%% data loader
class DataLoader:
    def __init__(self, taxon, marker, in_dir, calib_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.calib_dir = calib_dir
        self.get_calibrations()
        self.get_taxguide()
    
    def get_calibrations(self):
        self.maj_file = f'{self.calib_dir}/{self.taxon}_{self.marker}_maj.csv'
        self.wknn_file = f'{self.calib_dir}/{self.taxon}_{self.marker}_wknn.csv'
        self.dwknn_file = f'{self.calib_dir}/{self.taxon}_{self.marker}_dwknn.csv'
        
        maj = pd.read_csv(self.maj_file, index_col = 0)
        wknn = pd.read_csv(self.wknn_file, index_col = 0)
        dwknn = pd.read_csv(self.dwknn_file, index_col = 0)
        
        self.cal_tab = pd.concat([maj, wknn, dwknn])
        self.cal_tab.sort_values('F1_score', ascending=False, inplace=True)
    
    def get_taxguide(self):
        self.taxguide_file = f'{self.in_dir}/{self.taxon}_{self.marker}.taxguide'
        self.taxguide = pd.read_csv(self.taxguide_file, index_col = 0)
        self.taxid_guide = self.taxguide.reset_index().set_index('TaxName', drop = True).rename(columns={'index':'TaxID'})
        self.taxid_guide.index = self.taxid_guide.index.str.lower()
        self.tax_dict = self.taxguide['TaxName'].to_dict()
    
    def query_tax(self, *taxa, w_start=None, w_end=None):
        # query is case insensitive
        lower_taxa = [tax.lower() for tax in taxa]
        valid_taxa = []
        invalid_taxa = []
        
        # check which queried taxa can be found in the database
        for tax in lower_taxa:
            if tax in self.taxid_guide.index:
                valid_taxa.append(tax)
            else:
                invalid_taxa.append(tax)
        
        # locate taxIDs of queried taxa
        taxids = self.taxid_guide.loc[valid_taxa, 'TaxID'].tolist()
        
        # filter each subtab for the queried taxa
        # TODO: splitting into different tables not necessary
        # maj_subtab = self.maj.loc[self.maj['taxon'].isin(taxids)]
        # wknn_subtab = self.wknn.loc[self.wknn['taxon'].isin(taxids)]
        # dwknn_subtab = self.dwknn.loc[self.dwknn['taxon'].isin(taxids)]
        
        # subtab = pd.concat([maj_subtab, wknn_subtab, dwknn_subtab])
        subtab = self.cal_tab.loc[self.cal_tab['taxon'].isin(taxids)]
        
        # if a given range is specified, filter subtab
        if not w_start is None:
            subtab = subtab.loc[subtab['w_start'] >= w_start]
        if not w_end is None:
            subtab = subtab.loc[subtab['w_end'] <= w_end]
        
        results = pd.DataFrame(columns=subtab.columns)
        # split the subtab by taxon, then by window, extract the best parameter combination for said taxon - window param
        for tax, tax_subtab in subtab.groupby('taxon'):
            for win, win_subtab in tax_subtab.groupby('w_start'):
                # win_subtab.sort_values(by='F1_score', ascending=False, inplace=True)
                results = results.append(win_subtab.iloc[0])
        
        results.replace(self.tax_dict, inplace=True)
        return results

    def query_window(self, w_start, w_end):
        subtab = self.cal_tab.loc[(self.cal_tab['w_start'] >= w_start) & (self.cal_tab['w_end'] <= w_end)]
        results = []
        for win, win_subtab in subtab.groupby('w_start'):
            for tax, tax_subtab in win_subtab.groupby('taxon'):
                results.append(tax_subtab.iloc[0])
        
        results_tab = pd.concat(results, axis=1).T
        results_tab['taxon'].replace(self.tax_dict, inplace=True)
        return results_tab

#%%
in_dir = f'{taxa[0]}_{markers[0]}/out_dir'
loader = DataLoader(taxa[0], markers[0], in_dir, calib_dir)

tax_results = loader.query_tax('chromadorea', 'rhabditida', 'ascaris lumbricoides', 'enterobius vermicularis', 'strongyloides stercoralis')
win_results = loader.query_window(200, 600)
#%%
