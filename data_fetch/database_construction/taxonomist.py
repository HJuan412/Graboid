#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:40:24 2022

@author: hernan

This script fetches the taxonomy data for the downloaded records
"""

#%% modules
from Bio import Entrez

import bold_marker_vars
import logging
import numpy as np
import pandas as pd

#%% setup logger
logger = logging.getLogger('database_logger.taxonomist')

#%% variables
valid_databases = ['BOLD', 'NCBI']

#%% functions
def detect_taxidfiles(file_list):
    # TODO: NOTE: this function behaves exactly like lister.detect_summ(), may need more analogs to it in the future, may define a generic one in a tools module
    # given a list of taxID files, identify the database to which each belongs
    # works under the assumption that there is a single file per database
    taxid_files = {}
    for file in file_list:
        database = file.split('_')[-1].split('.')[0]
        if database in valid_databases:
            taxid_files[database] = file
    return taxid_files

def tax_slicer(tax_list, chunksize=500):
    # TODO: NOTE: this function behaves exactly like fetcher.acc_slicer(), may need more analogs to it in the future, may define a generic one in a tools module
    # slice the list of taxids into chunks
    n_taxs = len(tax_list)
    for i in np.arange(0, n_taxs, chunksize):
        yield tax_list[i:i+chunksize]

def extract_tax_data(record):
    # generates taxonomy dictionary ({rank:ScientificName, rank_id:TaxID, ...}) for the given records
    taxonomy = {}
    
    lineage = record['LineageEx'] + [{'TaxId':record['TaxId'], 'ScientificName':record['ScientificName'], 'Rank':record['Rank']}]
    for lin in lineage:
        taxonomy[lin['Rank']] = lin['ScientificName']
        taxonomy[f'{lin["Rank"]}_id'] = lin['TaxId']
    
    return taxonomy

#%% classes
class Taxer:
    def generate_outfiles(self):
        header = self.taxid_file.split('/')[-1].split('.')[0]
        self.tax_out = f'{self.out_dir}/{header}.tax'
    
    def fill_blanks(self):
        # fills missing records with the last known taxon name or id
        # get locarions of missing values in ALL RANKS
        blanks = self.tax_tab0[self.ranks].isnull()
        # begin filling from the SECOND highest rank
        for idx, rk in enumerate(self.ranks[1:]):
            # locations of missing values in current rank
            blank_idx = blanks.index[blanks[rk]]
            parent_rk = self.ranks[idx]
            # fill missing values at current rank with the value of their parent rank
            self.tax_tab0.at[blank_idx, rk] = self.tax_tab0.loc[blank_idx, parent_rk]
            self.tax_tab0.at[blank_idx, f'{rk}_id'] = self.tax_tab0.loc[blank_idx, f'{parent_rk}_id']
    
    # def fill_blanks(self, tab, idtab):
    #     # TODO: see if this version works for inheritor taxers, if it doesn't uncomment the method in each of them
    #     # fills missing records with the last known taxon name or id
    #     # get locarions of missing values in ALL RANKS
    #     blanks = tab.isnull()
    #     # begin filling from the SECOND highest rank
    #     for idx, rk in enumerate(self.ranks[1:]):
    #         # locations of missing values in current rank
    #         blank_idx = blanks.loc[blanks[rk] == 1].index
    #         parent_rk = self.ranks[idx]
    #         # fill missing values at current rank with the value of their parent rank
    #         tab.at[blank_idx, rk] = tab.loc[blank_idx, parent_rk]
    #         idtab.at[blank_idx, rk] = id.loc[blank_idx, parent_rk]

class TaxonomistNCBI(Taxer):
    # procures the taxonomic data for the NCBI records
    def __init__(self, taxid_file, ranks, out_dir):
        self.taxid_file = taxid_file
        self.ranks = ranks
        self.out_dir = out_dir
        
        self.generate_outfiles()
        self.read_taxid_file()
        self.make_tax_tables()
        
        self.logger = logging.getLogger('database_logger.taxonomist.NCBI')
        
        self.failed = []
    
    def read_taxid_file(self):
        # reads the acc:taxid table
        self.taxid_list = None
        self.taxid_reverse = None
        self.uniq_taxs = None

        taxid_list = pd.read_csv(self.taxid_file, index_col = 0, header = None)
        
        # generates a warning if taxid_list is empty
        if len(taxid_list) == 0:
            self.logger.warning(f'Summary file {self.in_file} is empty')
            return

        self.taxid_list = taxid_list[1] # index are the accession codes
        # get a reversed taxid list and a list of unique taxes
        self.taxid_reverse = pd.Series(index = taxid_list.values, data = taxid_list.index)
        self.uniq_taxs = taxid_list.unique()
    
    def make_tax_tables(self):
        # generate empty taxonomy table (with two columns per rank one for name, one for ID), guide table link a taxid with its full taxonomy
        cols = [f'{rk}_{tail}' for rk in self.ranks for tail in ('name', 'taxID')]
        self.tax_table = pd.DataFrame(index = self.taxid_list.index, columns = cols)
        # name tax_tab0 used for compatibility with method fill_blanks
        self.tax_tab0 = pd.DataFrame(index = self.uniq_taxs, columns = cols) # this will be used to store the taxonomic data and later distribute it to each record
    
    def dl_tax_records(self, tax_list, chunksize=500):
        # attempts to download the taxonomic records in chunks of size chunksize
        chunks = tax_slicer(tax_list, chunksize)
        failed = []
        for chunk in chunks:
            try:
                tax_handle = Entrez.efetch(db = 'taxonomy', id = chunk, retmode = 'xml')
                tax_records = Entrez.read(tax_handle)
                
                self.__update_guide(chunk, tax_records)
            except:
                failed += chunk
        
        self.failed = failed
        if len(failed) > 0:
            self.logger.warning('Failed to download {len(failed)} taxIDs of {len(self.uniq_taxs)}')
    
    def retry_dl(self, max_attempts=3):
        # if some taxids couldn't be downloaded, rety up to max_attempts times
        attempt = 1
        while attempt <= max_attempts and len(self.failed) > 0:
            self.dl_tax_records(self.failed)
    
    # def fill_blanks(self):
    #     # fills missing records with the last known taxon name or id
    #     # get locarions of missing values in ALL RANKS
    #     blanks = self.guide_table.isnull()
    #     # begin filling from the SECOND highest rank
    #     for idx, rk in enumerate(self.ranks[1:]):
    #         # locations of missing values in current rank
    #         blank_idx = blanks.loc[blanks[rk] == 1].index
    #         parent_rk = self.ranks[idx]
    #         # fill missing values at current rank with the value of their parent rank
    #         self.guide_table.at[blank_idx, rk] = self.guide_table.loc[blank_idx, parent_rk]
    #         self.guideid_table.at[blank_idx, rk] = self.guideid_table.loc[blank_idx, parent_rk]

    def __update_guide(self, taxids, records):
        # extract data from a single record and updates the guide table
        for taxid, record in zip(taxids, records):
            tax_dict = extract_tax_data(record)
            for rk in self.ranks:
                self.tax_tab0.at[taxid, rk] = tax_dict[rk]
                self.tax_tab0.at[taxid, f'{rk}_id'] = tax_dict[f'{rk}_id']
    
    def update_tables(self):
        # write the final taxonomic tables using the complete guide tables
        for taxid in self.uniq_taxs:
            instances = self.taxid_rev.loc[taxid] # records with the given ID
            self.tax_table.at[instances] = self.tax_tab0.loc[taxid].values
    
    def taxing(self, chunksize=500, max_attempts=3):
        if self.taxid_list is None:
            return

        self.dl_tax_records(chunksize)
        self.retry_dl(max_attempts)
        self.fill_blanks()
        self.update_tables()
        
        self.tax_table.to_csv(self.tax_out)

class TaxonomistBOLD(Taxer):
    # generates the taxomoic tables for the records downloaded from BOLD
    def __init__(self, taxid_file, ranks, out_dir):
        self.taxid_file = taxid_file
        self.ranks = ranks
        self.out_dir = out_dir

        self.generate_outfiles()
        self.__set_marker_vars()
        
        self.read_taxid_file()
        
        self.logger = logging.getLogger('database_logger.taxonomist.BOLD')

    def __set_marker_vars(self):
        # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
        marker = self.taxid_file.split('/')[-1].split('_')[1]
        self.marker_vars = list(marker)
    
    def read_taxid_file(self):
        bold_tab = pd.read_csv(self.in_file, sep = '\t', encoding = 'latin-1', index_col = 0, low_memory = False) # latin-1 to parse BOLD files

        if len(bold_tab) == 0:
            self.logger.warning(f'Summary file {self.in_file} is empty')
            self.bold_tab = None
            return
        # TODO: NOTE have to change this.
        bold_tab = bold_tab.loc[bold_tab['markercode'].isin(self.marker_vars)]
        self.bold_tab = bold_tab
        
    def get_tax_tab(self):
        # extract the relevant columns (tax name and taxID) fro the BOLD table
        cols = [f'{rk}_{tail}' for rk in self.ranks for tail in ('name', 'taxID')]
        tax_tab = self.bold_tab.loc[:, cols]
        # rename columns (replace '_name' and '_taxID' sufixes for '' and '_id')
        # name tax_tab0 used for compatibility with method fill_blanks
        self.tax_tab0 = tax_tab.rename(columns = {f'{rk}_{tail0}':f'{rk}{tail1}' for rk in self.ranks for tail0, tail1 in zip(('tax', 'id'), ('', '_id'))})
    
    # def fill_blanks(self):
    #     # fills missing records with the last known taxon name or id
    #     # get locarions of missing values in ALL RANKS
    #     blanks = self.tax_tab[self.ranks].isnull()
    #     # begin filling from the SECOND highest rank
    #     for idx, rk in enumerate(self.ranks[1:]):
    #         # locations of missing values in current rank
    #         blank_idx = blanks.index[blanks[rk]]
    #         parent_rk = self.ranks[idx]
    #         # fill missing values at current rank with the value of their parent rank
    #         self.tax_tab.at[blank_idx, rk] = self.tax_tab.loc[blank_idx, parent_rk]
    #         self.tax_tab.at[blank_idx, f'{rk}_id'] = self.tax_tab.loc[blank_idx, f'{parent_rk}_id']
    
    def taxing(self, chunksize=None, max_attempts=None):
        # chunksize and max_attempts kept for compatibility
        self.get_tax_tabs()
        self.fill_blanks()
        
        self.tax_tab0.to_csv(self.tax_out)

taxer_dict = {'BOLD':TaxonomistBOLD,
              'NCBI':TaxonomistNCBI}

class Taxonomist:
    def __init__(self, out_dir):
        self.taxid_files = {}
        self.set_ranks()
        self.out_dir = out_dir
        
        self.out_files = {}
    
    def get_taxidfiles(self, taxid_files):
        self.taxid_files = taxid_files
        # check that summaries are a list 
        if isinstance(self.taxid_files, list):
            self.taxid_files = detect_taxidfiles(taxid_files)
    
    def set_ranks(self, ranklist=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = ranklist
        # for taxr in self.taxers.values():
        #     taxr.set_ranks(ranklist)
    
    def taxing(self, taxid_files, chunksize=500, max_attempts=3):
        # taxid_files dict with {database:taxid_file}
        # check that summ_file container is not empty
        if len(taxid_files) == 0:
            logger.warning('No valid taxid files detected')
            return
        
        for database, taxid_file in taxid_files.items():
            taxer = taxer_dict[database](taxid_file, self.ranks, self.out_dir)
            taxer.taxing(chunksize, max_attempts)
            self.out_files[database] = taxer.tax_out
