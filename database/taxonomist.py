#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:40:24 2022

@author: hernan

This script fetches the taxonomy data for the downloaded records
"""

#%% modules
from Bio import Entrez

# from data_fetch.database_construction import bold_marker_vars
import http
import logging
import numpy as np
import pandas as pd

#%% setup logger
logger = logging.getLogger('Graboid.database.taxonomist')

#%% variables
valid_databases = ['BOLD', 'NCBI']

#%% functions
def detect_taxidfiles(file_list):
    # NOTE: this function behaves exactly like lister.detect_summ(), may need more analogs to it in the future, may define a generic one in a tools module
    # given a list of taxID files, identify the database to which each belongs
    # works under the assumption that there is a single file per database
    taxid_files = {}
    for file in file_list:
        database = file.split('_')[-1].split('.')[0]
        if database in valid_databases:
            taxid_files[database] = file
    return taxid_files

def tax_slicer(tax_list, chunksize=500):
    # NOTE: this function behaves exactly like fetcher.acc_slicer(), may need more analogs to it in the future, may define a generic one in a tools module
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
            parent_rk = self.ranks[idx]
            # locations of missing values in current rank
            blanks_in_rk = blanks[rk].values
            
            rk_vals = self.tax_tab0[[rk, f'{rk}_id']].values
            parent_vals = self.tax_tab0[[parent_rk, f'{parent_rk}_id']].values
            rk_vals[blanks_in_rk] = parent_vals[blanks_in_rk]
            self.tax_tab0[[rk, f'{rk}_id']] = rk_vals

class TaxonomistNCBI(Taxer):
    # procures the taxonomic data for the NCBI records
    def __init__(self, taxid_file, ranks, out_dir):
        self.taxid_file = taxid_file
        self.ranks = ranks
        self.out_dir = out_dir
        
        self.generate_outfiles()
        self.read_taxid_file()
        self.make_tax_tables()
        
        self.logger = logging.getLogger('Graboid.database.taxonomist.NCBI')
        
        self.failed = []
    
    def read_taxid_file(self):
        # reads the acc:taxid table
        self.taxid_list = None
        self.taxid_reverse = None
        self.uniq_taxs = None

        taxid_list = pd.read_csv(self.taxid_file, index_col = 0, header = None)
        
        # generates a warning if taxid_list is empty
        if len(taxid_list) == 0:
            self.logger.error(f'Summary file {self.in_file} is empty')
            return

        self.taxid_list = taxid_list[1] # index are the accession codes
        # get a reversed taxid list and a list of unique taxes
        self.taxid_reverse = pd.Series(index = self.taxid_list.values, data = taxid_list.index)
        self.uniq_taxs = np.unique(taxid_list)
    
    def make_tax_tables(self):
        # generate empty taxonomy table (with two columns per rank one for name, one for ID), guide table link a taxid with its full taxonomy
        cols = [f'{rk}{tail}' for rk in self.ranks for tail in ('', '_id')]
        self.tax_table = pd.DataFrame(index = self.taxid_list.index, columns = cols)
        # name tax_tab0 used for compatibility with method fill_blanks
        self.tax_tab0 = pd.DataFrame(index = self.uniq_taxs, columns = cols) # this will be used to store the taxonomic data and later distribute it to each record
    
    def dl_tax_records(self, tax_list, chunksize=500, max_attemps=3):
        # attempts to download the taxonomic records in chunks of size chunksize
        chunks = tax_slicer(tax_list, chunksize)
        n_chunks = int(np.ceil(len(tax_list)/chunksize))
        failed = []
        for idx, chunk in enumerate(chunks):
            print(f'Retrieving taxonomy. Chunk {idx + 1} of {n_chunks}')
            tax_records = []
            for attempt in range(max_attemps):
                try:
                    tax_handle = Entrez.efetch(db = 'taxonomy', id = chunk, retmode = 'xml')
                    tax_records = Entrez.read(tax_handle)
                    break
                except IOError:
                    logger.debug('Interrupted taxing due to conection error')
                    continue
                except http.client.client.Incomplete:
                    logger.debug('Interrupted taxing due to bad file')
                    continue
            if len(tax_records) != len(chunk):
                failed += list(chunk)
                continue
            try:
                self.update_guide(chunk, tax_records)
            except KeyError:
                failed += list(chunk)
                continue
        
        self.failed = failed
        if len(failed) > 0:
            self.logger.warning(f'Failed to download {len(failed)} taxIDs of {len(self.uniq_taxs)}')
    
    def retry_dl(self, max_attempts=3):
        # if some taxids couldn't be downloaded, rety up to max_attempts times
        attempt = 1
        while attempt <= max_attempts and len(self.failed) > 0:
            self.dl_tax_records(self.failed)
            
    def update_guide(self, taxids, records):
        # extract data from a single record and updates the guide table
        tax_dicts = [extract_tax_data(record) for record in records]
        tax_tab = pd.DataFrame(tax_dicts, index = taxids)
        tax_tab = tax_tab[[f'{rk}{tail}' for rk in self.ranks for tail in ('', '_id')]]
        self.tax_tab0.update(tax_tab)
    
    def update_tables(self):
        # write the final taxonomic tables using the complete guide tables
        for taxid in self.uniq_taxs:
            instances = self.taxid_reverse.loc[taxid] # records with the given ID
            self.tax_table.at[instances] = self.tax_tab0.loc[taxid].values
    
    def taxing(self, chunksize=500, max_attempts=3):
        if self.taxid_list is None:
            return

        self.dl_tax_records(self.uniq_taxs, chunksize)
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
        
        self.logger = logging.getLogger('Graboid.database.taxonomist.BOLD')

    def __set_marker_vars(self):
        # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
        marker = self.taxid_file.split('/')[-1].split('_')[1]
        self.marker_vars = list(marker)
    
    def read_taxid_file(self):
        bold_tab = pd.read_csv(self.taxid_file, index_col = 0)

        if len(bold_tab) == 0:
            self.logger.error(f'Summary file {self.in_file} is empty')
            self.bold_tab = None
            return
        self.bold_tab = bold_tab
        
    def get_tax_tab(self):
        # extract the relevant columns (tax name and taxID) fro the BOLD table
        cols = [f'{rk}_{tail}' for rk in self.ranks for tail in ('name', 'taxID')]
        tax_tab = self.bold_tab.loc[:, cols]
        # rename columns (replace '_name' and '_taxID' sufixes for '' and '_id')
        # name tax_tab0 used for compatibility with method fill_blanks
        self.tax_tab0 = tax_tab.rename(columns = {f'{rk}_{tail0}':f'{rk}{tail1}' for rk in self.ranks for tail0, tail1 in zip(('name', 'taxID'), ('', '_id'))})
    
    def taxing(self, chunksize=None, max_attempts=None):
        # chunksize and max_attempts kept for compatibility
        self.get_tax_tab()
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
    
    def taxing(self, taxid_files, chunksize=500, max_attempts=3):
        # taxid_files dict with {database:taxid_file}
        # check that summ_file container is not empty
        if len(taxid_files) == 0:
            logger.error('No valid taxid files detected')
            return
        
        for database, taxid_file in taxid_files.items():
            taxer = taxer_dict[database](taxid_file, self.ranks, self.out_dir)
            taxer.taxing(chunksize, max_attempts)
            logger.info(f'Finished retrieving taxonomy data from {database} database. Saved to {taxer.tax_out}')
            self.out_files[database] = taxer.tax_out
