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
import re

#%% setup logger
logger = logging.getLogger('Graboid.database.taxonomist')

#%% variables
valid_databases = ['BOLD', 'NCBI']

#%% functions
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

def unfold_lineage(record):
    # extracts data from a taxonomic record
    # generates lineage dict : {rank:[TaxID, SciName]}
    unfolded_lineage = {record['Rank']:[record['TaxId'], record['ScientificName']]}
    unfolded_lineage.update({lin['Rank']:[lin['TaxId'], lin['ScientificName']] for lin in record['LineageEx']})
    return unfolded_lineage

def unfold_records(records):
    # generates a dictionary with the unfolded taxonomy of every retrieved record (including the rank of the queried TaxID)
    unfolded = {}
    for record in records:
        unfolded[record['TaxId']] = unfold_lineage(record)
    return unfolded

#%% classes
class Taxer:
    def generate_outfiles(self):
        self.tax_out = re.sub('.*/', self.out_dir + '/', re.sub('\..*', '.tax', self.taxid_file))
    
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
    def __init__(self, taxid_file, ranks, out_dir, warn_dir):
        self.taxid_file = taxid_file
        self.ranks = ranks
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.generate_outfiles()
        self.make_tax_tables()
        self.tax_records = {}
        self.failed = []
        
    def read_taxid_file(self):
        # reads the acc:taxid table
        self.taxid_tab = None
        self.taxid_reverse = None
        self.uniq_taxs = None
        
        # index : accessions, values : taxids
        self.taxid_tab = pd.read_csv(self.taxid_file, index_col = 0)
        
        # generates a warning if taxid_tab is empty
        if len(self.taxid_tab) == 0:
            raise Exception(f'Summary file {self.in_file} is empty')

        # get a reversed taxid list and a list of unique taxes
        self.taxid_reverse = pd.Series(index = self.taxid_tab.values.flatten(), data = self.taxid_tab.index)
        self.uniq_taxs = np.unique(self.taxid_tab)
    
    def dl_tax_records(self, tax_list, chunksize=500, max_attempts=3):
        # attempts to download the taxonomic records in chunks of size chunksize
        chunks = tax_slicer(tax_list, chunksize)
        n_chunks = int(np.ceil(len(tax_list)/chunksize))
        retrieved = []
        failed = []
        for idx, chunk in enumerate(chunks):
            print(f'Retrieving taxonomy. Chunk {idx + 1} of {n_chunks}')
            tax_records = []
            for attempt in range(max_attempts):
                try:
                    tax_handle = Entrez.efetch(db = 'taxonomy', id = chunk, retmode = 'xml')
                    tax_records = Entrez.read(tax_handle)
                    break
                except IOError:
                    logger.warning(f'Interrupted taxing due to conection error. {max_attempts - attempt - 1} attempts remaining')
                    continue
                except http.client.HTTPException:
                    logger.warning(f'Interrupted taxing due to bad file.  {max_attempts - attempt - 1} attempts remaining')
                    continue
            if len(tax_records) != len(chunk):
                failed += list(chunk)
                continue
            retrieved += tax_records        
        self.failed = failed
        self.tax_records.update(unfold_records(retrieved))
    
    def retry_dl(self, max_attempts=3):
        # if some taxids couldn't be downloaded, rety up to max_attempts times
        attempt = 1
        while attempt <= max_attempts and len(self.failed) > 0:
            logger.warning(f'Attempting to download {len(self.failed)} failed taxIDs of {len(self.uniq_taxs)} from NCBI. Attempt {attempt} of {max_attempts}...')
            self.dl_tax_records(self.failed)
            attempt += 1
        if len(self.failed) > 0:
            logger.warning(f'Failed to download {len(self.failed)} records from NCBI after {max_attempts} retries. A list of the failed records was saved as {self.warn_dir}/failed_tax.ncbi')
            with open(self.warn_dir + '/failed_tax.ncbi', 'w') as handle:
                handle.write('\n'.join(self.failed))
    
    def build_guide(self):
        # generates a dataframe containing the complete taxonomy of every retrieved record, including only the specified ranks
        # generated dictionary {TaxID:[SciName, rank, parentTaxID]}
        taxes = {}
        for record in self.tax_records.values():
            prev_tx = 0
            for rk in self.ranks:
                try:
                    rk_taxon = record[rk]
                    taxes.update({rk_taxon[0]:[rk_taxon[1], rk, prev_tx]})
                    prev_tx = rk_taxon[0]
                except KeyError:
                    # rank not found in lineage
                    continue
        self.guide = pd.DataFrame.from_dict(taxes, orient='index', columns='SciName Rank parentTaxID'.split()).rename_axis('TaxID') # leave Rank column with uppercase to enable use of  dot notation (.Rank)
        # TODO: uncomment this line if you decide to use numeric codes for the ranks
        # self.guide.Rank = self.guide.Rank.replace({rk:idx for idx, rk in enumerate(self.ranks)})
    
    def build_tax_tab(self):
        tax_tab = self.taxid_tab.copy()
        # see if any of the sequence records is missing its taxonomic code in the guide table
        # this can happen if the record's tax ID is not among the specified ranks
        taxid_values = set(self.taxid_tab.TaxID.values())
        missing = taxid_values.difference(self.guide.index)
        # any missing taxID is replaced by the last known one for its retrieved record
        to_replace = {}
        for ms in missing:
            to_replace[ms] = None
            missing_record = self.tax_records[ms]
            for rk in self.ranks[::-1]:
                try:
                    to_replace[ms] = missing_record[rk][0]
                except KeyError:
                    continue
        # replace missing taxonomy codes and add ranks
        tax_tab.TaxID = tax_tab.TaxID.replace(to_replace)
        tax_tab['Rank'] = self.guide.loc[tax_tab.TaxID, 'Rank']
        self.tax_tab = tax_tab
    
    def taxing(self, chunksize=500, max_attempts=3):
        if self.taxid_tab is None:
            return

        self.dl_tax_records(self.uniq_taxs, chunksize)
        self.retry_dl(max_attempts)
        self.build_guide()
        self.build_tax_tab()
        
        self.tax_tab.to_csv(self.tax_out)

class TaxonomistBOLD(Taxer):
    # generates the taxomoic tables for the records downloaded from BOLD
    def __init__(self, taxid_file, ranks, out_dir, warn_dir):
        self.taxid_file = taxid_file
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.generate_outfiles()
        self.__set_marker_vars()
        self.ranks = ranks
        
    def __set_marker_vars(self):
        # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
        marker = re.sub('.*/', '', self.taxid_file).split('_')[1]
        self.marker_vars = list(marker)
    
    def read_taxid_file(self):
        bold_tab = pd.read_csv(self.taxid_file, index_col = 0)
        if len(bold_tab) == 0:
            self.bold_tab = None
            # interrupt execution and let constructor know
            raise Exception(f'Summary file {self.in_file} is empty')
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

class Taxonomist:
    # class attribute dictionary containing usable taxonomist tools
    taxer_dict = {'BOLD':TaxonomistBOLD,
                  'NCBI':TaxonomistNCBI}
    def __init__(self, out_dir, warn_dir, ranks=None):
        self.taxid_files = {}
        self.set_ranks(ranks)
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        
        self.out_files = {}
    
    def set_ranks(self, ranks=None):
        if ranks is None:
            self.ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        self.ranks = ranks
    
    def taxing(self, taxid_files, chunksize=500, max_attempts=3):
        # taxid_files : dict with {database:taxid_file}
        # taxid_file : pandas dataframe with index = accessions, columns = taxid if NCBI or full taxonomy if BOLD
        # check that summ_file container is not empty
        if len(taxid_files) == 0:
            raise Exception('No valid taxid files detected')
        
        for database, taxid_file in taxid_files.items():
            taxer = self.taxer_dict[database](taxid_file, self.ranks, self.out_dir, self.warn_dir)
            try:
                taxer.read_taxid_file()
            except Exception as ex:
                logger.error(ex)
                raise
            taxer.taxing(chunksize, max_attempts)
            logger.info(f'Finished retrieving taxonomy data from {database} database. Saved to {taxer.tax_out}')
            self.out_files[database] = taxer.tax_out
