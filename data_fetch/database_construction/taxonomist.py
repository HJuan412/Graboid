#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:40:24 2022

@author: hernan

This script fetches the taxonomy data for the downloaded records
"""

#%% modules
from Bio import Entrez
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp

import numpy as np
import os
import pandas as pd
#%% functions
def tax_slicer(tax_list, chunksize = 500):
    # slice the list of taxids into chunks
    n_taxs = len(tax_list)
    for i in np.arange(0, n_taxs, chunksize):
        yield tax_list[i:i+chunksize]

def extract_tax_data(record):
    # generates taxonomy dictionaries ({rank:ScientificName/TaxID}) for the given records
    taxonomy = {}
    taxonomy_id = {}
    
    lineage = record['LineageEx'] + [{'TaxId':record['TaxId'], 'ScientificName':record['ScientificName'], 'Rank':record['Rank']}]
    for lin in lineage:
        taxonomy[lin['Rank']] = lin['ScientificName']
        taxonomy_id[lin['Rank']] = lin['TaxID']
    
    return taxonomy, taxonomy_id

#%% classes
class TaxonomistNCBI():
    def __init__(self, taxon, marker, in_file, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_file = in_file
        self.out_dir = out_dir
        self.set_ranks()
        self.warnings = []
        self.dl_warnings = []
        self.__read_tax_file()
        self.__make_tax_table()
        self.tax_out = f'{out_dir}/{taxon}_{marker}_NCBI.tax'
        self.taxid_out = f'{out_dir}/{taxon}_{marker}_NCBI.taxid'
    
    def __read_tax_file(self):
        # load acc:taxid list, generate reverse list to find accessions based on taxid, generate list of unique taxids to avoid redundant downloads
        tax_tab = pd.read_csv(self.in_file, index_col = 0)
        self.taxid_list = tax_tab[1]
        self.taxid_rev = pd.Series(index = self.taxid_list.values, data = self.taxid_list.index)
        self.uniq_taxs = self.taxid_list.unique()
        if len(self.taxid_list) == 0:
            self.warnings.append(f'WARNING: No entries found in the taxonomy file {self.in_file}')
    
    def __make_tax_table(self):
        # generate empty taxonomy tables tax & taxid (used to store the taxonomic IDs), guide tables link a taxid with its full taxonomy
        self.tax_table = pd.DataFrame(index = self.taxid_list.index, columns = self.ranks)
        self.taxid_table = pd.DataFrame(index = self.taxid_list.index, columns = self.ranks)
        self.guide_table = pd.DataFrame(index = self.uniq_tax, columns = self.ranks) # this will be used to store the taxonomic data and later distribute it to each record
        self.guideid_table = pd.DataFrame(index = self.uniq_tax, columns = self.ranks)

    def __update_guide(self, taxids, records):
        # extract data from a single record and updates the guide tables
        for taxid, record in zip(taxids, records):
            tax_dict, id_dict = extract_tax_data(record)
            self.guide_table.at[taxid] = tax_dict
            self.guideid_table.at[taxid] = id_dict

    def __fill_blanks(self):
        # fills missing records with the last known taxon name or id
        blanks = self.guide_table.isnull()
        for idx, rk in enumerate(self.ranks[1:]):
            blank_idx = blanks.loc[blanks[rk] == 1].index
            parent_rk = self.ranks[idx]
            self.guide_table.at[blank_idx, rk] = self.guide_table.loc[blank_idx, parent_rk]
            self.guideid_table.at[blank_idx, rk] = self.guideid_table.loc[blank_idx, parent_rk]
    
    def __update_tables(self):
        # write the final taxonomic tables using the complete guide tables
        for taxid in self.uniq_taxs.values:
            instances = self.taxid_rev.loc[taxid] # records with the given ID
            self.tax_table.at[instances] = self.guide_table.loc[taxid]
            self.taxid_table.at[instances] = self.guideid_table.loc[taxid]

    def __dl_tax_records(self, chunksize = 500):
        # attempts to download the taxonomic records in chunks of size chunksize
        chunks = tax_slicer(self.uniq_taxs, chunksize)
        for chunk in chunks:
            try:
                tax_handle = Entrez.efetch(db = 'taxonomy', id = chunk, retmode = 'xml')
                tax_records = Entrez.read(tax_handle)
                
                self.__update_guide(chunk, tax_records)
            except:
                self.dl_warnings.append(chunk)
    
    def __check_warnings(self):
        n_warns = len(self.dl_warnings)
        total = len(self.uniq_taxs)
        if n_warns > 0:
            self.warnings.append(f'WARNING: failed to retireve {n_warns} of {total} taxonomic records:')
            self.warnings.append('#NCBI')
            self.warnings += self.dl_warnings
            self.warnings.append('#NCBI')
    
    def __save_tables(self):
        self.tax_table.to_csv(self.tax_out)
        self.taxid_table.to_csv(self.taxid_out)

    def set_ranks(self, ranklist = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = ranklist
    
    def taxing(self, chunksize = 500):
        self.__dl_tax_records()
        self.__fill_blanks()
        self.__update_tables()
        self.__check_warnings()
        self.__save_tables()

class TaxonomistBOLD():
    def __init__(self, taxon, marker, seq_file, summ_file, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.seq_file = seq_file
        self.summ_file = summ_file
        self.out_dir = out_dir
        self.set_ranks()
        self.warnings = []
        self.__get_accs()
        self.tax_out = f'{out_dir}/{taxon}_{marker}_BOLD.tax'
        self.taxid_out = f'{out_dir}/{taxon}_{marker}_BOLD.taxid'
    
    def __get_accs(self):
        # get a list of accessions of the downloaded records
        accessions = []
        with open(self.seq_file, 'r') as handle:
            for header, _ in sfp(handle):
                accessions.append(header.split('|')[0])
        self.accessions = accessions
        if len(self.accessions) == 0:
            self.warnings.append(f'WARNING: Provided BOLD sequence file {self.seq_file} is empty')
    
    def __get_tax_tab(self):
        # use the acc list and rank list to extract relevant taxon names and ids
        summ_tab = pd.read_csv(self.summ_file, sep = '\t', index_col = 0)
        col_headers = []
        for rk in self.ranks:
            col_headers.append(f'{rk}_taxID')
            col_headers.append(f'{rk}_name')
        self.tax_tab = summ_tab.loc[self.accessions, col_headers]
    
    def __fill_blanks(self):
        # TODO
        pass

    def __save_tables(self):
        # build and save tax and taxid tables
        name_headers = [f'{rk}_name' for rk in self.ranks]
        taxid_headers = [f'{rk}_taxID' for rk in self.ranks]
        tax_tab = self.tax_tab.loc[:,name_headers]
        taxid_tab = self.tax_tab.loc[:,taxid_headers]
        tax_tab.to_csv(self.tax_out)
        taxid_tab.to_csv(self.taxid_out)

    def set_ranks(self, ranklist = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = ranklist
    
    def taxing(self, chunksize = None):
        # chunksize kept for compatibility
        self.__get_tax_tab()
        self.__save_tables()

class Taxonomist():
    def __init__(self, taxon, marker, databases, in_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.in_dir = in_dir
        self.warn_dir = warn_dir
        self.prefix = f'{in_dir}/{taxon}_{marker}'
        self.warnings = []
        self.__set_taxers()
        # TODO: set out_dir
    
    def __set_taxers(self):
        # TODO: homoheneize taxonomist constructors, tidy up these conditionals
        taxers = {}
        if 'BOLD' in self.databases:
            taxers['BOLD'] = TaxonomistBOLD(self.taxon, self.marker, f'{self.prefix}_BOLD.tmp', f'{self.prefix}_BOLD.summ', self.in_dir)
        if 'NCBI' in self.databases:
            taxers['NCBI'] = TaxonomistNCBI(self.taxon, self.marker, f'{self.prefix}_NCBI.tax', self.in_dir)
        
        if len(taxers) == 0:
            self.warnings.append('WARNING: No databases found by the taxonomy reconstructor')
            
    def set_ranks(self, ranklist = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = ranklist
        for taxr in self.taxers.values():
            taxr.set_ranks(ranklist)
    
    def __check_warnings(self):
        for taxr in self.taxers.values():
            self.warnings += taxr.warnings
        
    def __save_warnings(self):
        if len(self.warnings) > 0:
            with open(f'{self.warn_dir}/warnings.taxr', 'w') as handle:
                handle.write('\n'.join(self.warnings))

    def taxing(self, chunksize = 500):
        for taxr in self.taxers.values():
            taxr.taxing(chunksize)
        self.__check_warnings()
        self.__save_warnings()
        # TODO: build acc2taxid file