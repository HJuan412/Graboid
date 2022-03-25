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

import bold_marker_vars
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
        taxonomy_id[lin['Rank']] = lin['TaxId']
    
    return taxonomy, taxonomy_id

def build_BOLD_guidetab(df, cols, ranks):
    guide_tab = pd.DataFrame(columns = cols)
    rank = ranks[-1]
    
    for taxon, subtab in df.groupby(f'{rank}_name'):
        guide_tab = pd.concat([guide_tab, subtab.iloc[0].to_frame().transpose()], ignore_index = True)
    
    if len(ranks) > 1:
        sub_df = df.loc[df[f'{rank}_name'].isna()]
        guide_tab = pd.concat([guide_tab, build_BOLD_guidetab(sub_df, cols, ranks[:-1])], ignore_index = True)
    
    return guide_tab
#%% classes
class TaxonomistNCBI():
    def __init__(self, taxon, marker, in_dir, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.set_ranks()
        self.warnings = []
        self.dl_warnings = []
        self.in_file = f'{self.in_dir}/{taxon}_{marker}_NCBI.taxtmp'
        self.tax_out = f'{out_dir}/{taxon}_{marker}_NCBI.tax'
        self.taxid_out = f'{out_dir}/{taxon}_{marker}_NCBI.taxid'
    
    def __read_infile(self):
        if not os.path.isfile(self.in_file):
            self.warnings.append(f'WARNING: NCBI summary file {self.in_file} not found')
            self.taxid_list = None
            return
        
        taxid_list = pd.read_csv(self.in_file, index_col = 0, header = None)
        
        if len(taxid_list) == 0:
            self.warnings.append(f'WARNING: NCBI summary file {self.in_file} is empty')
            self.taxid_list = None
            return
        self.taxid_list = taxid_list[1]
    
    def __process_taxtab(self):
        if self.taxid_list is None:
            self.taxid_rev = None
            self.uniq_taxs = None
            return
        self.taxid_rev = pd.Series(index = self.taxid_list.values, data = self.taxid_list.index)
        self.uniq_taxs = self.taxid_list.unique()
        
        
    # def __read_tax_file(self):
    #     # load acc:taxid list, generate reverse list to find accessions based on taxid, generate list of unique taxids to avoid redundant downloads
    #     tax_tab = pd.read_csv(self.in_file, header = None, index_col = 0)
    #     self.taxid_list = tax_tab[1]
    #     self.taxid_rev = pd.Series(index = self.taxid_list.values, data = self.taxid_list.index)
    #     self.uniq_taxs = self.taxid_list.unique()
    #     if len(self.taxid_list) == 0:
    #         self.warnings.append(f'WARNING: No entries found in the taxonomy file {self.in_file}')
    
    def __make_tax_tables(self):
        # generate empty taxonomy tables tax & taxid (used to store the taxonomic IDs), guide tables link a taxid with its full taxonomy
        self.tax_table = pd.DataFrame(index = self.taxid_list.index, columns = self.ranks)
        self.taxid_table = pd.DataFrame(index = self.taxid_list.index, columns = self.ranks)
        self.guide_table = pd.DataFrame(index = self.uniq_taxs, columns = self.ranks) # this will be used to store the taxonomic data and later distribute it to each record
        self.guideid_table = pd.DataFrame(index = self.uniq_taxs, columns = self.ranks)

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
        for taxid in self.uniq_taxs:
            instances = self.taxid_rev.loc[taxid] # records with the given ID
            self.tax_table.at[instances] = self.guide_table.loc[taxid].values
            self.taxid_table.at[instances] = self.guideid_table.loc[taxid].values

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
        # get input files and prepare to recieve data
        self.__read_infile()
        self.__process_taxtab()
        self.__make_tax_tables()
        
        if self.taxid_list is None:
            return

        self.__dl_tax_records()
        self.__fill_blanks()
        self.__update_tables()
        self.__check_warnings()
        self.__save_tables()

class TaxonomistBOLD():
    def __init__(self, taxon, marker, in_dir, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.set_ranks()
        self.__set_marker_vars(bold_marker_vars.marker_vars[marker])
        self.warnings = []
        self.in_file = f'{self.in_dir}/{taxon}_{marker}_BOLD.tmp'
        self.seq_out = f'{out_dir}/{taxon}_{marker}_BOLD.seqtmp'
        self.tax_out = f'{out_dir}/{taxon}_{marker}_BOLD.tax'
        self.taxid_out = f'{out_dir}/{taxon}_{marker}_BOLD.taxid'
    
    def __set_marker_vars(self, marker_vars):
        # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
        self.marker_vars = list(marker_vars)

    def __read_infile(self):
        if not os.path.isfile(self.in_file):
            self.warnings.append(f'WARNING: BOLD summary file {self.in_file} not found')
            self.bold_tab = None
            return
        
        bold_tab = pd.read_csv(self.in_file, sep = '\t', encoding = 'latin-1', index_col = 0, low_memory = False) # latin-1 to parse BOLD files

        if len(bold_tab) == 0:
            self.warnings.append(f'WARNING: BOLD summary file {self.in_file} is empty')
            self.bold_tab = None
            return
        bold_tab = bold_tab.loc[bold_tab['markercode'].isin(self.marker_vars)]
        self.bold_tab = bold_tab

    # def __get_accs(self):
    #     # get a list of accessions of the downloaded records
    #     accessions = []
    #     with open(self.seq_file, 'r') as handle:
    #         for header, _ in sfp(handle):
    #             accessions.append(header.split('|')[0])
    #     self.accessions = accessions
    #     if len(self.accessions) == 0:
    #         self.warnings.append(f'WARNING: Provided BOLD sequence file {self.seq_file} is empty')
    
    def __get_tax_tab(self):
        # use the acc list and rank list to extract relevant taxon names and ids
        cols = []
        for rk in self.ranks:
            cols.append(f'{rk}_name')
            cols.append(f'{rk}_taxID')
        self.tax_tab = self.bold_tab.loc[:, cols]
    
    def __fill_blanks(self):
        for idx, rk in enumerate(self.ranks[1:]):
            blank_idx = self.tax_tab.loc[self.tax_tab[f'{rk}_name'].isna()].index
            parent_rk = self.ranks[idx]
            self.tax_tab.at[blank_idx, f'{rk}_name'] = self.tax_tab.loc[blank_idx, f'{parent_rk}_name']
            self.tax_tab.at[blank_idx, f'{rk}_taxID'] = self.tax_tab.loc[blank_idx, f'{parent_rk}_taxID']

    def __build_guide_tabs(self):
        cols = []
        for rk in self.ranks:
            cols.append(f'{rk}_name')
            cols.append(f'{rk}_taxID')

        self.guide_table = build_BOLD_guidetab(self.tax_tab, cols, self.ranks)
    
    def __get_seqs(self):
        records = []
        
        for idx, row in self.bold_tab.iterrows():
            head_list = [str(idx), str(row['genbank_accession']), str(row['species_name'])]
            header = '>' + ' '.join(head_list)
            seq = row['nucleotides']
            records.append('\n'.join([header, seq]))
        
        with open(self.seq_out, 'w') as handle:
            handle.write('\n'.join(records))

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
        self.__read_infile()
        self.__get_tax_tab()
        self.__fill_blanks()
        self.__get_seqs()
        self.__save_tables()

class Taxonomist():
    def __init__(self, taxon, marker, databases, in_dir, out_dir, warn_dir):
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.prefix = f'{in_dir}/{taxon}_{marker}'
        self.warnings = []
        self.__set_taxers()
    
    def __set_taxers(self):
        taxers = {'NCBI': TaxonomistNCBI(self.taxon, self.marker, self.in_dir, self.out_dir),
                  'BOLD': TaxonomistBOLD(self.taxon, self.marker, self.in_dir, self.out_dir)}
        self.taxers = taxers
            
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