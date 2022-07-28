#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
import bold_marker_vars
import logging
import lister
import pandas as pd

#%% setup logger
logger = logging.getLogger('database_logger.merger')

#%% variables
valid_databases = ['BOLD', 'NCBI']
#%% functions
def detect_files(file_list):
    # TODO: NOTE: this function behaves exactly like lister.detect_summ(), may need more analogs to it in the future, may define a generic one in a tools module
    # given a list of taxID files, identify the database to which each belongs
    # works under the assumption that there is a single file per database
    files = {}
    for file in file_list:
        database = file.split('_')[-1].split('.')[0]
        if database in valid_databases:
            files[database] = file
    return files

def flatten_taxtab(tax_tab):
    # generates a two column tax from the tax tab, containing the taxID, rank and parent taxID of each taxon
    # tax name kept as index
    # assumes the tax tab has 2*n_taxes columns
    
    ncols = len(tax_tab.columns)
    stripped_cols = []
    # walk over the table, two columns at a time
    for col in range(0, ncols, 2):
        # extract scientific name and taxID columns for a given taxon
        rank = tax_tab.columns[col]
        lone_col = tax_tab.iloc[:,[col, col+1]]
        lone_col.rename(columns = {colname:newname for colname, newname in zip(lone_col.columns, ('SciName', 'taxID'))})
        # add rank column
        lone_col['Rank'] = rank

        if col > 0:
            # if we are not at the basal rank, get parent rank taxIDs
            parent_col = tax_tab[col]
            lone_col['Parent_TaxID'] = parent_col
        else:
            # set parent taxIDs as 0
            lone_col['Parent_TaxID'] = 0
        
        lone_col.set_index(lone_col.columns[0], inplace=True)
        stripped_cols.append(lone_col)
    return pd.concat(stripped_cols)
#%% classes
class Merger():
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.seq_out = None
        self.acc_out = None
        self.tax_out = None
        self.taxguide_out = None
    
    def get_files(self, seqfiles, taxfiles):
        self.seqfiles = seqfiles
        self.taxfiles = taxfiles
        # seqfiles and taxiles should be dictionaries with database:filename key:value pairs
        if isinstance(seqfiles, list):
            self.seqfiles = detect_files(seqfiles)
        if isinstance(taxfiles, list):
            self.taxfiles = detect_files(taxfiles)
        
        self.generate_outfiles()
    
    def generate_outfiles(self):
        for sample in self.seqfiles.values():
            header = sample.split('/')[-1].split('_')[:-1]
            header = '_'.join(header)
            self.seq_out = f'{self.out_dir}/{header}.fasta'
            self.acc_out = f'{self.out_dir}/{header}.acclist'
            self.tax_out = f'{self.out_dir}/{header}.tax'
            self.taxguide_out = f'{self.out_dir}/{header}.taxguide'
            break
    
    def merge_seqs(self):
        # reads given sequence files and extracts accessions
        # generates a merged fasta file and accession table
        records = []
        acc_tabs = []
        
        for database, seqfile in self.seqfiles:
            id_list = []
            with open(seqfile, 'r') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    id_list.append(record.id)
                    records.append(record)
            
            if len(id_list) == 0:
                logger.warning('No records found in file {seqfile}')
                
            acc_subtab = pd.DataFrame(id_list, columns = ['Accession'])
            acc_subtab['Database'] = database
            acc_tabs.append(acc_subtab)
        
        acc_tab = pd.concat(acc_tabs)
        
        # save merged seqs to self.seq_out and accession table to self.acc_out
        with open(self.seq_out, 'w') as seq_handle:
            SeqIO.write(records, seq_handle, 'fasta')
        acc_tab.to_csv(self.acc_out)
    
    def merge_taxons(self):
        mtax = MergerTax(self.taxfiles)
        mtax.merge_taxons(self.tax_out, self.taxguide_out)
    
    def merge(self, seqfiles, taxfiles):
        self.get_files(seqfiles, taxfiles)
        self.generate_outfiles()
        self.merge_seqs()
        self.merge_taxons()
    
    def merge_from_fasta(self, seqfile, taxfile):
        # Used when a fasta file was provided, generate acclist and taxguide
        header = seqfile.split('/')[-1].split('.')[0]
        self.acc_out = f'{self.out_dir}/{header}.acclist'
        self.taxguide_out = f'{self.out_dir}/{header}.taxguide'
        # generate acc list
        acc_list = []
        with open(seqfile, 'r') as seq_handle:
            for acc, seq in sfp(seq_handle):
                acc_list.append(acc)
        
        acc_tab = pd.DataFrame(acc_list, columns = ['Accession'])
        acc_tab['Database'] = 'NCBI'
        acc_tab.to_csv(self.acc_out)
        # generate taxguide
        tax_tab = pd.read_csv(taxfile, index_col = 0)
        guide_tab = flatten_taxtab(tax_tab)
        guide_tab.to_csv(self.taxguide_out)

class MergerTax():
    def __init__(self, tax_files):
        self.tax_files = tax_files
        self.NCBI = 'NCBI' in tax_files.keys()
        self.load_files()
        self.build_tax_guides()
    
    def load_files(self):
        tax_tabs = {}
        for database, tax_file in self.tax_files.items():
            tax_tabs[database] = pd.read_csv(tax_file, index_col = 0)
        self.tax_tabs = tax_tabs
    
    def build_tax_guides(self):
        tax_guides = {}
        for database, tax_tab in self.tax_tabs.items():
            tax_guides[database] = flatten_taxtab(tax_tab)
        
        self.tax_guides = tax_guides
    
    def unify_taxids(self):
        # unifies taxonomic codes used by different databases
        # check that NCBI is present (take it as guide if it is)
        if self.NCBI:
            guide_db = 'NCBI'
        else:
            guide_db = list(self.tax_guides.keys())[0]
        guide_tab = self.tax_guides[guide_db]
            
        for db, tab in self.tax_guides.items():
            if db == guide_db:
                continue
            # get taxons with a common scientific names between databases
            intersect = guide_tab.index.intersection(tab.index)
            diff = tab.index.difference(guide_tab.index)

            if len(intersect) > 0:
                # correct_tab indicates which taxids to replace and what to replace them with
                correct_tab = pd.concat([guide_tab.loc[intersect, ['Rank', 'taxID']],
                                         tab.loc[intersect, 'taxID']], axis = 1)
                correct_tab.columns = ['Rank', 'guide', 'tab']
                # tax_table to modify
                tax_tab = self.tax_tabs[db]
                # for each rank, look for the taxons to fix
                for rk, rk_subtab in correct_tab.groupby('Rank'):
                    for idx, row in rk_subtab.iterrows():
                        # replace taxID values
                        tax_tab.at[tax_tab[f'{rk}_id'] == row['tab']] = row['guide']
                # incorporate non redundant taxons to the guide_tab
                guide_tab = pd.concat([guide_tab, tab.loc[diff]])
        
        self.guide_tab = guide_tab
    
    def merge_taxons(self, tax_out, taxguide_out):
        self.unify_taxids()
        merged_taxons = pd.concat(self.tax_tabs.items())
        
        merged_taxons.to_csv(tax_out)
        self.guide_tab.to_csv(taxguide_out)
