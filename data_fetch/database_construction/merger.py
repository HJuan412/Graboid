#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
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
            self.tax_out = f'{self.out_dir}/header.tax'
            self.taxguide_out = f'{self.out_dir}/header.taxguide'
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

    # def __get_files(self):
    #     seqfiles = {}
    #     taxfiles = {}
    #     taxidfiles = {}
        
    #     for db in self.databases:
    #         seqfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.seqtmp'
    #         taxfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.tax'
    #         taxidfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.taxid'
        
    #     self.seqfiles = seqfiles
    #     self.taxfiles = taxfiles
    #     self.taxidfiles = taxidfiles
    
    # def __set_marker_vars(self, marker_vars):
    #     # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
    #     self.marker_vars = list(marker_vars)

    # def build_list(self):
    #     post_lister = lister.PostLister(self.taxon, self.marker, self.in_dir, self.in_dir, self.warn_dir)
    #     post_lister.detect()
    #     post_lister.compare()
    #     self.acc_tab = post_lister.filtered
        
    #     if len(post_lister.warnings) > 0:
    #         self.warnings.append('Post Lister warnings:')
    #         self.warnings += post_lister.warnings
    #         self.warnings.append('---')
        
    #     if self.acc_tab is None:
    #         self.warnings.append('WARNING: Failed to create accession list, check Post Lister warnings')
    
    # def build_seqfile(self):
    #     records = []

    #     for db in self.databases:
    #         acclist = set(self.acc_tab.loc[self.acc_tab['Database'] == db, 'Accession'].tolist())
    #         with open(self.seqfiles[db], 'r') as handle:
    #             for record in SeqIO.parse(handle, 'fasta'):
    #                 if record.id in acclist:
    #                     records.append(record)
        
    #     with open(self.out_file, 'w') as out_handle:
    #         SeqIO.write(records, out_handle, 'fasta')
        
    #     #TODO: handle old files
    
    def merge_taxons(self):
        mtax = MergerTax(self.taxfiles)
        mtax.merge_taxons(self.tax_out, self.taxguide_out)
    
    def merge(self, seqfiles, taxfiles):
        self.generate_outfiles()
        self.merge_seqs()
        self.merge_taxons()

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
        
    # def __load_files(self):
    #     self.ncbi_tax = pd.read_csv(self.taxfiles['NCBI'], index_col = 0)
    #     self.ncbi_taxid = pd.read_csv(self.taxidfiles['NCBI'], index_col = 0)
    #     self.bold_tax = pd.read_csv(self.taxfiles['BOLD'], index_col = 0)
    #     self.bold_taxid = pd.read_csv(self.taxidfiles['BOLD'], index_col = 0)
    #     self.ranks = self.ncbi_tax.columns.tolist()
    #     self.__adjust_headers()

    # def __adjust_headers(self):
    #     rank_name = {f'{rk}_name':rk for rk in self.ranks}
    #     rank_taxID = {f'{rk}_taxID':f'{rk}_id' for rk in self.ranks}
    #     ncbi_taxid = {rk:f'{rk}_id' for rk in self.ranks}
    #     self.bold_tax.rename(columns = rank_name, inplace = True)
    #     self.bold_taxid.rename(columns = rank_taxID, inplace = True)
    #     self.ncbi_taxid.rename(columns = ncbi_taxid, inplace = True)
    
    # def build_taxid_flats(self):
    #     ncbi_taxid_flat = pd.DataFrame(columns = ['Taxid', 'Rank'])
    #     bold_taxid_flat = pd.DataFrame(columns = ['Taxid', 'Rank'])
        
    #     for rk in self.ranks[::-1]:
    #         rk_id = f'{rk}_id'
    #         merged_ncbi = pd.concat([self.ncbi_tax[rk], self.ncbi_taxid[rk_id]], axis = 1)
    #         merged_bold = pd.concat([self.bold_tax[rk], self.bold_taxid[rk_id]], axis = 1)
            
    #         for taxon, subtab in merged_ncbi.groupby(rk):
    #             tax_id = subtab.iloc[0, 1]
    #             ncbi_taxid_flat.at[taxon] = [tax_id, rk]
    #         for taxon, subtab in merged_bold.groupby(rk):
    #             tax_id = subtab.iloc[0, 1]
    #             bold_taxid_flat.at[taxon] = [tax_id, rk]
        
    #     self.ncbi_flat = ncbi_taxid_flat
    #     self.bold_flat = bold_taxid_flat

    # def correct_BOLD(self):
    #     bold_corrected = self.bold_taxid.copy()
    #     for rk in self.ranks:
    #         ncbi_flat_sub = self.ncbi_flat.loc[self.ncbi_flat['Rank'] == rk]
    #         bold_flat_sub = self.bold_flat.loc[self.bold_flat['Rank'] == rk]
    #         intersection = ncbi_flat_sub.index.intersection(bold_flat_sub.index)
    #         to_correct = self.ncbi_flat.loc[intersection]
    
    #         for tax, row in to_correct.iterrows():
    #             rank = row['Rank']
    #             tax_id = row['Taxid']
    #             idx = self.bold_tax.loc[self.bold_tax[rank] == tax].index
    #             bold_corrected.at[idx, f'{rank}_id'] = tax_id
    #     self.bold_taxid = bold_corrected
    
    # def build_taxid_guide(self, tax_tab, taxid_tab):
    #     taxid_guide = pd.DataFrame(columns = ['TaxName', 'Rank', 'ParentTaxId'])
        
    #     for parent_idx, rk in enumerate(self.ranks[1:]):
    #         parent_rk = self.ranks[parent_idx]
    #         for tax, subtab in tax_tab.groupby(rk):
    #             idx = subtab.index[0]
    #             taxid = taxid_tab.loc[idx, f'{rk}_id']
    #             parent_taxid = taxid_tab.loc[idx, f'{parent_rk}_id']
    #             taxid_guide.at[taxid] = [tax, rk, parent_taxid]
    #     taxid_guide.to_csv(self.taxid_guidefile)
        
    # def build_taxfiles(self, acc_tab):
    #     # NCBI
    #     ncbi_accs = set(acc_tab.loc[acc_tab['Database'] == 'NCBI', 'Accession'].tolist())
    #     intersect_accs = ncbi_accs.intersection(set(self.ncbi_tax.index)) # some of the accessions in acc_tab may not be in ncbi_tax
    #     ncbi_tax_subtab = self.ncbi_tax.loc[intersect_accs]
    #     ncbi_taxid_subtab = self.ncbi_taxid.loc[intersect_accs]
        
    #     # BOLD
    #     bold_accs = set(acc_tab.loc[acc_tab['Database'] == 'BOLD', 'Accession'].tolist())
    #     bold_tax_subtab = self.bold_tax.loc[bold_accs]
    #     bold_taxid_subtab = self.bold_taxid.loc[bold_accs]
        
    #     tax_file = pd.concat([ncbi_tax_subtab, bold_tax_subtab])
    #     taxid_file = pd.concat([ncbi_taxid_subtab, bold_taxid_subtab])
        
    #     self.build_taxid_guide(tax_file, taxid_file)
    #     tax_file.to_csv(self.tax_outfile)
    #     taxid_file.to_csv(self.taxid_outfile)