#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
import lister
import pandas as pd

#%% classes
class Merger():
    def __init__(self, taxon, marker, databases, in_dir, out_dir, warn_dir, old_file = None):
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.old_file = old_file
        self.warnings = []
        self.out_file = f'{out_dir}/{taxon}_{marker}.fasta'
        self.__get_files()
        self.__rm_old_seqs()
    
    def __get_files(self):
        seqfiles = {}
        taxfiles = {}
        taxidfiles = {}
        
        for db in self.databases:
            seqfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.seqtmp'
            taxfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.tax'
            taxidfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.taxid'
        
        self.seqfiles = seqfiles
        self.taxfiles = taxfiles
        self.taxidfiles = taxidfiles
    
    def build_list(self):
        post_lister = lister.PostLister(self.taxon, self.marker, self.in_dir, self.in_dir, self.warn_dir)
        # TODO: handle set_marker_vars
        post_lister.detect()
        post_lister.compare()
        self.acc_tab = post_lister.filtered
        # TODO: manage post_lister warnings
    
    def build_seqfile(self):
        records = []

        for db in self.databases:
            acclist = set(self.acc_tab.loc[self.acc_tab['Database'] == db, 'Accession'].tolist())
            with open(self.seqfiles[db], 'r') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    if record.id in acclist:
                        records.append(record)
        
        with open(self.out_file, 'w') as out_handle:
            SeqIO.write(records, out_handle, 'fasta')
        
        #TODO: handle old files
    
    def merge_taxons(self):
        mtax = MergerTax(self.taxon, self.marker, self.databases, self.in_dir, self.out_dir, self.warn_dir, self.old_file)
        mtax.build_taxid_flats()
        mtax.correct_BOLD()
        mtax.build_taxfiles(self.acc_tab)

class MergerTax():
    def __init__(self, taxon, marker, databases, in_dir, out_dir, warn_dir, old_file = None):
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.old_file = old_file
        self.tax_outfile = f'{out_dir}/{taxon}_{marker}.tax'
        self.taxid_outfile = f'{out_dir}/{taxon}_{marker}.taxid'
        self.__get_files()
        self.__load_files()
    
    def __get_files(self):
        taxfiles = {}
        taxidfiles = {}
        
        for db in self.databases:
            taxfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.tax'
            taxidfiles[db] = f'{self.in_dir}/{self.taxon}_{self.marker}_{db}.taxid'
        
        self.taxfiles = taxfiles
        self.taxidfiles = taxidfiles
    
    def __load_files(self):
        self.ncbi_tax = pd.read_csv(self.taxfiles['NCBI'], index_col = 0)
        self.ncbi_taxid = pd.read_csv(self.taxidfiles['NCBI'], index_col = 0)
        self.bold_tax = pd.read_csv(self.taxfiles['BOLD'], index_col = 0)
        self.bold_taxid = pd.read_csv(self.taxidfiles['BOLD'], index_col = 0)
        self.ranks = self.ncbi_tax.columns.tolist()
        self.__adjust_headers()

    def __adjust_headers(self):
        rank_name = {f'{rk}_name':rk for rk in self.ranks}
        rank_taxID = {f'{rk}_taxID':f'{rk}_id' for rk in self.ranks}
        ncbi_taxid = {rk:f'{rk}_id' for rk in self.ranks}
        self.bold_tax.rename(columns = rank_name, inplace = True)
        self.bold_taxid.rename(columns = rank_taxID, inplace = True)
        self.ncbi_taxid.rename(columns = ncbi_taxid, inplace = True)
    
    def build_taxid_flats(self):
        ncbi_taxid_flat = pd.DataFrame(columns = ['Taxid', 'Rank'])
        bold_taxid_flat = pd.DataFrame(columns = ['Taxid', 'Rank'])
        
        for rk in self.ranks[::-1]:
            rk_id = f'{rk}_id'
            merged_ncbi = pd.concat([self.ncbi_tax[rk], self.ncbi_taxid[rk_id]], axis = 1)
            merged_bold = pd.concat([self.bold_tax[rk], self.bold_taxid[rk_id]], axis = 1)
            
            for taxon, subtab in merged_ncbi.groupby(rk):
                tax_id = subtab.iloc[0, 1]
                ncbi_taxid_flat.at[taxon] = [tax_id, rk]
            for taxon, subtab in merged_bold.groupby(rk):
                tax_id = subtab.iloc[0, 1]
                bold_taxid_flat.at[taxon] = [tax_id, rk]
        
        self.ncbi_flat = ncbi_taxid_flat
        self.bold_flat = bold_taxid_flat

    def correct_BOLD(self):
        bold_corrected = self.bold_taxid.copy()
        
        for rk in self.ranks:
            ncbi_flat_sub = self.ncbi_flat.loc[self.ncbi_flat['Rank'] == rk]
            bold_flat_sub = self.bold_flat.loc[self.bold_flat['Rank'] == rk]
            intersection = ncbi_flat_sub.index.intersection(bold_flat_sub.index)
            to_correct = self.ncbi_flat.loc[intersection]
            print(rk, to_correct)
            
    
            for tax, row in to_correct.iterrows():
                rank = row['Rank']
                tax_id = row['Taxid']
                idx = self.bold_tax.loc[self.bold_tax[rank] == tax].index
                bold_corrected.at[idx, f'{rank}_id'] = tax_id
        
        self.bold_taxid = bold_corrected
    
    def build_taxfiles(self, acc_tab):
        # NCBI
        ncbi_accs = set(acc_tab.loc[acc_tab['Database'] == 'NCBI', 'Accession'].tolist())
        ncbi_tax_subtab = self.ncbi_tax.loc[ncbi_accs]
        ncbi_taxid_subtab = self.ncbi_taxid.loc[ncbi_accs]
        
        # BOLD
        bold_accs = set(acc_tab.loc[acc_tab['Database'] == 'BOLD', 'Accession'].tolist())
        bold_tax_subtab = self.bold_tax.loc[bold_accs]
        bold_taxid_subtab = self.bold_taxid.loc[bold_accs]
        
        tax_file = pd.concat([ncbi_tax_subtab, bold_tax_subtab])
        taxid_file = pd.concat([ncbi_taxid_subtab, bold_taxid_subtab])
        
        tax_file.to_csv(self.tax_outfile)
        taxid_file.to_csv(self.taxid_outfile)