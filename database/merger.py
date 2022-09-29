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
import logging
import numpy as np
import pandas as pd
import pickle

#%% setup logger
logger = logging.getLogger('Graboid.database.merger')

#%% variables
valid_databases = ['BOLD', 'NCBI']
#%% functions
def detect_files(file_list):
    # NOTE: this function behaves exactly like lister.detect_summ(), may need more analogs to it in the future, may define a generic one in a tools module
    # given a list of taxID files, identify the database to which each belongs
    # works under the assumption that there is a single file per database
    files = {}
    for file in file_list:
        database = file.split('_')[-1].split('.')[0]
        if database in valid_databases:
            files[database] = file
    return files

def flatten_taxtab(tax_tab, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
    # generates a two column tax from the tax tab, containing the taxID, rank and parent taxID of each taxon
    # tax name kept as index
    tax_tab = tax_tab.copy()
    tax_tab[0] = 0
    
    flattened = []
    parents = [0] + [rk+'_id' for rk in ranks[:-1]]
    
    for rank, parent in zip(ranks, parents):
        stripped_cols = tax_tab[[rank, rank+'_id', parent]].copy()
        stripped_cols.rename(columns = {rank:'SciName', rank+'_id':'taxID', parent:'parent_taxID'}, inplace=True)
        stripped_cols['rank'] = rank
        for taxid, subtab in stripped_cols.groupby('taxID'):
            flattened.append(subtab.iloc[[0]])
    flattened_tab = pd.concat(flattened)
    flattened_tab.taxID = flattened_tab.taxID.astype(int)
    flattened_tab.parent_taxID = flattened_tab.parent_taxID.astype(int)
    flattened_tab = flattened_tab.loc[flattened_tab.taxID != flattened_tab.parent_taxID]
    flattened_tab.set_index('SciName', inplace=True)
    return flattened_tab

def dissect_guide(guide, current_rank, rank_n, rank_dict):
    # generate a rank dict (assign an irdered index to each taxonomic rank in the retrieved taxonomy)
    # update dict
    rank_dict.update({current_rank:rank_n})
    # get child ranks
    sub_guide = guide.loc[guide['rank'] == current_rank]
    child_taxa = set(sub_guide.taxID).intersection(set(guide.parent_taxID))
    # stop condition, parent rank has no children
    if len(child_taxa) == 0:
        return
    # update rank dict
    child_rank = guide.loc[guide.parent_taxID.isin(child_taxa), 'rank'].iloc[0]
    dissect_guide(guide, child_rank, rank_n+1, rank_dict)
    
#%% classes
class Merger():
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.seq_out = None
        self.acc_out = None
        self.tax_out = None
        self.taxguide_out = None
        self.set_ranks()
    
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
            self.rank_dict_out = f'{self.out_dir}/{header}.ranks'
            self.valid_rows_out = f'{self.out_dir}/{header}.rows'
            break
        
    def set_ranks(self, ranklist=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.ranks = ranklist
        
    def merge_seqs(self):
        # reads given sequence files and extracts accessions
        # generates a merged fasta file and accession table
        records = []
        acc_tabs = []
        
        for database, seqfile in self.seqfiles.items():
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
        logger.info(f'Merged {len(records)} sequence records to {self.seq_out}')
    
    def merge_taxons(self):
        mtax = MergerTax(self.taxfiles, self.ranks)
        mtax.merge_taxons(self.tax_out, self.taxguide_out, self.rank_dict_out, self.valid_rows_out)
    
    def merge(self, seqfiles, taxfiles):
        self.get_files(seqfiles, taxfiles)
        self.generate_outfiles()
        self.merge_seqs()
        self.merge_taxons()
    
    def merge_from_fasta(self, seqfile, taxfile):
        # Used when a fasta file was provided, generate acclist and taxguide
        header = seqfile.split('/')[-1].split('.')[0]
        self.acc_out = f'{self.out_dir}/{header}.acclist'
        self.tax_out = f'{self.out_dir}/{header}.tax'
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
        guide_tab = flatten_taxtab(tax_tab, self.ranks)
        guide_tab.to_csv(self.taxguide_out)
        logger.info(f'Generated files {header}.acclist and {header.taxguide} in directory {self.out_dir}')

class MergerTax():
    def __init__(self, tax_files, ranks):
        self.tax_files = tax_files
        self.ranks = ranks
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
            tax_guides[database] = flatten_taxtab(tax_tab, self.ranks)
        
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
                correct_tab = pd.concat([guide_tab.loc[intersect, ['rank', 'taxID']],
                                         tab.loc[intersect, 'taxID']], axis = 1)
                correct_tab.columns = ['rank', 'guide', 'tab']
                # tax_table to modify
                tax_tab = self.tax_tabs[db]
                # for each rank, look for the taxons to fix
                for rk, rk_subtab in correct_tab.groupby('rank'):
                    rk_vals = tax_tab[rk+'_id'].values
                    correction_vals = rk_subtab[['guide', 'tab']].values
                    for pair in correction_vals:
                        rk_vals[rk_vals == pair[1]] = pair[0]
                    tax_tab[rk+'_id'] = rk_vals
                    # for idx, row in rk_subtab.iterrows():
                    #     # replace taxID values
                    #     tax_tab.at[tax_tab[rk+'_id'] == row['tab']] = row['guide']
                        
            # incorporate non redundant taxons to the guide_tab
            guide_tab = pd.concat([guide_tab, tab.loc[diff]])
        
        self.guide_tab = guide_tab
    
    def build_rank_dict(self):
        root = set(self.guide_tab.parent_taxID).difference(set(self.guide_tab.taxID))
        root_rank = self.guide_tab.loc[self.guide_tab.parent_taxID.isin(root), 'rank'].iloc[0]
        self.rank_dict = {}
        
        dissect_guide(self.guide_tab, root_rank, 0, self.rank_dict)
    
    def get_valid_rows(self, tax_table):
        # generate a dictionary containing the valid rows present for each rank in the retrieved taxonomies
        # process tax_guide
        # assumption (only species need this correction)
        # chriterion, discard all species with 3 or more words, discard all species with a parent taxon outside the genus rank
    
        # locate all species with triple names
        guide_spp = self.guide_tab.loc[self.guide_tab['rank'] == 'species'].reset_index()
        species = [sp.split() for sp in guide_spp.SciName.tolist()]
        triple_spp = guide_spp.loc[[idx for idx, sp in enumerate(species) if len(sp) > 2], 'taxID'].values
        # locate 'orphaned' species
        species_parents = guide_spp.parent_taxID.values
        non_genus_parents = self.guide_tab.loc[(self.guide_tab.taxID.isin(species_parents)) & (self.guide_tab['rank'] != 'genus'), 'taxID'].values
        orphan_spp = guide_spp.loc[guide_spp.parent_taxID.isin(non_genus_parents), 'taxID'].values
    
        invalid_spp = set(np.concatenate([triple_spp, orphan_spp]))
        
        tax_table.reset_index(inplace=True)
    
        valid_rows = {}
    
        for rank, rk_subtab in self.guide_tab.groupby('rank'):
            rank_ids = set(rk_subtab.taxID)
            rank_rows = tax_table.loc[tax_table[f'{rank}_id'].isin(rank_ids)].index.to_numpy()
            valid_rows[rank] = rank_rows
        valid_rows['species'] = tax_table.loc[~tax_table.species_id.isin(invalid_spp)].index.to_numpy()
        self.valid_rows = valid_rows
        
    def merge_taxons(self, tax_out, taxguide_out, rank_dict_out, valid_rows_out):
        self.unify_taxids()
        self.build_rank_dict()
        merged_taxons = pd.concat(self.tax_tabs.values())
        self.get_valid_rows(merged_taxons)
        merged_taxons.to_csv(tax_out)
        logger.info(f'Unified taxonomies stored to {tax_out}')
        # drop duplicated records (if any)
        merged_taxons.loc[np.invert(merged_taxons.index.duplicated())]
        self.guide_tab.to_csv(taxguide_out)
        with open(rank_dict_out, 'wb') as rank_dict_handle:
            pickle.dump(self.rank_dict, rank_dict_handle)
        with open(valid_rows_out, 'wb') as valid_rows_handle:
            pickle.dump(self.valid_rows, valid_rows_handle)
