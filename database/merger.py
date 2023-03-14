#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
import logging
import pandas as pd
import re

#%% setup logger
logger = logging.getLogger('Graboid.database.merger')

#%% variables
valid_databases = ['BOLD', 'NCBI']
#%% functions
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

def reenganch(diff_table, origin_table, guide_table, ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']):
    # use this function to locate the base taxonomy of non redundant taxa before incorporating them into the guide
    # diff_table is the table of new taxa to be added to the guide table
    # origin_table is the table from which diff_table originates from
    # guide_table is the table that imposes its taxonomy codes on the rest
    # ranks is a list of ranks, genius
    
    # get the ranks present in the diff table (this is because we want to check the highest ranks first)
    ranks_in_diff = [rk for rk in ranks if rk in diff_table['rank'].values]
    # copy the guide tab because we're going to update it and we don't want to things up with the original
    guide = guide_table.copy()
    
    # new table
    reenganched = diff_table.copy()
    # start by the highest ranks
    for rk in ranks_in_diff:
        rk_tab = diff_table.loc[diff_table['rank'] == rk]
        for tax, row in rk_tab.iterrows():
            # get the taxon's parent, locate it's code in the guide table
            parentID = row.parent_taxID
            parent = origin_table.loc[origin_table.taxID == parentID].index[0]
            parent_new_id = guide.loc[parent].taxID
            # update parent code in reenganched table
            reenganched.at[tax, 'parent_taxID'] = parent_new_id
            # add new taxon to the copy of the guide table (in case it had children taxa in the diff_tab)
            guide.at[tax] = reenganched.loc[tax]
    return reenganched

def update_parents(diff_tab, diff_guide, lead_guide):
    # locate the parent ranks of non redundant taxa when merging tax guides
    # diff_tab is the table containing new taxa to be incorporated
    # diff_guide is the guide table from which diff_tab originates
    # lead_guide is the guide table from which the new codes will be taken
    fixed_tab = diff_tab.copy()
    # locate parent taxa for all elements in diff_tab
    parents = diff_guide.loc[diff_tab.parentTaxID].reset_index().set_index('SciName')
    parent_names = parents.index.unique()
    # retrieve the parent TaxIDs from the lead table (when possible)
    parent_codes = lead_guide.loc[lead_guide.SciName.isin(parent_names), 'SciName'].reset_index().set_index('SciName')
    # replace parent TaxIDs
    parents.loc[parent_codes.index, 'TaxID'] = parent_codes.TaxID
    fixed_tab.parentTaxID = parents.TaxID.values
    return fixed_tab

def dissect_guide(guide, current_rank, rank_n, rank_dict):
    # generate a rank dict (assign an ordered index to each taxonomic rank in the retrieved taxonomy)
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
    def __init__(self, out_dir, ranks=None):
        self.out_dir = out_dir
        self.set_ranks(ranks)
        self.nseqs = 0
    
    @property
    def base_rank(self):
        return self.mtax.base_rank
    @property
    def base_taxa(self):
        return self.mtax.base_taxa
    @property
    def rank_counts(self):
        return self.mtax.rank_counts
    
    def get_files(self, seqfiles, taxfiles, guidefiles):
        self.seqfiles = seqfiles
        self.taxfiles = taxfiles
        self.guide_files = guidefiles
        # seqfiles and taxfiles should be dictionaries with database:filename key:value pairs        
        self.generate_outfiles()
    
    def generate_outfiles(self):
        for sample in self.seqfiles.values():
            header = re.sub('.*/', self.out_dir + '/', re.sub('__.*', '', sample))
            self.seq_out = header + '.fasta'
            self.acc_out = header + '.acclist'
            self.tax_out = header +  '.tax'
            self.taxguide_out = header + '.taxguide'
            self.valid_rows_out = header + '.rows'
            break
        
    def set_ranks(self, ranks=None):
        if ranks is None:
            self.ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        else:
            self.ranks = ranks
        
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
                continue
                
            acc_subtab = pd.DataFrame(id_list, columns = ['Accession'])
            acc_subtab['Database'] = database
            acc_tabs.append(acc_subtab)
            
        acc_tab = pd.concat(acc_tabs)
        
        # save merged seqs to self.seq_out and accession table to self.acc_out
        with open(self.seq_out, 'w') as seq_handle:
            SeqIO.write(records, seq_handle, 'fasta')
        acc_tab.to_csv(self.acc_out)
        logger.info(f'Merged {len(records)} sequence records to {self.seq_out}')
        # update sequence count
        self.nseqs = len(records)
    
    def merge(self, seqfiles, taxfiles):
        self.get_files(seqfiles, taxfiles)
        self.generate_outfiles()
        self.merge_seqs()
        self.mtax = MergerTax(self.taxfiles, self.ranks)
        self.mtax.merge_taxons(self.tax_out, self.taxguide_out, self.valid_rows_out)

class MergerTax():
    def __init__(self, tax_files, guide_files, ranks):
        self.tax_files = tax_files
        self.guide_files = guide_files
        self.ranks = ranks
        # self.NCBI = 'NCBI' in tax_files.keys()
        # self.load_files()
        # self.build_tax_guides()
    
    # def load_files(self):
    #     tax_tabs = {}
    #     guide_tabs = {}
    #     for database, tax_file in self.tax_files.items():
    #         tax_tabs[database] = pd.read_csv(tax_file, index_col = 0)
    #         guide_tabs[database] = pd.read_csv(self.guide_files[database], index_col=0)
    #     self.tax_tabs = tax_tabs
    #     self.guide_tabs = guide_tabs
    
    def set_tabs(self, tax_tabs, guide_tabs):
        self.tax_tabs = tax_tabs
        self.guide_tabs = guide_tabs
        
    def merge_guides(self):
        # select a lead table to impose its taxIDs over the rest
        lead = 'NCBI'
        if not self.NCBI:
            lead = list(self.guide_tabs.keys())[0]
        lead_guide = self.guide_tabs[lead].reset_index().set_index('SciName')
        # check remaining guides, update them using the lead's codes
        for db, guide in self.guide_tabs.items():
            # skip the lead guide
            if db == lead:
                continue
            foll_guide = guide.reset_index().set_index('SciName')
            # get overlapping taxa and non redundant taxa
            overlap = lead_guide.index.intersection(foll_guide.index)
            difference = foll_guide.index.difference(lead_guide.index)
            # replace updated values in the corresponding taxonomy table
            old_idx = foll_guide.loc[overlap, 'TaxID'].values
            new_idx = lead_guide.loc[overlap, 'TaxID'].values
            repl_dict = {old:new for old, new in zip(old_idx, new_idx)}
            # foll_guide.TaxID = foll_guide.TaxID.replace(repl_dict)
            self.tax_tabs[db].TaxID = self.tax_tabs[db].TaxID.replace(repl_dict)
            # update lead_guide with the new taxa (update parents of the new taxa first)
            diff_tab = foll_guide.loc[difference]
            fixed_tab = update_parents(diff_tab, guide, self.guide_tabs[lead])
            lead_guide = pd.concat([lead_guide, fixed_tab])
        self.merged_guide = lead_guide.reset_index().set_index('TaxID')
    
    def merge_tax_tabs(self):
        self.merged_tax = pd.concat(list(self.tax_tabs.values()))
        
    # def build_tax_guides(self):
    #     tax_guides = {}
    #     for database, tax_tab in self.tax_tabs.items():
    #         tax_guides[database] = flatten_taxtab(tax_tab, self.ranks)
        
    #     self.tax_guides = tax_guides
    
    # def unify_taxids(self):
    #     # unifies taxonomic codes used by different databases
    #     # check that NCBI is present (take it as guide if it is)
    #     if self.NCBI:
    #         guide_db = 'NCBI'
    #     else:
    #         guide_db = list(self.tax_guides.keys())[0]
    #     guide_tab = self.tax_guides[guide_db]
            
    #     for db, tab in self.tax_guides.items():
    #         if db == guide_db:
    #             continue
    #         # get taxons with a common scientific names between databases
    #         intersect = guide_tab.index.intersection(tab.index)
    #         diff = tab.index.difference(guide_tab.index)

    #         if len(intersect) > 0:
    #             # correct_tab indicates which taxids to replace and what to replace them with
    #             correct_tab = pd.concat([guide_tab.loc[intersect, ['rank', 'taxID']],
    #                                      tab.loc[intersect, 'taxID']], axis = 1)
    #             correct_tab.columns = ['rank', 'guide', 'tab']
    #             # tax_table to modify
    #             tax_tab = self.tax_tabs[db]
    #             # for each rank, look for the taxons to fix
    #             for rk, rk_subtab in correct_tab.groupby('rank'):
    #                 rk_vals = tax_tab[rk+'_id'].values
    #                 correction_vals = rk_subtab[['guide', 'tab']].values
    #                 for pair in correction_vals:
    #                     rk_vals[rk_vals == pair[1]] = pair[0]
    #                 tax_tab[rk+'_id'] = rk_vals
                        
    #         # incorporate non redundant taxons to the guide_tab
    #         non_r_taxa = reenganch(tab.loc[diff], tab, guide_tab, self.ranks)
    #         guide_tab = pd.concat([guide_tab, non_r_taxa])
        
    #     self.guide_tab = guide_tab
    
    # def build_rank_dict(self):
    #     root = set(self.guide_tab.parent_taxID).difference(set(self.guide_tab.taxID))
    #     root_rank = self.guide_tab.loc[self.guide_tab.parent_taxID.isin(root), 'rank'].iloc[0]
    #     base_rank = self.ranks[0]
    #     self.base_taxa = self.guide_tab.loc[self.guide_tab['rank'] == base_rank].index.tolist()
    #     self.base_rank = base_rank
    #     self.rank_counts = self.guide_tab.value_counts('rank').to_dict()
    #     self.rank_dict = {}
        
    #     dissect_guide(self.guide_tab, root_rank, 0, self.rank_dict)
    
    # def get_valid_rows(self, tax_table):
    #     # generate a dictionary containing the valid rows present for each rank in the retrieved taxonomies
    #     # process tax_guide
    #     # assumption (only species need this correction)
    #     # chriterion, discard all species with 3 or more words, discard all species with a parent taxon outside the genus rank
    
    #     # locate all species with triple names
    #     guide_spp = self.guide_tab.loc[self.guide_tab['rank'] == 'species'].reset_index()
    #     species = [sp.split() for sp in guide_spp.SciName.tolist()]
    #     triple_spp = guide_spp.loc[[idx for idx, sp in enumerate(species) if len(sp) > 2], 'taxID'].values
    #     # locate 'orphaned' species
    #     species_parents = guide_spp.parent_taxID.values
    #     non_genus_parents = self.guide_tab.loc[(self.guide_tab.taxID.isin(species_parents)) & (self.guide_tab['rank'] != 'genus'), 'taxID'].values
    #     orphan_spp = guide_spp.loc[guide_spp.parent_taxID.isin(non_genus_parents), 'taxID'].values
    
    #     invalid_spp = set(np.concatenate([triple_spp, orphan_spp]))
        
    #     tax_table.reset_index(inplace=True)
    
    #     valid_rows = {}
    
    #     for rank, rk_subtab in self.guide_tab.groupby('rank'):
    #         rank_ids = set(rk_subtab.taxID)
    #         rank_rows = tax_table.loc[tax_table[f'{rank}_id'].isin(rank_ids)].index.to_numpy()
    #         valid_rows[rank] = rank_rows
    #     valid_rows['species'] = tax_table.loc[~tax_table.species_id.isin(invalid_spp)].index.to_numpy()
    #     self.valid_rows = valid_rows
        
    # def merge_taxons(self, tax_out, taxguide_out, valid_rows_out):
    #     self.unify_taxids()
    #     self.build_rank_dict()
    #     merged_taxons = pd.concat(self.tax_tabs.values())
    #     self.get_valid_rows(merged_taxons)
    #     merged_taxons.to_csv(tax_out, index=False)
    #     logger.info(f'Unified taxonomies stored to {tax_out}')
    #     # drop duplicated records (if any)
    #     merged_taxons.loc[np.invert(merged_taxons.index.duplicated())]
    #     self.guide_tab.to_csv(taxguide_out)
    #     with open(valid_rows_out, 'wb') as valid_rows_handle:
    #         pickle.dump(self.valid_rows, valid_rows_handle)
