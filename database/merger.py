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
import numpy as np
import pandas as pd
import re

#%% setup logger
logger = logging.getLogger('Graboid.database.merger')

#%% variables
valid_databases = ['BOLD', 'NCBI']
#%% functions
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

def expand_guide(guide, ranks):
    # expand the taxonomy guide to allow for easy ancestor lookup
    # builds table of index = TaxID, columns = ranks
    expanded = pd.DataFrame(index = guide.index, columns = ranks)

    for rk in ranks[::-1]:
        # start from the lowest ranks up
        # locate all records for the given rank and add them to the expanded table
        rk_tab = guide.loc[guide.Rank == rk]
        expanded.loc[rk_tab.index, rk] = rk_tab.index
        # fill parent tax for taxa at the given rank
        filled_rows = expanded.loc[expanded[rk].notna(), rk]
        parents = guide.loc[filled_rows.values, 'parentTaxID'].values
        parent_tab = guide.loc[parents].reset_index().set_index(filled_rows.index)
        for parent_rk, parent_subtab in parent_tab.groupby('Rank'):
            expanded.loc[parent_subtab.index, parent_rk] = parent_subtab.TaxID
    
    # clear orphans
    for rk_idx, rk in enumerate(ranks):
        empty_tax = expanded.loc[expanded[rk].isna()].index
        expanded.loc[empty_tax, ranks[rk_idx + 1:]] = np.nan
    return expanded.astype(float)

def tax_summary(guide_tab, tax_tab, ranks):
    # builds a human readable dataframe containing the rank, parent taxon and record count for every taxon present in the database
    summary_tab = guide_tab.copy()
    # count records per taxon
    summary_tab['Records'] = tax_tab.TaxID.value_counts()
    summary_tab.Records = summary_tab.Records.fillna(0)
    rv_ranks = ranks[::-1]
    for rk in rv_ranks[:-1]:
        for parentID, subtab in summary_tab.loc[summary_tab.Rank == rk].groupby('parentTaxID'):
            summary_tab.loc[parentID, 'Records'] += subtab.Records.sum()
    # translate taxIDs to human readable
    tr_dict = {idx:name for idx, name in summary_tab.SciName.iteritems()}
    summary_tab.parentTaxID = summary_tab.parentTaxID.replace(tr_dict)
    summary_tab = summary_tab.rename(columns = {'SciName':'Taxon', 'parentTaxID':'Parent'})
    summary_tab.Records = summary_tab.Records.astype(int)
    return summary_tab.set_index('Taxon')

# Add lineage codes to the tax guide (used to sort taxa by lineage)
def make_codes(codes, name_array):
    # takes a list of started codes for a group of taxa, along with a name array (2d np.array containing the taxa names as a matrix)
    # finds the shortest abreviated code for each taxon
    made_codes = codes.copy()
    code_idxs = np.arange(len(made_codes))
    
    prev_uniq = '' # used when a sequence ends before being separated
    for idx, col in enumerate(name_array.T):
        # if a given position is invariable among the taxa, it is ommited
        uniq_chars = np.unique(col)
        if len(uniq_chars) == 1 and uniq_chars[0] != '0':
            prev_uniq = uniq_chars[0]
            # all rows have the same VALID value
            continue
        if '0' in uniq_chars:
            # a name ended before diverging from the rest (shouldn't happen), update all codes with the previous character and continue
            made_codes = [mc + prev_uniq for mc in made_codes]
        for u_char in uniq_chars[uniq_chars != '0']:
            # group names among those sharing the same character at a variable position
            char_loc = col == u_char
            char_idxs = code_idxs[char_loc]
            char_codes = [made_codes[i] + u_char for i in char_idxs]
            if char_loc.sum() > 1:
                # multiple taxa share the value, update their codes and recursively keep building until they are separated
                for ch_idx, new_code in zip(char_idxs, make_codes(char_codes, name_array[char_idxs][:, idx:])):
                    made_codes[ch_idx] = new_code
            else:
                # a single character has this value, no need to continue recursion, update its code and move to the next names
                made_codes[char_idxs[0]] += u_char
        return made_codes

def build_codes(name_array):
    # build the abreviated taxonomic codes for the taxa given in name_array
    codes = name_array[:,0].tolist() # always include the first character of the names in the code (codes must be a list, not a numpy array)
    code_idxs = np.arange(len(codes)) # use this to keep track of code locations in the codes list
    uniq_chars = np.unique(codes)
    
    for u_char in uniq_chars:
        # split the taxa by their initial letters
        char_loc = name_array[:, 0] == u_char
        char_idxs = code_idxs[char_loc] # positions of names beginning with u_char
        char_sub_array = name_array[char_loc][:,1:]
        char_codes = [codes[ch_idx] for ch_idx in char_idxs] # get the codes beginning with u_char
        if char_loc.sum() > 1:
            # only extend when multiple taxa start with the same letter
            for ch_idx, new_code in zip(char_idxs, make_codes(char_codes, char_sub_array)):
                codes[ch_idx] = new_code
    return codes

def guide_postproc(guide, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
    # update the taxonomy guide with the lineage codes
    guide['LinCode'] = ''
    # set base rank code(s)
    base_rk = ranks[0]
    base_subtab = guide.loc[guide.Rank == base_rk, 'SciName'].sort_values()
    base_names = base_subtab.tolist()
    longest = max([len(name) for name in base_names])
    base_name_array = np.array([list(name.ljust(longest, '0')) for name in base_names])
    guide.loc[base_subtab.index, 'LinCode'] = build_codes(base_name_array)
    # set the codes for the sub taxa
    for rk in ranks[1:]:
        # locate rows for the current taxon (tarting from the second rank because base is already set)
        rk_subtab = guide.loc[guide.Rank == rk].sort_values('SciName')
        # group taxa in rank by parent taxon
        for parent, pr_subtab in rk_subtab.groupby('parentTaxID'):
            parent_code = guide.loc[parent, 'LinCode'] # codes for each taxon at this rank should be preceded by their parent's code
            names = pr_subtab['SciName'].tolist()
            longest = max([len(name) for name in names])
            name_array = np.array([list(name.ljust(longest, '0')) for name in names]) # turn names list into a 2d numpy array
            # build codes and update guide
            codes = [f'{parent_code}.{code}' for code in build_codes(name_array)]
            guide.loc[pr_subtab.index, 'LinCode'] = codes
#%% classes
class Merger():
    def __init__(self, out_dir, ranks=None):
        self.out_dir = out_dir
        self.set_ranks(ranks)
        self.nseqs = 0
    
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
            self.expguide_out = header + '.guideexp'
            self.taxsumm_out = header + '.taxsumm'
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
    
    def merge(self, seqfiles, taxfiles, guidefiles):
        self.get_files(seqfiles, taxfiles, guidefiles)
        self.generate_outfiles()
        self.merge_seqs()
        self.mtax = MergerTax(taxfiles, guidefiles)
        self.mtax.merge(self.taxguide_out, self.tax_out, self.ranks) # self.ranks is used by guide_postproc
        self.ext_guide = expand_guide(self.mtax.merged_guide, self.ranks)
        self.ext_guide.to_csv(self.expguide_out)
        tax_summary(self.mtax.merged_guide, self.mtax.merged_tax, self.ranks).to_csv(self.taxsumm_out)
        logger.info(f'Stored expanded taxonomic guide to {self.expguide_out}')
        logger.info(f'Stored taxonomy summary to {self.taxsumm_out}')
    
    @property
    def rank_counts(self):
        return self.mtax.rank_counts
    
    @property
    def tax_tab(self):
        return self.mtax.merged_tax
    

class MergerTax():
    def __init__(self, tax_files, guide_files):
        self.tax_files = tax_files
        self.guide_files = guide_files
        self.NCBI = 'NCBI' in tax_files.keys()
        self.load_files()
    
    def load_files(self):
        tax_tabs = {}
        guide_tabs = {}
        for database, tax_file in self.tax_files.items():
            tax_tabs[database] = pd.read_csv(tax_file, index_col = 0)
            guide_tabs[database] = pd.read_csv(self.guide_files[database], index_col=0)
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
        logger.info(f'Merged taxonomy codes using {lead} as a guide. Generated index contains {len(self.merged_guide)} unique taxa.')
    
    def merge_tax_tabs(self):
        self.merged_tax = pd.concat(list(self.tax_tabs.values()))
    
    def merge(self, guide_out, tax_out, ranks):
        self.merge_guides()
        self.merge_tax_tabs()
        guide_postproc(self.merged_guide, ranks) # update lineage codes
        self.merged_guide.to_csv(guide_out)
        self.merged_tax.to_csv(tax_out)
        logger.info(f'Stored merged taxonomic codes to {guide_out}')
        logger.info(f'Stored merged record taxonomies to {tax_out}')
        
        # get data
        self.rank_counts = self.merged_guide.Rank.value_counts()
