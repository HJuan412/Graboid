#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:33:36 2021

@author: hernan
This script generates accession list tables from the summary files generated by db_survey
"""

#%% libraries
from database import bold_marker_vars
import logging
import pandas as pd
import os
import re

#%% set logger
logger = logging.getLogger('Graboid.database.lister')

#%% functions
# data loading
# using these readfuncs makes it easier to incorporate new databases in the future
def read_BOLD_summ(summ_file):
    # extract a list of accessions from a BOLD summary
    bold_tab = pd.read_csv(summ_file, sep = '\t', encoding = 'latin-1', dtype = str) # latin-1 to parse BOLD files
    # sometimes BOLD has multiple records, remove repeats
    duplicated = bold_tab.loc[bold_tab.sampleid.duplicated()]
    single = bold_tab.loc[~bold_tab.sampleid.duplicated()].reset_index(drop=False).set_index('sampleid', drop=False)
    for samp_id, cluster in duplicated.groupby('sampleid'):
        seqs = cluster.nucleotides.sum()
        single.loc[samp_id, 'nucleotides'] += seqs
    accs = single.sampleid.tolist()
    single.to_csv(summ_file, sep='\t', index=False)
    return accs

def read_NCBI_summ(summ_file):
    # extract a list of accessions from an NCBI or ENA summary
    ncbi_tab = pd.read_csv(summ_file, sep = '\t')
    accs = ncbi_tab.iloc[:,0].tolist()
    return accs

def read_summ(summ_file, database):
    readers = {'NCBI':read_NCBI_summ,
               'BOLD':read_BOLD_summ}
    accs = readers[database](summ_file)
    # if database == 'NCBI':
    #     ncbi_tab = pd.read_csv(summ_file, sep = '\t')
    #     accs = ncbi_tab.iloc[:,0].tolist()
    # elif database == 'BOLD':
    #     bold_tab = pd.read_csv(summ_file, sep = '\t', encoding = 'latin-1', dtype = str) # latin-1 to parse BOLD files
    #     accs = bold_tab['sampleid'].tolist()
    
    if len(accs) == 0:
        raise Exception(f'No records present in {summ_file}')
    return accs

# summary manipulation
def get_shortaccs_ver(acc_list):
    # split the accession code from the version number, return both
    # assign version number 1 whenever necesary
    splitaccs = [acc.split('.') for acc in acc_list]
    vers = [acc[-1] if len(acc) > 1 else '1' for acc in splitaccs]
    shortaccs = [acc.split(f'.{ver}')[0] for acc, ver in zip(acc_list, vers)]
    return shortaccs, vers

def build_acc_subtab(acc_list, database):
    # build table with index = short accession (no version number) and columns = accession, version number
    shortaccs, vers = get_shortaccs_ver(acc_list)
    acc_subtab = pd.DataFrame({'Accession': shortaccs, 'Version':vers}, index = shortaccs)
    acc_subtab['Database'] = database
    return acc_subtab

def clear_repeats(merged_tab):
    # locates and clears repeated records in the merged table, prioritizing NCBI records when possible
    # locate repeated indexes
    rep_idx = merged_tab.index[merged_tab.index.duplicated()]
    nbases = merged_tab['Database'].nunique()
    if len(rep_idx) == 0:
        # no repeats, continue with original table
        return merged_tab
    
    logger.info(f'Found {len(rep_idx)} repeated records between {nbases} databases')
    
    # handle repeats
    to_keep = []
    for idx in rep_idx:
        # filter repeated entries for the same record
        sub_merged = merged_tab.loc[idx]
        # try to use an NCBI record if available, otherwise pick a substitute
        lead = 'NCBI'
        if not 'NCBI' in sub_merged['Database'].values:
            lead = sub_merged['Database'].values[0]
        to_keep = sub_merged.loc[sub_merged['Database'] == lead].iloc[[0]] # double bracket lets iloc extract a single row as a dataframe
    
    # generate final table
    cropped_merged = merged_tab.drop(index = rep_idx)
    clear_merged = pd.concat([cropped_merged, pd.DataFrame(to_keep)])
    
    logger.info(f'Listed {len(clear_merged)} records between {nbases} databases')
    return clear_merged
#%% classes
class Lister:
    # this class compares summaries of multiple database and generates a consensus
    # prioritizes NCBI in case of conflict
    valid_databases = ['BOLD', 'NCBI']
    def __init__(self, out_dir):
        self.summ_files = {}
        self.out_file = None
        self.out_dir = out_dir
    
    def get_summ_files(self, summ_files):
        self.summ_files = {}
        for db, file in summ_files.items():
            if db in self.valid_databases:
                self.summ_files.update({db:file})
                sample_file = file
            else:
                logger.warning(f'Database {db} not valid. File {file} will be ignored')
        if len(self.summ_files) == 0:
            raise Exception('No valid survey files detected')
        # generate the output file
        self.out_file = re.sub('.*/', self.out_dir + '/', re.sub('__.*', '.acc', sample_file))
    
    def build_list(self, summ_files):
        # summ_files dict with {database:summ_file}
        # generates a consensus list of accession codes from the summary files
        # detect summary files
        self.get_summ_files(summ_files)
        
        # read summ_files
        acc_tabs = []
        for db, file in summ_files.items():
            try:
                db_accs = read_summ(file, db)
            except Exception as excp:
                logger.warning(excp)
                continue
            acc_subtab = build_acc_subtab(db_accs, db)
            acc_tabs.append(acc_subtab)
        
        if len(acc_tabs) == 0:
            logger.warning('No records located. Repeat survey step')
            raise Exception('No records located. Repeat survey step')
        # merge subtabs
        merged = pd.concat(acc_tabs)
        merged_clear = clear_repeats(merged)
        
        # store tab
        merged_clear.to_csv(self.out_file)
        logger.info(f'Accession list stored to {self.out_file}')

#%% These classes are no longer needed mut may be of use in the future
# class SummProcessor():
#     def __init__(self, taxon, marker, database, in_file, out_dir):
#         self.taxon = taxon
#         self.marker = marker
#         self.database = database
#         self.in_file = in_file
#         self.out_dir = out_dir
#         self.set_readfunc(database)
#         self.build_acc_subtab()
#         self.warnings = {0:[], 1:[]}
    
#     def set_readfunc(self, dbase):
#         # database determines how the file is read
#         if dbase == 'BOLD':
#             self.readfunc = read_BOLD_summ
#         elif dbase == 'NCBI':
#             self.readfunc = read_NCBI_summ
                
#     def get_shortaccs_ver(self):
#         # split the accession code from the version number, return both
#         accs = self.readfunc(self.in_file)
#         splitaccs = [acc.split('.') for acc in accs]
#         vers = [acc[-1] if len(acc) > 1 else '1' for acc in splitaccs]
#         shortaccs = [acc.split(f'.{ver}')[0] for acc, ver in zip(accs, vers)]
#         return shortaccs, vers
    
#     def build_acc_subtab(self):
#         #  build table with index = short accession (no version number) and columns = accession, version number
#         shortaccs, vers = self.get_shortaccs_ver()
#         acc_subtab = pd.DataFrame({'Accession': shortaccs, 'Version':vers}, index = shortaccs)
#         acc_subtab['Database'] = self.database
#         acc_subtab['Status'] = 'New'
#         self.acc_subtab = acc_subtab
        
#         if len(self.acc_subtab) == 0:
#             self.warnings[0].append(f'WARNING: No records in summary of the {self.database} database')

#     def compare_with_old(self, old_tab):
#         # see if any of the surveyed records already exist, if they do, see if there is a newer version
#         intersect = self.acc_subtab.index(old_tab.index)
#         compare = self.acc_subtab.loc[intersect, 'Version'] > self.old_tab.loc[intersect, 'Version']
#         to_drop = compare.loc[~compare].index # drop accs if the version is <= than the old record
#         to_replace = compare.loc[compare].index # replace records in old record if this version is greater
        
#         self.acc_subtab.drop(to_drop, inplace = True)
#         self.acc_subtab.at[to_replace, 'Status'] = 'Update'

#         if len(to_drop) > 0 and len(self.acc_subtab) == 0:
#             # all records dropped
#             self.warnings[1].append(f'NOTICE: No new or updated records found in the {self.database} database')

# class SummProcessor2:
#     def __init__(self, database, in_file, out_dir):
#         self.in_file = in_file
#         self.out_dir = out_dir
#         self.set_readfunc(database)
#         self.build_acc_subtab()
#         self.logger = logging.getLogger('database_logger.lister.SummProcessor')
    
#     def set_readfunc(self, dbase):
#         # database determines how the file is read
#         if dbase == 'BOLD':
#             self.readfunc = read_BOLD_summ
#         elif dbase == 'NCBI':
#             self.readfunc = read_NCBI_summ
                
#     def get_shortaccs_ver(self):
#         # split the accession code from the version number, return both
#         accs = self.readfunc(self.in_file)
#         splitaccs = [acc.split('.') for acc in accs]
#         vers = [acc[-1] if len(acc) > 1 else '1' for acc in splitaccs]
#         shortaccs = [acc.split(f'.{ver}')[0] for acc, ver in zip(accs, vers)]
#         return shortaccs, vers
    
#     def build_acc_subtab(self):
#         #  build table with index = short accession (no version number) and columns = accession, version number
#         shortaccs, vers = self.get_shortaccs_ver()
#         acc_subtab = pd.DataFrame({'Accession': shortaccs, 'Version':vers}, index = shortaccs)
#         acc_subtab['Database'] = self.database
#         acc_subtab['Status'] = 'New'
#         self.acc_subtab = acc_subtab
        
#         if len(self.acc_subtab) == 0:
#             self.logger.warning(f'No records present in summary of the {self.database} database')

# class PreLister():
#     # This class is used within the Surveyor class to compare the summary with the previous databse (if present)
#     def __init__(self, taxon, marker, in_file, out_dir, warn_dir, old_file = None):
#         self.taxon = taxon
#         self.marker = marker
#         self.in_file = in_file
#         self.out_dir = out_dir
#         self.warn_dir = warn_dir
#         self.old_file = old_file
#         self.warnings = ['']
#         self.out_file = f'{out_dir}/{taxon}_{marker}_NCBI.acc'
    
#     def __read_infile(self):
#         if not os.path.isfile(self.in_file):
#             self.warnings.append(f'WARNING: file {self.in_file} not found')
#             return
        
#         acc_dict = {}
#         with open(self.in_file, 'r') as handle:
#             for line in handle.read().splitlines():
#                 split_line = line.split('.')
#                 shortacc = split_line[0]
#                 version = split_line[1]
#                 acc_dict[shortacc] = (line, version)
        
#         if len(acc_dict) == 0:
#             self.warnings.append(f'WARNING: file {self.in_file} is empty')
#             self.acc_tab = None
#             return

#         self.acc_tab = pd.DataFrame.from_dict(acc_dict, orient = 'index', columns = ['Accession', 'Version'])
    
#     def __read_oldfile(self):
#         if self.old_file is None:
#             self.old_tab = None
#             return
        
#         if not os.path.isfile(self.old_file):
#             self.warnings.append(f'WARNING: file {self.in_file} not found')
#             return

#         if len(self.old_tab) == 0:
#             self.warnings.append(f'WARNING: file {self.old_file} is empty')
#             self.old_tab = None
#             return

#         self.old_tab = pd.read_csv(self.old_file, index_col = 0)
        
    
#     def merge_with_old(self):
#         if not self.acc_tab is None:
#             if not self.old_tab is None:
#                 intersect = self.acc_tab.index.intersection(self.old_tab.index)
#                 sub_acc = self.acc_tab.loc[intersect]
#                 old_acc = self.old_tab.loc[intersect]

#                 filtered = sub_acc.loc[old_acc['Version'] >= sub_acc['Version']].index # records to drop
#                 self.out_tab = self.acc_tab.drop(index = filtered)
#                 return
            
#             self.out_tab = self.acc_tab
#             return
        
#         self.out_tab = None
#         return
    
#     def save_tab(self):
#         if self.out_tab is None:
#             self.warnings.append('WARNING: No out file generated')
#             return
#         if len(self.out_tab) == 0:
#             self.warnings.append(f'WARNING: No new sequences in {self.in_file}')
#             return
        
#         self.out_tab.to_csv(self.out_file)
    
#     def save_warnings(self):
#         if len(self.warnings) > 1:
#             with open(f'{self.warn_dir}/warnings.surv', 'a') as warn_handle:
#                 warn_handle.write('\n'.join(self.warnings))
    
#     def pre_list(self):
#         self.__read_infile()
#         self.__read_oldfile()
#         self.merge_with_old()
#         self.save_tab()
#         self.save_warnings()

# class PostLister():
#     # This class generates the accession table with the sequences to be ADDED
#     def __init__(self, taxon, marker, in_dir, out_dir, warn_dir, old_file = None):
#         self.taxon = taxon
#         self.marker = marker
#         self.in_dir = in_dir
#         self.out_dir = out_dir
#         self.warn_dir = warn_dir
#         self.old_file = old_file
#         self.warnings = []
#         self.__set_marker_vars(bold_marker_vars.marker_vars[marker])
    
#     def __set_marker_vars(self, marker_vars):
#         # BOLD records may have variations of the marker name (18S/18s, COI-3P/COI-5P)
#         self.marker_vars = list(marker_vars)

#     def __make_boldaccs(self):
#         # this will hold the short_acc and version for each BOLD record
#         bold_dict = {}
#         for idx in self.bold_tab.index:
#             split_idx = idx.split('-')
#             bold_dict[split_idx[0]] = (idx, split_idx[1])
#         self.bold_accs = pd.DataFrame.from_dict(bold_dict, orient = 'index', columns = ['Accession', 'Version'])
    
#     def __make_boldgbdict(self):
#         # this will hold all BOLD entries with a genbank accession (used to check for overlap with the ncbi records)
#         bold_gb = {}
#         bold_subtab = self.bold_tab.loc[~self.bold_tab['genbank_accession'].isna(), 'genbank_accession']
#         for idx, acc in bold_subtab.iteritems():
#             split_idx = idx.split('-')
#             split_acc = acc.split('.')
#             bold_gb[split_acc[0]] = split_idx[0]
#         self.bold_gb = pd.Series(bold_gb)

#     def __detect_BOLD(self):
#         bold_file = f'{self.in_dir}/{self.taxon}_{self.marker}_BOLD.tmp'
#         if not os.path.isfile(bold_file):
#             self.warnings.append(f'WARNING: BOLD summary file {bold_file} not found')
#             self.bold_tab = None
#             self.bold_accs = None
#             return
        
#         bold_tab = pd.read_csv(bold_file, sep = '\t', encoding = 'latin-1', index_col = 0, low_memory = False) # latin-1 to parse BOLD files

#         if len(bold_tab) == 0:
#             self.warnings.append(f'WARNING: BOLD summary file {self.in_file} is empty')
#             self.bold_tab = None
#             return
#         bold_tab = bold_tab.loc[bold_tab['markercode'].isin(self.marker_vars)]
#         self.bold_tab = bold_tab
#         self.__make_boldaccs()
#         self.__make_boldgbdict()

#     def __detect_NCBI(self):
#         ncbi_file = f'{self.in_dir}/{self.taxon}_{self.marker}_NCBI.acc'
        
#         if not os.path.isfile(ncbi_file):
#             self.warnings.append(f'WARNING: NCBI summary file {ncbi_file} not found')
#             self.ncbi_tab = None
#             return
        
#         self.ncbi_tab = pd.read_csv(ncbi_file, index_col = 0)
    
#     def __read_oldfile(self):
#         if self.old_file is None:
#             self.old_tab = None
#             return
        
#         if not os.path.isfile(self.old_file):
#             self.warnings.append(f'WARNING: file {self.in_file} not found')
#             return

#         if len(self.old_tab) == 0:
#             self.warnings.append(f'WARNING: file {self.old_file} is empty')
#             self.old_tab = None
#             return

#         self.old_tab = pd.read_csv(self.old_file, index_col = 0)

#     def detect(self):
#         # read accession tables
#         self.__detect_BOLD()
#         self.__detect_NCBI()
#         self.__read_oldfile()
    
#     def __bold_old(self):
#         # compare bold_accs vs old_tab
#         self.filtered_bold = None
        
#         if self.bold_tab is None:
#             return

#         if not self.old_tab is None:
#             intersect = self.bold_accs.index.intersection(self.old_tab.index) 
#             if len(intersect) == 0:
#                 return
            
#             bold_acc = self.bold_accs.loc[intersect]
#             old_acc = self.old_tab.loc[intersect, ['Accession', 'Version']]
            
#             filtered = bold_acc.loc[bold_acc['Version'] > old_acc['Version']]
#             if len(filtered) == 0:
#                 return
#         else:
#             filtered = self.bold_accs.copy()
#         filtered['Database'] = 'BOLD'
#         self.filtered_bold = filtered
    
#     def __bold_ncbi(self):
#         # compare filtered_bold vs ncbi_tab
#         self.filtered = None
#         if self.ncbi_tab is None:
#             return

#         self.filtered = self.ncbi_tab.copy()
#         self.filtered['Database'] = 'NCBI'
        
#         if self.filtered_bold is None:
#             return
        
#         if len(self.bold_gb) == 0:
#             self.filtered = pd.concat([self.filtered, self.filtered_bold])
#             return
        
#         intersect = self.bold_gb.index.intersection(self.filtered.index)
#         intersect_idx = self.bold_gb.loc[intersect].values # get the index of intersecting values
#         filtered_bold = self.filtered_bold.drop(intersect_idx)
#         self.filtered = pd.concat([self.filtered, filtered_bold])

#     def compare(self):
#         self.__bold_old()
#         self.__bold_ncbi()
#         # TODO: compare ncbi_with old (see if any sequence must be updated)