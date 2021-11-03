#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:30:40 2021

@author: hernan

Compare and merge temporal sequence files
"""

#%% libraries
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from Bio.SeqRecord import SeqRecord
from glob import glob
import pandas as pd
#%% functions
def build_filetab(seq_dir):
    # list temporal sequence files, register taxon, marker and database
    files = glob(f'{seq_dir}/*tmp')
    filetab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'File'])
    
    for idx, file in enumerate(files):
        split_file = file.split('/')[-1].split('.')[0].split('_')
        tax, mark, dbase = split_file
        
        if mark != '': # don't include BOLDr files
            filetab.at[idx] = [tax, mark, dbase, file]
    return filetab

def build_ddict(dbases):
    # build a {dbase:{}} dictionary. Will contain the seqdict for each database
    dicts_dict = {}
    for db in dbases:
        dicts_dict[db] = {}
    
    return dicts_dict

def build_tdict(dbases):
    # build a {dbase:infotab} dictionary. Will contain the information dataframe for each database
    tabs_dict = {}
    for db in dbases:
        tabs_dict[db] = build_infotab({}, db)
    
    return tabs_dict

def build_seqdict(seqfile):
    # build a {short_accession:(accession, header, sequence)} dictionary from a fasta file
    seqdict = {}
    with open(seqfile, 'r') as handle:
        for header, seq in sfp(handle):
            acc = header.split(' ')[0]
            acc_short = acc.split('.')[0]
            seqdict[acc_short] = (acc, header, seq)
    return seqdict

def build_infotab(seqdict, dbase):
    # build a dataframe with columns [Accession, Version, Database] from the given seqdict
    tab = pd.DataFrame.from_dict(seqdict, orient='index', columns = ['Accession', '1', '2']) # columns 1 & 2 are the header and seq elements in the seqdict, will be discarded

    acc_list = tab['Accession'].tolist()
    ver_list = [int(acc.split('.')[-1]) for acc in acc_list]

    info_tab = pd.DataFrame(index = tab.index, columns = ['Accession', 'Version', 'Database'])
    info_tab['Database'] = dbase
    info_tab.at[:,'Accession'] = acc_list
    info_tab.at[:, 'Version'] = ver_list
    return info_tab

#%% classes
class MergeTool():
    def __init__(self, taxon, marker, filetab, out_dir, db_order = ['NCBI', 'BOLD', 'ENA']):
        self.dicts_dict = build_ddict(db_order)
        self.tabs_dict = build_tdict(db_order)
        self.read_files(filetab)
        self.merged_tab = pd.DataFrame(columns = ['Version', 'Database'])
        self.out_prefix = f'{out_dir}/{taxon}_{marker}'
        self.db_order = db_order
    
    def add_seqdict(self, seqfile, dbase):
        # build a seqdict from the given seqfile, if dbase is not known, add it at lowest priority
        if not dbase in self.db_order:
            self.db_order.append(dbase)
            self.dicts_dict[dbase] = {}
            self.tabs_dict = build_infotab({}, dbase)

        seqdict = build_seqdict(seqfile)
        self.dicts_dict[dbase] = seqdict
        self.tabs_dict[dbase] = build_infotab(seqdict, dbase)
        return
    
    def read_files(self, filetab):
        # read and call add_seqdict for each file in the given table (use at construction)
        for _, row in filetab.iterrows():
            dbase = row['Database']
            file = row['File']                
            self.add_seqdict(file, dbase)
    
    def clear_seqdict(self, dbase):
        # remove a seqdict, clear the merged_table
        if dbase in self.db_order:
            self.dicts_dict[dbase] = {}
            self.tabs_dict[dbase] = build_infotab({}, dbase)
        
        self.merged_tab = self.merged_tab.iloc[0:0]
        return

    def merge_infotabs(self):
        # merge all infotabs following the given order (repeated entries are ignored, first dbases are prioritized)
        for dbase in self.db_order:
            discard_accs = set(self.merged_tab.index)
            db_dict = self.dicts_dict[dbase]
            db_tab = self.tabs_dict[dbase]
            db_accs = set(db_dict.keys()).difference(discard_accs) # incorporate all entries not already present
            db_subtab = db_tab.loc[db_accs]
            self.merged_tab = pd.concat([self.merged_tab, db_subtab])
        return
    
    def save_merged_tab(self):
        out_file = self.out_prefix + '_info.tab'
        self.merged_tab.to_csv(out_file)

    def merge_seqfiles(self):
        # merge sequences based on the merged_tab and store results (only run after merge_infotabs)
        records = []
        for dbase, subtab in self.merged_tab.groupby('Database'):
            db_accs = subtab.index
            db_dict = self.dicts_dict[dbase]
            for acc in db_accs:
                acc, header, seq = db_dict[acc]
                record = SeqRecord(Seq(seq), header, description = '')
                records.append(record)

        out_file = self.out_prefix + '.fasta'
        with open(out_file, 'w') as out_handle:
            SeqIO.write(records, out_handle, 'fasta')
    
    def process(self):
        # direct processing of the taxon - marker duo
        self.merge_infotabs()
        self.save_merged_tab()
        self.merge_seqfiles()

class Merger():
    def __init__(self, in_dir, out_dir, warn_dir, db_order = ['NCBI', 'BOLD', 'ENA']):
        """
        Constructor of the merger class.

        Parameters
        ----------
        in_dir : str
            Directory containing the temporal sequence files.
        out_dir : str
            Directory containing the output files.
        warn_dir : str
            Directory containing the warning files.
        db_order : list, optional
            Establish the priority for considering repeated sequences.
            The default is ['NCBI', 'BOLD', 'ENA'].

        Returns
        -------
        None.

        """
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.db_order = db_order
        self.seq_tab = build_filetab(in_dir)
    
    def check_seq_tab(self):
        # make sure seq_tab is not empty
        if len(self.seq_tab) == 0:
            with open(f'{self.warn_dir}/merger.warn', 'w') as warn_handle:
                warn_handle.write('No sequence files found in directory {self.in_dir}')
            return False
        return True
    
    def merge(self):
        if self.check_seq_tab():
            for tax, sub_tab0 in self.seq_tab.groupby('Taxon'):
                for mark, sub_tab1 in sub_tab0.groupby('Marker'):
                    merge_agent = MergeTool(tax, mark, sub_tab1, self.seq_dir, self.db_order)
                    merge_agent.process()
