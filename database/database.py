#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:59:33 2024

@author: hernan

Director script for database creation, exporting & updating (maybe)
"""

#%% libraries
import json
import logging
import os
import pandas as pd
import re
import shutil
from Bio import Entrez

# graboid modules
from Graboid.database import fetch_BOLD
from Graboid.database import fetch_FASTA
from Graboid.database import fetch_NCBI
from Graboid.database import fetch_tools
# from mapping import director as mp
# from mapping import matrix
from Graboid.mapping import mapping as mpp

#%% set logger
logger = logging.getLogger('Graboid.database')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

#%%
def set_entrez(email, apikey):
    Entrez.email = email
    Entrez.api_key = apikey

def make_db_dir(db_dir):
    # check that the given database directory is available
    if os.path.isdir(db_dir):
        raise Exception(f'Error: directory {db_dir} already exists')
    tmp_dir = f'{db_dir}/tmp'
    warn_dir = f'{db_dir}/warning'
    guide_dir = f'{db_dir}/guide'
    os.makedirs(tmp_dir)
    os.makedirs(warn_dir)
    os.makedirs(guide_dir)
    return tmp_dir, warn_dir, guide_dir

def build_summary(db_dir,
                  guide_file,
                  marker_len,
                  nseqs,
                  aln_seqs,
                  ranks):
    summ_file = f'{db_dir}/summary'
    with open(summ_file, 'w') as summary:
        summary.write(f'Database location: {db_dir}\n')
        summary.write(f'Guide sequence (length): {guide_file} ({marker_len})\n')
        summary.write(f'N sequences: {nseqs}\n')
        summary.write(f'Sequences in alignment: {aln_seqs}\n')
        summary.write('Taxa:\n')
        summary.write('Rank (N taxa):\n')
        summary.write('\n'.join([f'\t{rk} ({count})' for rk, count in ranks.items()]))
    
def retrieve(out_dir,
             tmp_dir,
             warn_dir,
             taxon=None,
             marker=None,
             ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
             ncbi=True,
             bold=False,
             fasta_file=None,
             tax_file=None,
             chunk_size=500,
             max_attempts=3,
             workers=1):
    
    # retrieve taxdmp
    names_tab, nodes_tab = fetch_tools.get_taxdmp(tmp_dir)
    
    if fasta_file:
        # retrieve records from fasta file
        if tax_file is None:
            raise Exception('Error: Must provide a taxonomy table to construct a graboid database from a local fasta file.')
        db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = fetch_FASTA.retrieve_data(fasta_file, tax_file, out_dir, names_tab, nodes_tab, ranks, db_name='reference')
    else:
        # retrieve records from repositories
        bold_exclude = []
        if taxon is None or marker is None:
            raise Exception('Error: Must provide <taxon> and <marker> search terms to construct a graboid database from online repositories.')
        if ncbi:
            ncbi_out = tmp_dir if bold else out_dir
            ncbi_name = 'NCBI' if bold else 'reference'
            ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, names_tab, nodes_tab, tmp_dir, warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
            if not bold:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
        if bold:
            bold_out = tmp_dir if ncbi else out_dir
            bold_name = 'BOLD' if ncbi else 'reference'
            bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, names_tab, nodes_tab, tmp_dir, warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
            if ncbi:
                db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, out_dir, db_name='reference')
                db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, out_dir, db_name='reference')
            else:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
    return db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs

class Retriever:
    def __init__(self, out_dir, tmp_dir, warn_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        # retrieve taxdmp
        self.names_tab, self.nodes_tab = fetch_tools.get_taxdmp(tmp_dir)
        
    def get_local(self, fasta_file, tax_file, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'], db_name='reference'):
        # retrieve records from fasta file
        self.db_seqs, self.db_lineages, self.db_taxonomy, self.db_names, self.db_nseqs = fetch_FASTA.retrieve_data(fasta_file, tax_file, self.out_dir, self.names_tab, self.nodes_tab, ranks, db_name=db_name)
        self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
    
    def get_remote(self, taxon, marker, ncbi=True, bold=False, chunk_size=500, max_attempts=3, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'], db_name='reference', workers=1):
        # retrieve records from repositories
        bold_exclude = [] # 
        if ncbi:
            ncbi_out = self.tmp_dir if bold else self.out_dir
            ncbi_name = 'NCBI' if bold else db_name
            ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
            if not bold:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
        if bold:
            bold_out = self.tmp_dir if ncbi else self.out_dir
            bold_name = 'BOLD' if ncbi else db_name
            bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
            if ncbi:
                db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, self.out_dir, db_name=db_name)
                db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, self.out_dir, db_name=db_name)
            else:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
        self.db_seqs = db_seqs
        self.db_lineages = db_lineages
        self.db_taxonomy = db_taxonomy
        self.db_names = db_names
        self.db_nseqs = db_nseqs
        self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
"""
Build a graboid database using either online repositories or a local fasta file

Parameters
----------
db_name : str
    Name for the generated database.
ref_seq : str
    Reference sequence to build the alignment upon.
ranks : list
    Taxonomic ranks to be used. Default: Phylum, Class, Order, Family, Genus, Species.
ncbi : Bool, optional
    Search NCBI database. The default is True.
bold : Bool, optional
    Search BOLD database. The default is False.
keep : bool, optional
    Keep temporal files. The default is False.
taxon : str, optional
    Taxon to look for in the online repositories. The default is None.
marker : str, optional
    Marker to look for in the online repositories. The default is None.
fasta : str, optional
    Path to local fasta file. The default is None.
description : TYPE, optional
    DESCRIPTION. The default is ''.
chunksize : int, optional
    Number of sequences to retrieve each pass. The default is 500.
max_attempts : int, optional
    Number of retries for failed passes. The default is 3.
evalue : float, optional
    evalue threshold when building the alignment. The default is 0.005.
dropoff : float, optional
    Percentage of mesa height drop to determine a border. The default is 0.05.
min_height : float, optional
    Minimum sequence coverage to consider for a mesa candidate. The default is 0.1.
min_width : int, optional
    Minimum weight needed for a candidate to register. The default is 2.
threads : int, optional
    Threads to use when building the alignment. The default is 1.
email : str, optional
    Valid email to be paired with an NCBI API key.
apikey : str, optional
    NCBI API key.

"""
class Constructor:
    def __init__(self, email, apikey):
        # set entrez api key
        set_entrez(email, apikey)
    
    def setup(self, db_dir, guide_file):
        # check sequences in ref_seq
        self.marker_len = mpp.check_guide(guide_file)
        
        # make database directory tree, copy guide file
        print('Setting up working directory...')
        self.db_dir = db_dir
        self.tmp_dir, self.warn_dir, self.guide_dir = make_db_dir(db_dir)
        self.guide_file = re.sub('^', f'{self.guide_dir}/', re.sub('.*/', '', guide_file))
        shutil.copyfile(guide_file, self.guide_file)
        
        # retrieve taxdmp
        self.names_tab, self.nodes_tab = fetch_tools.get_taxdmp(self.tmp_dir)
        
        # add file handler for logger
        fh = logging.FileHandler(f'{db_dir}/database.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def get_local(self, fasta_file, tax_file, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        self.fasta_file = fasta_file
        self.tax_file = tax_file
        self.ranks = ranks
        
        # retrieve records from fasta file
        self.db_seqs, self.db_lineages, self.db_taxonomy, self.db_names, self.db_nseqs = fetch_FASTA.retrieve_data(fasta_file, tax_file, self.db_dir, self.names_tab, self.nodes_tab, ranks, db_name='reference')
        self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
        self.description = f'Database built from file: {fasta_file}. {self.db_nseqs} sequences.'
    
    def get_remote(self, taxon, marker, ncbi=True, bold=False, chunk_size=500, max_attempts=3, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'], workers=1):
        self.taxon = taxon
        self.marker = marker
        self.ncbi = ncbi
        self.bold = bold
        self.ranks = ranks
        
        # retrieve records from repositories
        bold_exclude = [] # 
        if ncbi:
            ncbi_out = self.tmp_dir if bold else self.db_dir
            ncbi_name = 'NCBI' if bold else 'reference'
            ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
            if not bold:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
        if bold:
            bold_out = self.tmp_dir if ncbi else self.db_dir
            bold_name = 'BOLD' if ncbi else 'reference'
            bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
            if ncbi:
                db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, self.db_dir, db_name='reference')
                db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, self.db_dir, db_name='reference')
            else:
                db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
        self.db_seqs = db_seqs
        self.db_lineages = db_lineages
        self.db_taxonomy = db_taxonomy
        self.db_names = db_names
        self.db_nseqs = db_nseqs
        self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
        self.description = f'Database built from search terms: {taxon} + {marker}. {self.db_nseqs} sequences.'
    
    def build_map(self, evalue=0.005, threads=1):
        # build map
        print('Beginning sequence mapping...')
        print('Building blast reference database...')
        self.guide_db = f'{self.guide_dir}/guide_db'
        self.guide_header = mpp.makeblastdb(self.guide_file, self.guide_db)
        self.guide_len = mpp.get_guide_len(self.guide_db)
        print('Building map...')
        map_prefix = f'{self.db_dir}/reference'
        self.map_file, self.map_nrows, self.map_ncols, self.mapped_accs = mpp.build_map(self.db_seqs, self.guide_db, map_prefix, self.marker_len, evalue, threads)
        print('Sequence mapping is done!')
        self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages, self.mapped_accs)
    
    def build_summ(self):
        try:
            source = self.guide_file
        except:
            source = []
            if self.ncbi:
                source.append('NCBI')
            if self.bold:
                source.append('BOLD')
            source = ' '.join(source)
        summ = pd.Series({'db_dir':self.db_dir,
                          'guide_file':self.guide_file,
                          'guide_length':self.marker_len,
                          'data_source':source,
                          'retrieved_seqs':self.db_nseqs,
                          'aligned_seqs':self.map_nrows,
                          'seq_file':self.db_seqs,
                          'tax_file':self.db_taxonomy,
                          'lineages_file':self.db_lineages,
                          'names_file':self.db_names,
                          'blast_db':self.guide_db,
                          'map_file':self.map_file,
                          'ranks':' '.join(self.ranks),
                          'description':self.description})
        summ = pd.concat([summ, pd.Series(self.rank_counts)])
        summ.to_csv(f'{self.db_dir}/summary.csv')
    
    def build_local(self):
        pass
    def build_remote(self,
                     db_dir,
                     guide_file,
                     taxon,
                     marker,
                     ncbi=True,
                     bold=False,
                     chunk_size=500,
                     max_attempts=3,
                     ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
                     evalue=0.005,
                     threads=1):
        self.setup(db_dir, guide_file)
        self.get_remote(taxon, marker, ncbi, bold, chunk_size, max_attempts, ranks, workers=threads)
        self.build_map(evalue, threads)
        self.build_summ()

class Loader:
    def __init__(self, db_dir):
        if not os.path.isdir(db_dir):
            raise Exception(f'Database directory {db_dir} not found')
        self.db_dir = db_dir
        try:
            self.summary = pd.read_csv(f'{db_dir}/summary.csv', index_col=0)
        except FileNotFoundError:
            raise (f'Could not find summary file in directory {db_dir}')
        self.guide = self.summary['guide_file']
        self.seqs = self.summary['seq_file']
        self.taxonomy = self.summary['tax_file']
        self.lineages = self.summary['lineages_file']
        self.names = self.summary['names_file']
        self.guide_db = self.summary['blast_db']
        self.map = self.summary['map_file']
        self.ranks = self.summary['ranks']
    
    def check_files(self):
        files = 'guide seqs taxonomy lineages names map'.split()
        for fl in files:
            if not os.path.isfile(getattr(self, fl)):
                raise Exception(f'Missing {fl} file!')
    

def make_main(db_dir,
              guide_file,
              ranks,
              ncbi=True,
              bold=False,
              keep=False,
              taxon=None,
              marker=None,
              fasta_file=None,
              tax_file=None,
              description='',
              chunk_size=500,
              max_attempts=3,
              evalue=0.005,
              dropoff=0.05,
              min_height=0.1,
              min_width=2,
              threads=1,
              email='',
              apikey='',
              omit_missing=True):
    
    # set entrez api key
    set_entrez(email, apikey)
    
    # check sequences in ref_seq
    mpp.check_guide(guide_file)
    
    # make database directory tree, copy guide file
    print('Setting up working directory...')
    tmp_dir, warn_dir, guide_dir = make_db_dir(db_dir)
    guide_file_new = re.sub('^', f'{guide_dir}/', re.sub('.*/', '', guide_file))
    shutil.copyfile(guide_file, guide_file_new)
    
    # add file handler for logger
    fh = logging.FileHandler(f'{db_dir}/database.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    print('Retrieving records...')
    # get sequence data (taxon & marker or fasta)
    db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = retrieve(db_dir, tmp_dir, warn_dir, taxon, marker, ranks, ncbi, bold, fasta_file, tax_file, chunk_size, max_attempts, threads)
    rank_counts = fetch_tools.count_ranks(db_taxonomy, db_lineages)
    print('Data retrieval is done!')
    
    # build map
    print('Beginning sequence mapping...')
    print('Building blast reference database...')
    guide_db = f'{guide_dir}/guide_db'
    guide_header = mpp.makeblastdb(guide_file, guide_db)
    guide_len = mpp.get_guide_len(guide_db)
    print('Building map...')
    map_prefix = f'{db_dir}/reference'
    map_matrix_file, map_acc_file, map_nrows, map_ncols = mpp.build_map(db_seqs, guide_db, map_prefix, evalue, threads)
    print('Sequence mapping is done!')
    print('Finished building database!')
    
    # assemble meta file
    # generate db description
    if not description:
        if fasta_file:
            description = f'Database built from file: {fasta_file}. {db_nseqs} sequences.'
        else:
            description = f'Database built from search terms: {taxon} + {marker}. {db_nseqs} sequences.'
            
    meta_dict = {'seq_file':db_seqs,
                 'tax_file':db_taxonomy,
                 'lineages_file':db_lineages,
                 'names_file':db_names,
                 'guide_db':guide_db,
                 'guide_dir':guide_dir,
                 'map_mat_file':map_matrix_file,
                 'map_acc_file':map_acc_file,
                 'ranks':ranks,
                 'nseqs':db_nseqs,
                 'description': description}
    with open(db_dir + '/meta.json', 'w') as meta_handle:
        json.dump(meta_dict, meta_handle, indent=2)
    
    # write summaries
    build_summary(db_dir, guide_file, guide_len, db_nseqs, map_nrows, rank_counts)

def check_database(db_dir):
    """
    This function is used to retrieve the path to the database component files.

    Parameters
    ----------
    db_dir : str
        Path to the database directory.

    Returns
    -------
    taxonomy_file : str
        Path to the taxonomy table file.
    lineages_file : str
        Path to the lineages table file.
    names_file : str
        Path to the names table file.
    guide_db : str
        Path to the guide sequence BLAST reference.
    map_file : str
        Path to the alignment array file.
    map_acc_file : str
        Path to the accession list file.

    """
    with open(f'{db_dir}/meta.json', 'r') as h:
        db_metadata = json.load(h)
    
    # reference files
    taxonomy_file = db_metadata['tax_file']
    lineages_file = db_metadata['lineages_file']
    names_file = db_metadata['names_file']
    guide_db = db_metadata['guide_db']
    map_file = db_metadata['map_mat_file']
    map_acc_file = db_metadata['map_acc_file']
        
    return taxonomy_file, lineages_file, names_file, guide_db, map_file, map_acc_file