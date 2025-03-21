#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:59:33 2024

@author: hernan

Director script for database creation, exporting & updating (maybe)
"""

#%% libraries
from glob import glob
import logging
import numpy as np
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

# def retrieve(out_dir,
#              tmp_dir,
#              warn_dir,
#              taxon=None,
#              marker=None,
#              ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
#              ncbi=True,
#              bold=False,
#              fasta_file=None,
#              tax_file=None,
#              chunk_size=500,
#              max_attempts=3,
#              workers=1):
    
#     # retrieve taxdmp
#     names_tab, nodes_tab = fetch_tools.get_taxdmp(tmp_dir)
    
#     if fasta_file:
#         # retrieve records from fasta file
#         if tax_file is None:
#             raise Exception('Error: Must provide a taxonomy table to construct a graboid database from a local fasta file.')
#         db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = fetch_FASTA.retrieve_data(fasta_file, tax_file, out_dir, names_tab, nodes_tab, ranks, db_name='reference')
#     else:
#         # retrieve records from repositories
#         bold_exclude = []
#         if taxon is None or marker is None:
#             raise Exception('Error: Must provide <taxon> and <marker> search terms to construct a graboid database from online repositories.')
#         if ncbi:
#             ncbi_out = tmp_dir if bold else out_dir
#             ncbi_name = 'NCBI' if bold else 'reference'
#             ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, names_tab, nodes_tab, tmp_dir, warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
#             if not bold:
#                 db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
#         if bold:
#             bold_out = tmp_dir if ncbi else out_dir
#             bold_name = 'BOLD' if ncbi else 'reference'
#             bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, names_tab, nodes_tab, tmp_dir, warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
#             if ncbi:
#                 db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, out_dir, db_name='reference')
#                 db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, out_dir, db_name='reference')
#             else:
#                 db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
#     return db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs

# class Retriever:
#     def __init__(self, out_dir, tmp_dir, warn_dir):
#         self.out_dir = out_dir
#         self.tmp_dir = tmp_dir
#         self.warn_dir = warn_dir
#         # retrieve taxdmp
#         self.names_tab, self.nodes_tab = fetch_tools.get_taxdmp(tmp_dir)
        
#     def get_local(self, fasta_file, tax_file, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'], db_name='reference'):
#         # retrieve records from fasta file
#         self.db_seqs, self.db_lineages, self.db_taxonomy, self.db_names, self.db_nseqs = fetch_FASTA.retrieve_data(fasta_file, tax_file, self.out_dir, self.names_tab, self.nodes_tab, ranks, db_name=db_name)
#         self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
    
#     def get_remote(self, taxon, marker, ncbi=True, bold=False, chunk_size=500, max_attempts=3, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'], db_name='reference', workers=1):
#         # retrieve records from repositories
#         bold_exclude = [] # 
#         if ncbi:
#             ncbi_out = self.tmp_dir if bold else self.out_dir
#             ncbi_name = 'NCBI' if bold else db_name
#             ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
#             if not bold:
#                 db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
#         if bold:
#             bold_out = self.tmp_dir if ncbi else self.out_dir
#             bold_name = 'BOLD' if ncbi else db_name
#             bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
#             if ncbi:
#                 db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, self.out_dir, db_name=db_name)
#                 db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, self.out_dir, db_name=db_name)
#             else:
#                 db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
#         self.db_seqs = db_seqs
#         self.db_lineages = db_lineages
#         self.db_taxonomy = db_taxonomy
#         self.db_names = db_names
#         self.db_nseqs = db_nseqs
#         self.rank_counts = fetch_tools.count_ranks(self.db_taxonomy, self.db_lineages)
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
    
    def get_remote(self,
                   taxon,
                   marker,
                   ncbi=True,
                   bold=False,
                   chunk_size=500,
                   max_attempts=3,
                   ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
                   workers=1):
        self.taxon = taxon
        self.marker = marker
        self.ncbi = ncbi
        self.bold = bold
        self.ranks = ranks
        
        # retrieve records from repositories
        if ncbi and bold:
            # retrieve from both repositories
            ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, self.tmp_dir, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, db_name='NCBI')
            bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, self.tmp_dir, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name='BOLD')
            db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, self.db_dir, db_name='reference')
            db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, self.db_dir, db_name='reference')
        elif ncbi:
            # retrieve only from NCBI
            db_seqs, db_taxs, warn_failed, db_lineages, db_taxonomy, db_names, db_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, self.db_dir, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, db_name='reference')
        elif bold:
            # retrieve only from BOLD
            db_seqs, db_taxs, db_lineages, db_taxonomy, db_names, db_nseqs = fetch_BOLD.retrieve_data(taxon, marker, self.db_dir, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, [], max_attempts, ranks=ranks, db_name='reference')
        
        # bold_exclude = [] # 
        
        # if ncbi:
        #     ncbi_out = self.tmp_dir if bold else self.db_dir
        #     ncbi_name = 'NCBI' if bold else 'reference'
        #     ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs, bold_exclude = fetch_NCBI.retrieve_data(taxon, marker, ncbi_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, chunk_size, max_attempts, ranks, workers, ncbi_name)
        #     if not bold:
        #         db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = ncbi_seqs, ncbi_lineages, ncbi_taxonomy, ncbi_names, ncbi_nseqs
        # if bold:
        #     bold_out = self.tmp_dir if ncbi else self.db_dir
        #     bold_name = 'BOLD' if ncbi else 'reference'
        #     bold_seqs, bold_taxs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs = fetch_BOLD.retrieve_data(taxon, marker, bold_out, self.names_tab, self.nodes_tab, self.tmp_dir, self.warn_dir, bold_exclude, max_attempts, ranks=ranks, db_name=bold_name)
            
        #     if ncbi:
        #         db_seqs, db_nseqs = fetch_tools.merge_records(ncbi_seqs, bold_seqs, self.db_dir, db_name='reference')
        #         db_taxonomy, db_lineages, db_names = fetch_tools.merge_taxonomies(ncbi_taxonomy, ncbi_lineages, ncbi_names, bold_taxonomy, bold_lineages, bold_names, self.db_dir, db_name='reference')
        #     else:
        #         db_seqs, db_lineages, db_taxonomy, db_names, db_nseqs = bold_seqs, bold_lineages, bold_taxonomy, bold_names, bold_nseqs
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
    
    # def build_summ(self):
    #     try:
    #         source = self.guide_file
    #     except:
    #         source = []
    #         if self.ncbi:
    #             source.append('NCBI')
    #         if self.bold:
    #             source.append('BOLD')
    #         source = ' '.join(source)
    #     summ = pd.Series({'db_dir':self.db_dir,
    #                       'guide_file':self.guide_file,
    #                       'guide_length':self.marker_len,
    #                       'data_source':source,
    #                       'retrieved_seqs':self.db_nseqs,
    #                       'aligned_seqs':self.map_nrows,
    #                       'seq_file':self.db_seqs,
    #                       'tax_file':self.db_taxonomy,
    #                       'lineages_file':self.db_lineages,
    #                       'names_file':self.db_names,
    #                       'blast_db':self.guide_db,
    #                       'map_file':self.map_file,
    #                       'ranks':' '.join(self.ranks),
    #                       'description':self.description})
    #     summ = pd.concat([summ, pd.Series(self.rank_counts)])
    #     summ.to_csv(f'{self.db_dir}/summary.csv')
    
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
        # self.build_summ()

class GraboidDatabase:
    def __init__(self,
                 matrix,
                 accs,
                 bounds,
                 coverage,
                 coverage_norm,
                 taxonomy,
                 names_tab,
                 lineage_tab,
                 db_dir):
        self.matrix = matrix
        self.accs = accs
        self.bounds = bounds
        self.coverage = coverage
        self.coverage_norm = coverage_norm
        self.taxonomy = taxonomy
        self.names_tab = names_tab
        self.lineage_tab = lineage_tab
        self.summary()
        self.db_dir = db_dir
        
    @property
    def lineage(self):
        # subsection of lineage_tab corresponding to the reference instances
        return self.lineage_tab.loc[self.taxonomy.TaxId]
    
    def summary(self):
        self.tax_counts = self.lineage.apply(lambda x : len(np.unique(x[x != 0])))
        self.unk_counts = (self.lineage == 0).sum(axis = 0)
    
def load_map(map_file):
    # map_file: __map.npz file
    
    # load a map file and the corresponding accession file
    # from npz file, extract: alignment map, bounds array, coverage array
    # calculate normalized coverage
    map_ = np.load(map_file)
    matrix = map_['matrix']
    bounds = map_['bounds']
    coverage = map_['coverage']
    coverage_norm = coverage / coverage.max()
    # retrieve accession list
    accs = map_['accs']
    
    return matrix, accs, bounds, coverage, coverage_norm

def load_database(db_dir):
    if not os.path.isdir(db_dir):
        raise Exception(f'Database directory {db_dir} not found')
    
    # locate database files
    db_files = {'seqs_file':f'{db_dir}/reference.seqs',
                'tax_file':f'{db_dir}/reference.taxonomy',
                'lin_file':f'{db_dir}/reference.lineage',
                'names_file':f'{db_dir}/reference.names',
                'map_file':f'{db_dir}/reference__map.npz'}
    blastdb_dir = f'{db_dir}/guide'
    
    for file, filename in db_files.items():
        if not os.path.isfile(filename):
            raise Exception(f'Missing {file} file!')
    
    # check blast_db directory & files
    if not os.path.isdir(blastdb_dir):
        raise Exception('Missing blast db directory')
    blastdb_files = glob(f'{blastdb_dir}/guide_db*')
    if len(blastdb_files) != 9:
        #raise Exception(f'Found {len(blastdb_files)} files, expected 9')
        # todo: replace this with a warning
        pass
    
    
    # load map files
    matrix, accs, bounds, coverage, coverage_norm = load_map(db_files['map_file'])
    
    # load taxonomy data
    taxonomy = pd.read_csv(db_files['tax_file'], names=['Accession', 'TaxId'], skiprows=[0], index_col=0).loc[accs]
    lineage_tab = pd.read_csv(db_files['lin_file'], index_col=0)
    names_tab = pd.read_csv(db_files['names_file'], index_col=0)['SciName']
    
    result = GraboidDatabase(matrix, accs, bounds, coverage, coverage_norm, taxonomy, names_tab, lineage_tab, db_dir)
    return result
