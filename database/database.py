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
import re
import shutil
from Bio import Entrez

# graboid modules
from . import fetch_BOLD
from . import fetch_FASTA
from . import fetch_NCBI
from . import fetch_tools
# from mapping import director as mp
# from mapping import matrix
from mapping import mapping as mpp

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