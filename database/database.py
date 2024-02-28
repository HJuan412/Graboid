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
import fetch_BOLD
import fetch_FASTA
import fetch_NCBI
import fetch_tools
from DATA import DATA
from mapping import director as mp
from preprocess import feature_selection as fsele

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

def make_db_dir(db_name):
    db_dir = f'{DATA.DATAPATH}/{db_name}'
    tmp_dir = db_dir + '/tmp'
    warn_dir = db_dir + '/warning'
    ref_dir = db_dir + '/ref'
    os.makedirs(tmp_dir)
    os.makedirs(warn_dir)
    os.makedirs(ref_dir)
    return db_dir, tmp_dir, warn_dir, ref_dir

def build_summary(db_dir,
                  db_name,
                  ref_seq,
                  marker_len,
                  nseqs,
                  aln_seqs,
                  ranks,
                  mesas):
    summ_file = db_dir + '/summary'
    with open(summ_file, 'w') as summary:
        summary.write(f'Database name: {db_name}\n')
        summary.write(f'Database location: {db_dir}\n')
        summary.write(f'Reference sequence (length): {ref_seq} ({marker_len})\n')
        summary.write(f'N sequences: {nseqs}\n')
        summary.write(f'Sequences in alignment: {aln_seqs}\n')
        summary.write('Taxa:\n')
        summary.write('Rank (N taxa):\n')
        summary.write('\n'.join([f'\t{rk} ({count})' for rk, count in ranks]))
        summary.write('\nAlignment mesas:\n')
    # mesas summary
    mesa_tab = pd.DataFrame(mesas, columns = 'start end bases average_cov'.split())
    mesa_tab.index.name = 'Mesa'
    mesa_tab = mesa_tab.astype({'Start':int, 'End':int, 'Bases':int})
    mesa_tab['Average_cov'] = mesa_tab.average_cov.round(2)
    mesa_tab.to_csv(summ_file, sep='\t', mode='a')
    
def retrieve():
    return

def make_main(db_name,
              ref_seq,
              ranks,
              ncbi=True,
              bold=False,
              keep=False,
              taxon=None,
              marker=None,
              fasta_file=None,
              tax_file=None,
              description='',
              chunksize=500,
              max_attempts=3,
              evalue=0.005,
              dropoff=0.05,
              min_height=0.1,
              min_width=2,
              threads=1,
              email='',
              apikey=''):
    
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
    
    # set entrez api key
    set_entrez(email, apikey)
    
    # check sequences in ref_seq
    n_refseqs = mp.check_fasta(ref_seq)
    if n_refseqs != 1:
        raise Exception(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
    
    # check that the database name is available
    if DATA.database_exists(db_name):
        raise Exception(f'A database with the name {db_name} already exists')
    
    # make database directory tree, copy reference file
    db_dir, tmp_dir, warn_dir, ref_dir = make_db_dir(db_name)
    ref_file = ref_dir + '/ref.fasta'
    shutil.copyfile(ref_seq, ref_file)
    
    # add file handler for logger
    fh = logging.FileHandler(db_dir + '/database.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # retrieve taxdmp
    names_tab, nodes_tab = fetch_tools.get_taxdmp(tmp_dir)
    # Retrieve data
    if ncbi:
        ncbi_seqs, ncbi_taxs, warn_failed, ncbi_lineage, ncbi_taxonomy, ncbi_names = fetch_NCBI.retrieve_data(taxon, marker, tmp_dir, names_tab, nodes_tab, *ranks)
    if bold:
        bold_seqs, bold_taxs, bold_lineage, bold_taxonomy, bold_names = fetch_BOLD.retrieve_data(taxon, marker, tmp_dir, names_tab, nodes_tab, *ranks)
    if not fasta_file is None:
        fasta_seqs, fasta_lineage, fasta_taxonomy, fasta_names = fetch_FASTA.retrieve_data(fasta_file, tax_file, tmp_dir, names_tab, nodes_tab, *ranks)
        
    # get sequence data (taxon & marker or fasta)
    # setup director
    db_director = db.Director(db_dir, tmp_dir, warn_dir)
    # user specified ranks to use
    db_director.set_ranks(ranks)
    # set databases
    databases = ['NCBI', 'BOLD'] if bold else ['NCBI']
    # retrieve sequences + taxonomies
    print('Beginning data retrieval...')
    db_director.direct(taxon, marker, databases, fasta, chunksize, max_attempts)
    # clear temporal files
    db_director.clear_tmp(keep)
    print('Data retrieval is done!')
    
    # build map
    print('Beginning sequence mapping...')
    print('Building blast reference database...')
    marker_len = mp.build_blastdb(ref_seq = ref_file,
                                  db_dir = ref_dir,
                                  clear = True,
                                  logger = logger)
    map_director = mp.Director(db_dir, warn_dir, logger)
    map_director.direct(fasta_file = db_director.seq_file,
                        db_dir = ref_dir,
                        evalue = evalue,
                        dropoff = dropoff,
                        min_height = min_height,
                        min_width = min_width,
                        threads = threads)
    print('Sequence mapping is done!')
    
    # quantify information
    print('Calculating information contents...')
    selector = fsele.Selector(db_dir, db_director.ranks)
    selector.build_tabs(map_director.matrix,
                        map_director.accs,
                        db_director.tax_tab,
                        db_director.ext_guide)
    print('Calculations are done!')
    print('Finished building database!')
    
    # assemble meta file
    # generate db description
    if description == '':
        if fasta is None:
            description = f'Database built from search terms: {taxon} + {marker}. {db_director.nseqs} sequences.'
        else:
            description = f'Database built from file: {fasta}. {db_director.nseqs} sequences.'
    meta_dict = {'name':db_name,
                 'seq_file':db_director.seq_file,
                 'tax_file':db_director.tax_file,
                 'guide_file':db_director.guide_file,
                 'expguide_file':db_director.expguide_file,
                 'ranks':db_director.ranks,
                 'nseqs':db_director.nseqs,
                 'mat_shape':map_director.matrix.shape,
                 'rank_counts':db_director.rank_counts.to_dict(),
                 'tax_summ_file':db_director.tax_summ,
                 'mat_file':map_director.mat_file,
                 'reference':re.sub('.*/', '', ref_seq),
                 'ref_file':ref_file,
                 'blast_db':ref_dir + '/db',
                 'acc_file':map_director.acc_file,
                 'order_file':selector.order_file,
                 'diff_file':selector.diff_file,
                 'description': description}
    with open(db_dir + '/meta.json', 'w') as meta_handle:
        json.dump(meta_dict, meta_handle, indent=2)
    
    # write summaries
    build_summary(db_dir,
                  db_name,
                  ref_seq,
                  marker_len,
                  db_director.nseqs,
                  map_director.matrix.shape[0],
                  db_director.rank_counts.items(),
                  map_director.mesas)