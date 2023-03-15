#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
import argparse
import json
import logging
import os
import pandas as pd
import shutil

from Bio import Entrez
from DATA import DATA
from database import director as db
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

#%% functions
# Entrez
def set_entrez(email, apikey):
    Entrez.email = email
    Entrez.api_key = apikey

#%% main function
def main(db_name,
         ref_seq,
         taxon=None,
         marker=None,
         fasta=None,
         description='',
         ranks=None,
         bold=True,
         cp_fasta=False,
         chunksize=500,
         max_attempts=3,
         evalue=0.005,
         dropoff=0.05,
         min_height=0.1,
         min_width=2,
         min_seqs=10,
         filt_rank='genus',
         threads=1,
         keep=False,
         clear=False,
         email='',
         apikey=''):
    # Arguments:
    # required:
    #     db_name : name for the generated database
    #     ref_seq : reference sequence to build the alignment upon
    #     taxon & marker or fasta : search criteria or sequence file
    # optional:
    #   # data
    #     ranks : taxonomic ranks to be used. Default: Phylum, Class, Order, Family, Genus, Species
    #     bold : Search BOLD database. Valid when fasta = None
    #     cp_fasta : Copy fasta file to tmp_dir. Valid when fasta != None
    #   # ncbi
    #     chunksize : Number of sequences to retrieve each pass
    #     max_attempts : Number of retries for failed passes
    #   # mapping
    #     evalue : evalue threshold when building the alignment
    #     # mesas
    #       dropoff : percentage of mesa height drop to determine bound
    #       min_height : minimum sequence coverage to consider for a mesa candidate
    #       min_width : minimum weight needed for a candidate to register
    #     min_seqs : minimum sequence thresholds at the given filt_rank, to conserve a taxon
    #     filt_rank : taxonomic rank at which to apply the min_seqs threshold
    #     threads : threads to use when building the alignment
    #   # cleanup
    #     keep : keep the temporal files
    #     clear : clear the db_name directory (if it exists)
    
    # set entrez api key
    set_entrez(email, apikey)
    # check sequences in ref_seq
    n_refseqs = mp.check_fasta(ref_seq)
    if n_refseqs != 1:
        logger.error(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
        return
        
    # prepare output directories
    # check db_name (if it exists, check clear (if true, overwrite, else interrupt))
    db_dir = DATA.DATAPATH + '/' + db_name
    if db_name in DATA.DBASES:
        logger.info(f'A database with the name {db_name} already exists...')
        if not clear:
            logger.warning('Choose a diferent name or set "clear" as True')
            return
        logger.info(f'Removing existing database: {db_name}...')
        shutil.rmtree(db_dir)
    
    # create directories
    tmp_dir = db_dir + '/tmp'
    warn_dir = db_dir + '/warning'
    ref_dir = db_dir + '/ref'
    ref_file = ref_dir + '/ref.fasta'
    os.makedirs(tmp_dir)
    os.makedirs(warn_dir)
    os.makedirs(ref_dir)
    shutil.copyfile(ref_seq, ref_file)
    # add file handler
    fh = logging.FileHandler(db_dir + '/database.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # get sequence data (taxon & marker or fasta)
    # setup director
    db_director = db.Director(db_dir, tmp_dir, warn_dir)
    # user specified ranks to use
    db_director.set_ranks(ranks)
    # set databases
    databases = ['NCBI', 'BOLD'] if bold else ['NCBI']
    # retrieve sequences + taxonomies
    db_director.direct(taxon, marker, databases, fasta, chunksize, max_attempts)
    # clear temporal files
    db_director.clear_tmp(keep)
        
    # build map
    marker_len = mp.build_blastdb(ref_seq = ref_file,
                                  db_dir = ref_dir,
                                  clear = True,
                                  logger = logger)
    map_director = mp.Director(db_dir, warn_dir)
    map_director.direct(fasta_file = db_director.seq_file,
                        db_dir = ref_dir,
                        evalue = evalue,
                        dropoff=dropoff,
                        min_height=min_height,
                        min_width=min_width,
                        threads = threads,
                        logger = logger)
    
    # quantify information
    selector = fsele.Selector(db_dir)
    selector.build_tabs(map_director.matrix,
                        map_director.bounds,
                        map_director.coverage,
                        map_director.accs,
                        db_director.tax_file,
                        min_seqs = min_seqs,
                        rank = filt_rank)
    
    # assemble meta file
    meta_dict = {'seq_file':db_director.seq_file,
                 'tax_file':db_director.tax_file,
                 'guide_file':db_director.guide_file,
                 'expguide_file':db_director.expguide_file,
                 'ranks':db_director.ranks,
                 'nseqs':db_director.nseqs,
                 'rank_counts':db_director.rank_counts,
                 'tax_summ_file':db_director.tax_summ,
                 'mat_file':map_director.mat_file,
                 'ref_dir':ref_dir,
                 'ref_file':ref_file,
                 'acc_file':map_director.acc_file,
                 'order_file':selector.order_file,
                 'diff_file':selector.diff_file}
    with open(db_dir + '/meta.json', 'w') as meta_handle:
        json.dump(meta_dict, meta_handle)
    
    # generate db description
    if description == '':
        if fasta is None:
            description = f'Database built from search terms: {taxon} + {marker}. {db_director.nseqs} sequences.'
        else:
            description = f'Database built from file: {fasta}. {db_director.nseqs} sequences.'
    with open(db_dir + '/desc.json', 'w') as handle:
        json.dump(description, handle)
    
    print('Finished building database!')
    # write summaries
    # Database summary
    summ_file = db_dir + '/summary'
    with open(summ_file, 'w') as summary:
        summary.write(f'Database name: {db_name}\n')
        summary.write(f'Database location: {db_dir}\n')
        summary.write(f'Reference sequence (length): {ref_seq} ({marker_len})\n')
        summary.write(f'N sequences: {db_director.nseqs}\n')
        summary.write('Taxa:\n')
        summary.write('Rank (N taxa):\n')
        summary.write('\n'.join([f'\t{rk} ({count})' for rk, count in db_director.rank_counts.items()]))
        summary.write('\nMesas:\n')
    # mesas summary
    mesa_tab = pd.DataFrame(map_director.mesas, columns = 'start end bases average_cov'.split())
    mesa_tab.index.name = 'mesa'
    mesa_tab = mesa_tab.astype({'start':'int', 'end':'int', 'bases':'int'})
    mesa_tab['average_cov'] = mesa_tab.average_cov.round(2)
    mesa_tab.to_csv(summ_file, sep='\t', mode='a')

#%% main execution
parser = argparse.ArgumentParser(prog='Graboid DATABASE',
                                 usage='%(prog)s MODE_ARGS [-h]',
                                 description='Graboid DATABASE downloads records from the specified taxon/marker pair from the NCBI and BOLD databases')
parser.add_argument('-o', '--out_dir',
                    help='Output directory for the generated database files',
                    type=str)
parser.add_argument('-T', '--taxon',
                    help='Taxon to search for (use this in place of --fasta)',
                    type=str)
parser.add_argument('-M', '--marker',
                    help='Marker sequence to search for (use this in place of --fasta)',
                    type=str)
parser.add_argument('-F', '--fasta',
                    help='Pre-constructed fasta file (use this in place of --taxon and --marker)',
                    type=str)
parser.add_argument('--bold',
                    help='Include the BOLD database in the search',
                    action='store_true')
parser.add_argument('-r', '--ranks',
                    help='Set taxonomic ranks to include in the taxonomy table. Default: Phylum Class Order Family Genus Species',
                    nargs='*')
parser.add_argument('-c', '--chunksize',
                    default=500,
                    help='Number of records to download per pass. Default: 500',
                    type=int)
parser.add_argument('-m', '--max_attempts',
                    default=3,
                    help='Max number of attempts to download a chunk of records. Default: 3',
                    type=int)
parser.add_argument('--mv',
                    help='If a fasta file was provided, move it to the output directory',
                    action='store_true')
parser.add_argument('--keep',
                    help='Keep temporal files',
                    action='store_true')
parser.add_argument('--email',
                    help='Provide an email adress and an API key in order to use the NCBI Entrez utilities',
                    type=str)
parser.add_argument('--api_key',
                    help='API key associated to the provided email adress',
                    type=str)
parser.add_argument('-ref', '--ref_seq',
                    default=None,
                    help='Marker sequence to be used as base of the alignment',
                    type=str)
parser.add_argument('-e', '--evalue',
                    default=0.005,
                    help='E-value threshold for the BLAST matches. Default: 0.005',
                    type=float)
parser.add_argument('-t', '--threads',
                    default=1,
                    help='Number of threads to be used in the BLAST alignment. Default: 1',
                    type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    # check sequences in ref_seq
    n_refseqs = mp.check_fasta(args.ref_seq)
    if n_refseqs != 1:
        print(f'Reference file must contain ONE sequence. File {args.ref_seq} contains {n_refseqs}')
        pass
    main(out_dir = args.out_dir,
         email = args.email,
         api_key = args.api_key,
         ranks = args.rank,
         bold = args.bold,
         taxon = args.taxon,
         marker = args.marker,
         fasta = args.fasta,
         chunksize = args.chunksize,
         max_attempts = args.max_attempts,
         mv = args.mv,
         ref_seq = args.ref_seq,
         evalue = args.evalue,
         threads = args.threads,
         keep = args.keep)