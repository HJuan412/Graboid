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
import re
import shutil

from Bio import Entrez
from DATA import DATA
from database import surveyor as surv
from database import lister as lstr
from database import fetcher as ftch
from database import taxonomist as txnm
from database import merger as mrgr
from mapping import director as mp
from preprocess import feature_selection as fsele

#%% set logger
logger = logging.getLogger('Graboid.database')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

#%% functions
# Entrez
def set_entrez(email, apikey):
    Entrez.email = email
    Entrez.api_key = apikey

#%%
class Director:
    def __init__(self, out_dir, tmp_dir, warn_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.warn_dir = warn_dir
        
        # set workers
        self.surveyor = surv.Surveyor(tmp_dir)
        self.lister = lstr.Lister(tmp_dir)
        self.fetcher = ftch.Fetcher(tmp_dir)
        self.taxonomist = txnm.Taxonomist(tmp_dir, warn_dir)
        self.merger = mrgr.Merger(out_dir)
    
    def clear_tmp(self):
        for file in os.listdir(self.tmp_dir):
            os.remove(file)
    
    def set_ranks(self, ranks=None):
        # set taxonomic ranks to retrieve for the training data.
        # this method ensures the taxonomic ranks are sorted in descending order, regarless of how they were input by the user
        # also checks that ranks are valid
        valid_ranks = 'domain subdomain superkingdom kingdom phylum subphylum superclass class subclass division subdivision superorder order suborder superfamily family subfamily genus subgenus species subspecies'.split()
        if ranks is None:
            ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']
        else:
            rks_formatted = set([rk.lower() for rk in ranks])
            rks_sorted = []
            for rk in valid_ranks:
                if rk in rks_formatted:
                    rks_sorted.append(rk)
            ranks = rks_sorted
            if len(ranks) == 0:
                self.logger.warning("Couldn't read given ranks. Using default values instead")
                ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
                
        # propagate to taxonomist and merger
        logger.info(f'Taxonomic ranks set as {" ".join(ranks)}')
        self.taxonomist.set_ranks(ranks)
        self.merger.set_ranks(ranks)
        self.ranks = ranks
    
    def retrieve_fasta(self, fasta_file, chunksize=500, max_attempts=3):
        # retrieve sequence data from a prebuilt fasta file
        # sequences should have a valid genbank accession
        print('Retrieving sequences from file {fasta_file}')
        seq_path = re.sub('.*/', self.out_dir + '/', re.sub('.fa.*', '__fasta.seqtmp', fasta_file))
        os.symlink(fasta_file, seq_path)
        # create a symbolic link to the fasta file to follow file nomenclature system without moving the original file
        print(f'Retrieving TaxIDs from {fasta_file}...')
        self.fetcher.fetch_tax_from_fasta(seq_path)
    
    def retrieve_download(self, taxon, marker, databases, chunksize=500, max_attempts=3):
        # retrieve sequence data from databases
        # needs a valid taxon (ideally at a high level such as phylum or class) and marker gene
        if taxon is None:
            raise Exception('No taxon provided')
        if marker is None:
            raise Exception('No marker provided')
        if databases is None:
            raise Exception('No databases provided')
        print('Surveying databases...')
        for db in databases:
            self.surveyor.survey(taxon, marker, db, max_attempts)
        print('Building accession lists...')
        self.lister.build_list(self.surveyor.out_files)
        print('Fetching sequences...')
        self.fetcher.fetch(self.lister.out_file, self.surveyor.out_files, chunksize, max_attempts)
        
    def process(self, chunksize=500, max_attempts=3):
        print('Reconstructing taxonomies...')
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)

        self.merger.merge(self.fetcher.seq_files, self.taxonomist.out_files)
        print('Done!')
        
    def direct(self, taxon, marker, databases, fasta_file, chunksize=500, max_attempts=3):
        # retrieve sequence and taxonomy data
        if fasta_file is None:
            self.retrieve_download(taxon, marker, databases, chunksize, max_attempts)
        else:
            self.retrieve_fasta(fasta_file, chunksize, max_attempts)
        # process data
        self.process(chunksize, max_attempts)
        
    @property
    def seq_file(self):
        return self.merger.seq_out
    @property
    def acc_file(self):
        return self.merger.acc_out
    @property
    def tax_file(self):
        return self.merger.tax_out
    @property
    def guide_file(self):
        return self.merger.taxguide_out
    @property
    def valid_file(self):
        return self.merger.valid_rows_out
    @property
    def nseqs(self):
        return self.merger.nseqs
    @property
    def base_rank(self):
        return self.merger.base_rank
    @property
    def base_taxa(self):
        return ' '.join(self.merger.base_taxa)
    @property
    def rank_counts(self):
        return self.merger.rank_counts

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
        raise Exception(f'Reference file must contain ONE sequence. File {ref_seq} contains {n_refseqs}')
        
    # prepare output directories
    # check db_name (if it exists, check clear (if true, overwrite, else interrupt))
    db_dir = DATA.DATAPATH + '/' + db_name
    if db_name in DATA.DBASES:
        print(f'A database with the name {db_name} already exists...')
        if not clear:
            raise Exception('Choose a diferent name or set "clear" as True')
        print(f'Removing existing database: {db_name}...')
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
    fh = logging.FileHandler(tmp_dir + '/log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # get sequence data (taxon & marker or fasta)
    # setup director
    db_director = Director(db_dir, tmp_dir, warn_dir)
    # user specified ranks to use
    db_director.set_ranks(ranks)
    # set databases
    databases = ['NCBI', 'BOLD'] if bold else ['NCBI']
    # retrieve sequences + taxonomies
    db_director.direct(taxon, marker, databases, fasta, chunksize, max_attempts)
    # clear temporal files
    if not keep:
        db_director.clear_tmp()
        
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
                 'rank':db_director.ranks,
                 'nseqs':db_director.nseqs,
                 'base_rank':db_director.base_rank,
                 'base_taxa':db_director.base_taxa,
                 'rank_counts':db_director.rank_counts,
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
        summary.write(f'\tBase taxon (lvl): {db_director.base_taxa} ({db_director.base_rank})\n')
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