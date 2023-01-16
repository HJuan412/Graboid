#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:53:46 2021

@author: hernan
Director for database creation and updating
"""

#%% libraries
import argparse
import logging
import os
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
        self.taxonomist = txnm.Taxonomist(tmp_dir)
        self.merger = mrgr.Merger(out_dir)
    
    def clear_tmp(self):
        tmp_files = self.get_tmp_files()
        for file in tmp_files:
            os.remove(file)
    
    def set_ranks(self, ranks=['phylum', 'class', 'order', 'family', 'genus', 'species']):
        # set taxonomic ranks to retrieve for the training data.
        # propagate to taxonomist and merger
        fmt_ranks = [rk.lower() for rk in ranks]
        logger.INFO(f'Taxonomic ranks set as {" ".join(fmt_ranks)}')
        self.taxonomist.set_ranks(fmt_ranks)
        self.merger.set_ranks(fmt_ranks)

    def direct_fasta(self, fasta_file, chunksize=500, max_attempts=3, cp_fasta=False):
        # direct database construction from a prebuilt fasta file
        # sequences should have a valid genbank accession
        fasta_name = re.sub('.*/', '', re.sub('\..*', '', fasta_file))
        seq_path = f'{self.out_dir}/{fasta_name}.fasta'
        if cp_fasta:
            shutil.copy(fasta_file, seq_path)
        logger.info(f'Moved fasta file {fasta_file} to location {self.out_dir}')
        # generate taxtmp file
        print(f'Retrieving TaxIDs for {fasta_file}...')
        self.fetcher.fetch_tax_from_fasta(fasta_file)
        
        print('Reconstructing taxonomies...')
        # taxonomy needs no merging so it is saved directly to out_dir
        self.taxonomist.out_dir = self.out_dir # dump tax table to out_dir
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        
        print('Building output files...')
        self.merger.merge_from_fasta(seq_path, self.taxonomist.out_files['NCBI'])
        self.taxonomist.out_files = {} # clear out_files container so the generated file is not found by get_tmp_files
        print('Done!')
    
    def direct(self, taxon, marker, databases, chunksize=500, max_attempts=3):
        # build database from zero, needs a valid taxon (ideally at a high level such as phylum or class) and marker gene
        print('Surveying databases...')
        for db in databases:
            self.surveyor.survey(taxon, marker, db, max_attempts)
        print('Building accession lists...')
        self.lister.build_list(self.surveyor.out_files)
        print('Fetching sequences...')
        self.fetcher.set_bold_file(self.surveyor.out_files['BOLD'])
        self.fetcher.fetch(self.lister.out_file, chunksize, max_attempts)
        print('Reconstructing taxonomies...')
        self.taxonomist.taxing(self.fetcher.tax_files, chunksize, max_attempts)
        print('Merging sequences...')
        self.merger.merge(self.fetcher.seq_files, self.taxonomist.out_files)
        print('Done!')
    
    def get_tmp_files(self):
        tmp_files = []
        for file in self.surveyor.out_files.values():
            tmp_files.append(file)
        tmp_files.append(self.lister.out_file)
        for file in self.fetcher.seq_files.values():
            tmp_files.append(file)
        for file in self.fetcher.tax_files.values():
            tmp_files.append(file)
        for file in self.taxonomist.out_files.values():
            tmp_files.append(file)
        return tmp_files
    
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
    def rank_file(self):
        return self.merger.rank_dict_out
    @property
    def valid_file(self):
        return self.merger.valid_rows_out

#%% main function
def main(db_name, ref_seq, taxon=None, marker=None, fasta=None, ranks=None, bold=True, cp_fasta=False, chunksize=500, max_attempts=3, evalue=0.005, threads=1, keep=False, clear=False):
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
    #     threads : threads to use when building the alignment
    #   # cleanup
    #     keep : keep the temporal files
    #     clear : clear the db_name directory (if it exists)
    
    # prepare output directories
    # check db_name (if it exists, check clear (if true, overwrite, else interrupt))
    db_dir = DATA.DATAPATH + '/' + db_name
    if db_name in DATA.DBASES:
        print(f'A database with the name {db_name} already exists...')
        if not clear:
            print('Choose a diferent name or set "clear" as True')
            return
        print(f'Removing existing database: {db_name}...')
        os.rmdir(db_dir)
    # create directories
    tmp_dir = db_dir + '/tmp'
    warn_dir = db_dir + '/warning'
    ref_dir = db_dir + '/ref'
    os.makedirs(tmp_dir)
    os.makedirs(warn_dir)
    os.makedirs(ref_dir)
    
    # get sequence data (taxon & marker or fasta)
    # setup director
    db_director = Director(db_dir, tmp_dir, warn_dir)
    # user specified ranks to use
    if not ranks is None:
        db_director.set_ranks(ranks)
    # set databases
    databases = ['NCBI', 'BOLD'] if bold else ['NCBI']
    # retrieve sequences + taxonomies
    if not fasta is None:
        # build db using fasta file (overrides taxon, mark)
        db_director.direct_fasta(fasta,
                                 chunksize,
                                 max_attempts,
                                 cp_fasta)
    elif not (taxon is None or marker is None):
        #build db using tax & mark
        db_director.direct(taxon,
                           marker,
                           databases,
                           chunksize,
                           max_attempts)
    else:
        print('No search parameters provided. Either set a path to a fasta file in the --fasta argument or a taxon and a marker in the --taxon and --marker arguments')
        return
    # clear temporal files
    if not keep:
        db_director.clear_tmp()
        
    # build map
    mp.build_blastdb(ref_seq = ref_seq,
                     ref_dir = db_dir,
                     ref_name = 'ref',
                     clear = True)
    map_director = mp.Director(db_dir, warn_dir)
    map_director.direct(fasta_file = db_director.seq_file,
                        db_dir = ref_dir,
                        evalue = evalue,
                        threads = threads,
                        keep = keep)
    # quantify information
    selector = fsele.Selector(db_dir)
    selector.set_matrix(map_director.matrix, map_director.bounds, db_director.tax_file)
    selector.build_tabs()
    selector.save_order_mat()
    # write summaries

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
parser.add_argument('-r', '--ref_seq',
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
    pass