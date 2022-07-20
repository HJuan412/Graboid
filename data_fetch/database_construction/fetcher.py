#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:21:22 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import logging
import numpy as np
import pandas as pd
import urllib3

#%% setup logger
logger = logging.getLogger('database_logger.fetcher')
#%% functions
# Entrez
# TODO: remove mail and apikey
def set_entrez(email = "hernan.juan@gmail.com", apikey = "7c100b6ab050a287af30e37e893dc3d09008"):
    Entrez.email = email
    Entrez.api_key = apikey

# accession list handling
def acc_slicer(acc_list, chunksize):
    # slice the list of accessions into chunks
    n_seqs = len(acc_list)
    for i in np.arange(0, n_seqs, chunksize):
        yield acc_list[i:i+chunksize]

# fetch functions (OBSOLETE: MAY DELETE LATER)
###############################################################################
def fetch_ncbi(acc_list, out_seqs, out_taxs):
    # download sequences from NCBI, generates a temporal fasta and am acc:taxID list
    seq_handle = Entrez.efetch(db = 'nucleotide', id = acc_list, rettype = 'fasta', retmode = 'xml')
    seqs_recs = Entrez.read(seq_handle)
    
    seqs = []
    taxs = []

    for acc, seq in zip(acc_list, seqs_recs):
        seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence']), name = seq['TSeq_orgname'], description = None))
        taxs.append(','.join([acc, seq['TSeq_taxid']]))
    
    with open(out_seqs, 'a') as out_handle0:
        SeqIO.write(seqs, out_handle0, 'fasta')

    with open(out_taxs, 'a') as out_handle1:
        out_handle1.write('\n'.join(taxs + ['']))
    
def fetch_api(apiurl, out_handle):
    http = urllib3.PoolManager()
    r = http.request('GET', apiurl, preload_content = False)
    for chunk in r.stream():
        out_handle.write(chunk)
    r.release_conn()
    return

def fetch_bold(taxon, marker, out_seqs, out_taxs = None):
    # downloads sequence AND summary
    apiurl = f'http://www.boldsystems.org/index.php/API_Public/combined?taxon={taxon}&marker={marker}&format=tsv'
    with open(out_seqs, 'ab') as out_handle:
        fetch_api(apiurl, out_handle)
    return
###############################################################################
def fetch_seqs(acc_list, database, out_header, chunk_size=500, max_attempts=3):
    # this performs a single pass over the given acc_list
    # chunks that can't be downloaded are returned in the failed list
    failed = []
    
    # select the corresponding fetcher
    fetcher = fetch_dict[database](out_header)
    # split acc_list
    chunks = acc_slicer(acc_list, chunk_size)
    
    # try to download each chunk of records
    for chunk_n, chunk in enumerate(chunks):
        done = fetcher.fetch(chunk, chunk_n, max_attempts)
        if not done:
            failed += chunk
    return failed
#%% classes
class NCBIFetcher:
    # This class handles record download from NCBI
    def __init__(self, out_header):
        # out_header is used to generate the names for the output files
        self.out_seqs = f'{out_header}.seqtmp'
        self.out_taxs = f'{out_header}.taxtmp'
        self.logger = logging.getLogger('database_logger.fetcher.NCBI')

    def fetch(self, acc_list, chunk_n=0, max_attempts=3):
        # download acc_list sequences from NCBI to a fasta file and tax ids to an acc:taxID list
        # return True if successful
        chunksize = len(acc_list)
        attempt = 1
        while attempt <= max_attempts:
            try:
                # contact NCBI and attempt download
                seq_handle = Entrez.efetch(db = 'nucleotide', id = acc_list, rettype = 'fasta', retmode = 'xml')
                seqs_recs = Entrez.read(seq_handle)
                
                seqs = []
                taxs = []
                
                # generate seq records and acc:taxid series
                for acc, seq in zip(acc_list, seqs_recs):
                    seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence']), description = ''))
                    taxs.append(pd.Series({'Accession':acc, 'TaxID':seq['TSeq_taxid']}))
                
                # save seq recods to fasta
                with open(self.out_seqs, 'a') as out_handle0:
                    SeqIO.write(seqs, out_handle0, 'fasta')
                # save tax table to csv
                taxs = pd.DataFrame(taxs)
                taxs.to_csv(self.out_taxs, header = False, index = False, mode='a')
                return True
            except:
                self.logger.warning(f'Chunk {chunk_n} download interrupted at {len(seqs)} of {chunksize}. {max_attempts - attempt} attempts remaining')
                attempt += 1
        return False

class BOLDFetcher:
    def __init__(self, out_header):
        self.out_seqs = f'{out_header}.seqtmp'
        self.out_taxs = f'{out_header}.taxtmp'
        self.logger = logging.getLogger('database_logger.fetcher.BOLD')

    def fetch(self, acc_list, chunk_n=0, max_attempts=3):
        # TODO: Rework this function to work on full records downloaded by surveyor (do bold postproc's job)
        # downloads sequence AND summary
        self.__set_outfiles()
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/combined?taxon={self.taxon}&marker={self.marker}&format=tsv'
        with open(self.out_seqs, 'ab') as out_handle:
            fetch_api(apiurl, out_handle)
        return

fetch_dict = {'BOLD':BOLDFetcher,
              'NCBI':NCBIFetcher}

class Fetcher():
    # TODO: fetch failed sequences
    def __init__(self, acc_file, out_dir):
        self.load_accfile(acc_file)
        self.out_header = ''
        self.out_dir = out_dir
    
    def load_accfile(self, acc_file):
        self.acc_file = acc_file
        self.acc_tab = pd.read_csv(acc_file, index_col = 0)
        
        # generate header for output files
        out_header = acc_file.split('/')[-1].split('.')[0]
        self.out_header = f'{self.out_dir}/{out_header}'
        failed_file = acc_file.split('.')[0]
        self.failed_file = f'{failed_file}_failed.acc'
    
    def check_accs(self):
        if len(self.acc_tab) == 0:
            logger.warning(f'Accession table from {self.acc_file} is empty')
            return False
        return True

    def fetch(self, chunk_size=500, max_attempts=3):
        # Fetches sequences and taxonomies, splits acc lists into chunks of chunk_size elements
        # for each chunk, try to download up to max_attempts times

        if not self.check_accs():
            # acc_table is empty
            return
        
        failed = [] # here we store accessions that failed to download
        
        # split the acc_table by database
        for database, sub_tab in self.acc_tab.groupby('Database'):
            acc_list = sub_tab['Accession'].to_list()
            out_header = f'{self.out_header}_{database}'
            # first pass
            failed0 = fetch_seqs(acc_list, database, out_header, chunk_size, max_attempts)
            # do a second pass if some sequences couldn't be downloaded
            if len(failed0) == 0:
                continue
            failed0 = fetch_seqs(failed0, database, out_header, chunk_size, max_attempts)
            
            if len(failed0) > 0:
                # failed to download sequences after two passes
                logger.warning(f'Failed to download {len(failed0)} of {len(acc_list)} records from {database}. Failed accessions saved to {self.failed_file}')
                failed += failed0
        
        # generate a sub table containing the failed sequences (if any)
        if len(failed) > 0:
            failed_tab = self.acc_tab.loc[failed]
            failed_tab.to_csv(self.failed_file)
