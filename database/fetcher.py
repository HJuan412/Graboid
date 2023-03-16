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
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from Bio.SeqRecord import SeqRecord

import http
import logging
import numpy as np
import pandas as pd
import re

#%% setup logger
logger = logging.getLogger('Graboid.database.fetcher')
logging.captureWarnings(True)

#%% functions
# accession list handling
def acc_slicer(acc_list, chunksize):
    # slice the list of accessions into chunks
    n_seqs = len(acc_list)
    for i in np.arange(0, n_seqs, chunksize):
        yield acc_list[i:i+chunksize]

def fetch_seqs(acc_list, database, out_header, bold_file=None, chunk_size=500, max_attempts=3):
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
    # bold_file kept for compatibility
    def __init__(self, out_header, bold_file=None):
        # out_header is used to generate the names for the output files
        self.out_seqs = f'{out_header}.seqtmp'
        self.out_taxs = f'{out_header}.taxtmp'
        
    def fetch(self, acc_list, chunk_size=500, max_attempts=3):
        # download acc_list sequences from NCBI to a fasta file and tax ids to an acc:taxID list
        # chunks that can't be downloaded are returned in the failed list
        failed = []
        # split acc_list
        chunks = acc_slicer(acc_list, chunk_size)
        n_chunks = int(np.ceil(len(acc_list)/chunk_size))
        print(f'Downloading {len(acc_list)} records from NCBI...')
        for chunk_n, chunk in enumerate(chunks):
            print(f'Downloading chunk {chunk_n + 1} of {n_chunks}')
            seq_recs = []
            for attempt in range(max_attempts):
                # try to retrieve the sequence records up to max_attempt times per chunk
                try:
                    # contact NCBI and attempt download
                    logger.debug(f'Starting chunk download (size: {len(chunk)})')
                    seq_handle = Entrez.efetch(db = 'nucleotide', id = chunk, rettype = 'fasta', retmode = 'xml')
                    seq_recs = Entrez.read(seq_handle)
                    logger.debug(f'Done retrieving {len(seq_recs)} records')
                    break
                except IOError:
                    print(f'Chunk {chunk_n} download interrupted. {max_attempts - attempt} attempts remaining')
                    continue
                except http.client.IncompleteRead:
                    print(f'Chunk {chunk_n} download interrupted. {max_attempts - attempt} attempts remaining')
                    continue
                except KeyboardInterrupt:
                    print('Manually aborted')
                    return
            if len(seq_recs) != len(chunk):
                failed += chunk
                continue
            
            # generate seq records and retrieve taxids
            seqs = []
            taxids = []
            for acc, seq in zip(chunk, seq_recs):
                seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence']), description = ''))
                taxids.append(seq['TSeq_taxid'])
            
            # save seq recods to fasta
            with open(self.out_seqs, 'a') as out_handle0:
                SeqIO.write(seqs, out_handle0, 'fasta')
            # save tax table to csv
            taxs = pd.DataFrame({'Accession':chunk, 'TaxID':taxids})
            taxs.to_csv(self.out_taxs, header = chunk_n == 0, index = False, mode='a')
        return failed

class BOLDFetcher:
    def __init__(self, out_header, bold_file):
        self.out_seqs = f'{out_header}.seqtmp'
        self.out_taxs = f'{out_header}.taxtmp'
        self.bold_file = bold_file
        
    def fetch(self, acc_list, chunk_size=0, max_attempts=3):
        # read table and get records in acc_list
        bold_tab = pd.read_csv(self.bold_file, sep = '\t', encoding = 'latin-1', dtype = str) # latin-1 to parse BOLD files
        bold_tab = bold_tab.loc[bold_tab['sampleid'].isin(acc_list)]
        bold_tab.set_index('sampleid', inplace=True)
        # get sequences and taxonomy table
        seq_subtab = bold_tab['nucleotides']
        tax_subtab = bold_tab.loc[:, 'phylum_taxID':'subspecies_name']
        # save results
        records = []
        for acc, seq in seq_subtab.iteritems():
            records.append(SeqRecord(id=acc, seq = Seq(seq.replace('-', '')), description = ''))
        
        with open(self.out_seqs, 'a') as out_handle:
            SeqIO.write(records, out_handle, 'fasta')
        tax_subtab.to_csv(self.out_taxs)
        return [] # empty list returned for compatibility with NCBIFetcher's failed list

fetch_dict = {'BOLD':BOLDFetcher,
              'NCBI':NCBIFetcher}

class Fetcher():
    # class attribute containing valid fetcher tools
    fetch_dict = {'BOLD':BOLDFetcher,
                  'NCBI':NCBIFetcher}
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.seq_files = {}
        self.tax_files = {}
        self.bold_file = None
    
    def load_acctab(self, acc_file):
        self.acc_file = acc_file
        self.acc_tab = pd.read_csv(acc_file, index_col = 0)
        
        # generate header for output files
        self.out_header = re.sub('.*/', self.out_dir + '/', re.sub('\..*', '', acc_file))
        self.failed_file = self.out_header +'_failed.acc'
        # check that acc tab is not empty
        if len(self.acc_tab) == 0:
            raise Exception(f'Accession table from {self.acc_file} is empty')

    def fetch(self, acc_tab, summ_files, chunk_size=500, max_attempts=3):
        # Fetches sequences and taxonomies, splits acc lists into chunks of chunk_size elements
        # for each chunk, try to download up to max_attempts times
        
        # load and check files
        try:
            self.load_acctab(acc_tab)
        except Exception as excp:
            logger.warning(excp)
            raise
        
        failed = [] # here we store accessions that failed to download
        
        # split the acc_table by database
        for database, sub_tab in self.acc_tab.groupby('Database'):
            acc_list = sub_tab['Accession'].to_list()
            out_header = f'{self.out_header}__{database}'
            # set fetcher, include corresponding summary file (does nothing for mcbi fetcher)
            fetcher = fetch_dict[database](out_header, summ_files[database])
            # update out file containers
            self.seq_files[database] = fetcher.out_seqs
            self.tax_files[database] = fetcher.out_taxs
            # first pass
            failed0 = fetcher.fetch(acc_list, chunk_size, max_attempts)
            # do a second pass if some sequences couldn't be downloaded
            if len(failed0) == 0:
                continue
            failed0 = fetcher.fetch(failed0, chunk_size, max_attempts)
            
            if len(failed0) > 0:
                # failed to download sequences after two passes
                logger.warning(f'Failed to download {len(failed0)} of {len(acc_list)} records from {database}. Failed accessions saved to {self.failed_file}')
                failed += failed0
            logger.info(f'Retrieved {len(acc_list) - len(failed0)}/{len(acc_list)} sequences from {database} database.')
        
        # generate a sub table containing the failed sequences (if any)
        if len(failed) > 0:
            failed_tab = self.acc_tab.loc[failed]
            failed_tab.to_csv(self.failed_file)
        
        total_records = len(self.acc_tab)
        failed_records = len(failed)
        logger.info(f'Finished retrieving {total_records - failed_records} of {total_records} records.')
        if failed_records > 0:
            logger.info(f'{failed_records} saved to {self.failed_files}')
        if len(failed) == len(self.acc_tab):
            raise Exception('Failed to retrieve sequences')
    
    def fetch_tax_from_fasta(self, fasta_file):
        # generate output file
        out_file = re.sub('.*/', self.out_dir + '/', re.sub('.seqtmp', '.taxtmp', fasta_file))
        self.seq_files['NCBI'] = fasta_file
        self.tax_files['NCBI'] = out_file
        
        # get list of accessions
        with open(fasta_file, 'r') as fasta_handle:
            acc_list = [re.sub(' .*', '', acc) for acc, seq in sfp(fasta_handle)]
        
        if len(acc_list) == 0:
            raise Exception(f'No valid accession codes could be retrieved from file {fasta_file}')
        # retrieve taxIDs and build TaxID tab
        summ_handle = Entrez.esummary(db='nucleotide', id=acc_list, retmode='xml')
        summ_recs = Entrez.read(summ_handle)
        taxids = [int(summ['TaxId']) for summ in summ_recs]
        
        taxs = pd.DataFrame({'Accession':acc_list, 'TaxID':taxids})
        # save tax table to csv
        taxs.to_csv(out_file, header=False, index=False)
