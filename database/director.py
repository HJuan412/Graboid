#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:08:53 2023

@author: hernan
This script contains de database director, which handles sequence lookup and retrieval
"""

#%% libraries
import logging
import os
import re

from database import surveyor as surv
from database import lister as lstr
from database import fetcher as ftch
from database import taxonomist as txnm
from database import merger as mrgr

#%% set logger
logger = logging.getLogger('Graboid.database.director')

#%% classes
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
    
    def clear_tmp(self, keep=True):
        if keep:
            return
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
                logger.warning("Couldn't read given ranks. Using default values instead")
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

        self.merger.merge(self.fetcher.seq_files, self.taxonomist.tax_files, self.taxonomist.guide_files)
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
    def expguide_file(self):
        return self.merger.expguide_out
    @property
    def nseqs(self):
        return self.merger.nseqs
    @property
    def rank_counts(self):
        return self.merger.rank_counts.loc[self.merger.ranks]
    @property
    def tax_summ(self):
        return self.merger.taxsumm_out