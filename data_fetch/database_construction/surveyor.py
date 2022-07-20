#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:12:47 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
import logging
import os
import urllib3

#%% setup logger
logger = logging.getLogger('database_logger.surveyor')

#%% classes
# survey tools
class SurveyTool:
    # Template for Survey tool, each instance will handle a signle taxon - marker - database trio
    # Count download attempts and register success
    def __init__(self, taxon, marker, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.database = self.get_dbase()
        self.generate_outfile(out_dir)
        self.attempt = 1
        self.max_attempts = 3
        self.done = False
    
    def generate_outfile(self, out_dir):
        self.out_file = f'{out_dir}/{self.taxon}_{self.marker}_{self.database}.summ'
    
    def survey(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.attempt_dl()
        if self.done:
            self.logger.info(f'Done getting summary from {self.database} in {self.attempt} attempts.')
        else:
            # surveyor was unable to download a summary, generate a warning, delete incomplete file
            self.logger.warning(f'Failed to get summary from {self.database} after {self.max_attempts} attempts.')
            os.remove(self.out_file)

class SurveyWAPI(SurveyTool):
    # Survey tool using an API
    def attempt_dl(self):
        self.attempt = 1
        while self.attempt <= self.max_attempts:
            # connect to server & send request
            http = urllib3.PoolManager()
            apiurl = self.get_url()
            r = http.request('GET', apiurl, preload_content = False)

            # stream request and store in outfile
            try:
                with open(self.out_file, 'wb') as handle:
                    for chunk in r.stream():
                        handle.write(chunk)
                # close connection
                r.release_conn()
                self.done = True # signal success
                break
            except:
                # update attempt count
                self.attempt += 1
                self.logger.warning(f'Download of {self.taxon} {self.marker} interrupted, {self.ntries - self.attempt} attempts remaining')

# Specific survey tools
# each of these uses a survey method to attempt to download a summary
class SurveyBOLD(SurveyWAPI):
    # TODO: BOLD downloads take too long regardless of mode (summary or sequence), furthermore, the api used to build the summary doesn't allow filtering by marker
    # consider using the full data retrieval API
    def get_logger(self):
        self.logger = logging.getLogger('database_logger.surveyor.BOLD')
    
    def get_dbase(self):
        return 'BOLD'

    def get_url(self):
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={self.taxon}&format=tsv'
        return apiurl

class SurveyENA(SurveyWAPI):
    def get_logger(self):
        self.logger = logging.getLogger('database_logger.surveyor.ENA')
    def get_dbase(self):
        return 'ENA'

    def get_url(self):
        apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{self.taxon}%22%20AND%20%22{self.marker}%22'
        return apiurl

class SurveyNCBI(SurveyTool):
    # This surveyor uses the Entrez package instead of an API, defines its own survey method
    def get_logger(self):
        self.logger = logging.getLogger('database_logger.surveyor.NCBI')
    
    def get_dbase(self):
        return 'NCBI'

    def attempt_dl(self):
        self.attempt = 1
        while self.attempt <= self.max_attempts:
            # use entrez API to download summary
            try:
                search = Entrez.esearch(db='nucleotide', term = f'{self.taxon}[Organism] AND {self.marker}[all fields]', idtype = 'acc', retmax="100000")
                search_record = Entrez.read(search)
                with open(self.out_file, 'w') as handle:
                    handle.write('\n'.join(search_record['IdList']))
                self.done = True # signal success
                break
            except:
                self.attempt += 1

class Surveyor:
    # This class manages the download process for all taxon - marker - database trio
    
    # tooldict is a class attribute used to identify the survey tool to be used
    tooldict = {'BOLD':SurveyBOLD,
                'ENA':SurveyENA,
                'NCBI':SurveyNCBI}

    def __init__(self, taxon, marker, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.out_dir = out_dir
        self.out_files = {}

    def survey(self, database, ntries=3):
        # Survey each given database for the taxon / marker duo.
        # ntries determines the number of attempts
        if not database in Surveyor.tooldict.keys():
            logger.error(f'Database name {database} is not valid')
            return
        
        tool = Surveyor.tooldict[database](self.taxon, self.marker, self.out_dir)
        logging.info(f'Surveying database {database} for {self.taxon} {self.marker}')
        
        tool.survey(ntries)
        if tool.done:
            self.out_files[database] = tool.out_file