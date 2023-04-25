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
logger = logging.getLogger('Graboid.database.surveyor')

#%% classes
# survey tools
class SurveyTool:
    # Template for Survey tool, each instance will handle a signle taxon - marker - database trio
    # Count download attempts and register success
    def __init__(self, taxon, marker, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.out_dir = out_dir
        self.out_file = f'{out_dir}/{taxon}_{marker}__{self.database}.summ' # database identifier separated by __
        self.attempt = 1
        self.max_attempts = 3
        self.done = False
    
    def survey(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.attempt_dl()
        if self.done:
            logger.info(f'Done getting summary from {self.database} in {self.attempt} attempts.')
        else:
            # surveyor was unable to download a summary, generate a warning, delete incomplete file
            logger.warning(f'Failed to get summary from {self.database} after {self.max_attempts} attempts.')
            try:
                os.remove(self.out_file)
            except:
                pass

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
            except Exception as excp:
                # update attempt count
                logger.warning(f'Download of {self.taxon} {self.marker} interrupted (Exception: {excp}), {self.ntries - self.attempt} attempts remaining')
                self.attempt += 1

# Specific survey tools
# each of these uses a survey method to attempt to download a summary
class SurveyBOLD(SurveyWAPI):
    @property
    def database(self):
        return 'BOLD'
    
    def get_url(self):
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/combined?taxon={self.taxon}&marker={self.marker}&format=tsv' # this line downloads sequences AND taxonomies
        return apiurl

class SurveyENA(SurveyWAPI):
    @property
    def database(self):
        return 'ENA'
    
    def get_url(self):
        apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{self.taxon}%22%20AND%20%22{self.marker}%22'
        return apiurl

class SurveyNCBI(SurveyTool):
    @property
    def database(self):
        return 'NCBI'
    
    # This surveyor uses the Entrez package instead of an API, defines its own survey method
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
            except Exception as excp:
                logger.warning(f'Download of {self.taxon} {self.marker} interrupted (Exception: {excp}), {self.max_attempts - self.attempt} attempts remaining')
                self.attempt += 1

class Surveyor:
    # This class manages the download process for all taxon - marker - database trio
    
    # tooldict is a class attribute used to identify the survey tool to be used
    tooldict = {'BOLD':SurveyBOLD,
                'ENA':SurveyENA,
                'NCBI':SurveyNCBI}

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.out_files = {}

    def survey(self, taxon, marker, database, max_attempts=3):
        # Survey each given database for the taxon / marker duo.
        # ntries determines the number of attempts
        try:
            tool = Surveyor.tooldict[database](taxon, marker, self.out_dir)
        except KeyError:
            logger.error(f'Database name {database} is not valid')
            raise
            
        print(f'Surveying database {database} for {taxon} {marker}')
        tool.survey(max_attempts)
        if tool.done:
            self.out_files[database] = tool.out_file
        else:
            raise Exception(f'Failed survey for {taxon} {marker} in {database} database')