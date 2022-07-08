#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:12:47 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
import lister
import logging
import pandas as pd
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
        self.generate_outfile(out_dir)
        self.attempt = 0
        self.ntries = 3
        self.done = False
        self.warn = False
    
    def generate_outfile(self, out_dir):
        # This generates the output file for the downloaded summary
        dbase = self.get_dbase()
        self.out_file = f'{out_dir}/{self.taxon}_{self.marker}_{dbase}.summ'
    
    def survey(self, ntries = 3):
        self.ntries = ntries
        dbase = self.get_dbase()
        print(f'Surveying {dbase} for {self.taxon} {self.marker}')
        self.attempt_dl()
        if self.done:
            self.logger.info(f'Done getting summary from {dbase} in {self.attempt} attempts.')
        else:
            self.logger.warning(f'Failed to get summary from {dbase} after {ntries} attempts.')
            self.warn = True

class SurveyWAPI(SurveyTool):
    # Survey tool using an API
    def attempt_dl(self):
        self.attempt = 0
        while self.attempt < self.ntries:
            # connect to server & send request
            http = urllib3.PoolManager()
            apiurl = self.get_url()
            r = http.request('GET', apiurl, preload_content = False)

            # stream request and store in outfile
            try:
                with open(self.out_file, 'wb') as handle: # changed to 'wb' from 'ab', see if it still works
                    for chunk in r.stream():
                        handle.write(chunk)
                # close connection
                r.release_conn()
                self.done = True # signal success
                break
            except:
                # update attempt count
                self.attempt += 1
                self.logger.warning(f'Download interrupted, {self.ntries - self.attempt} remaining')

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
        self.attempt = 0
        while self.attempt < self.ntries:
            try:
                search = Entrez.esearch(db='nucleotide', term = f'{self.taxon}[Organism] AND {self.marker}[all fields]', idtype = 'acc', retmax="100000")
                search_record = Entrez.read(search)
                with open(self.out_file, 'w') as handle:
                    handle.write('\n'.join(search_record['IdList']))
                self.done = True # signal success
                break
            except:
                self.attempt += 1

class Surveyor():
    # This class manages the download process for all taxon - marker - database trio
    def __init__(self, taxon, marker, out_dir, warn_dir, old_file = None, databases = ['NCBI']):
        """
        Parameters
        ----------
        taxon : str
            Taxon to survey for.
        marker : str
            Marker to survey for.
        out_dir : str
            Path to the output directory.
        warn_dir : str
            Path to the warning directory.
        databases : list. Optionañ
            Public databases in which to survey for. Currently, surveyor only looks in NCBI.

        Returns
        -------
        None.

        """
        self.taxon = taxon
        self.marker = marker
        self.databases = databases
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.warn_report = pd.Series(name = 'Attempts')
        self.__get_surv_tools()
        # prelister must be defined AFTER the survey tools
        self.prelister = lister.PreLister(self.taxon, self.marker, self.survey_tools['NCBI'].out_file, self.out_dir, self.warn_dir, old_file)
    
    def __get_surv_tools(self):
        # set up the tools to survey the specified databases
        # BOLD is currently deactivated, records are downloaded via the Fetcher module. Kept just in case
        self.survey_tools = {}
        tooldict = {'BOLD':SurveyBOLD,
                    'ENA':SurveyENA,
                    'NCBI':SurveyNCBI}

        for db in self.databases:
            if db in tooldict.keys():
                tool = tooldict[db]
                self.survey_tools[db] = tool(self.taxon, self.marker, self.out_dir)

    def save_warn(self):
        # Generate a warning file if a taxon - marker - database can't be downloaded.
        if len(self.warn_report) > 0:
            self.warn_report.to_csv(f'{self.warn_dir}/warnings.surv')

    def survey(self, ntries = 3):
        # Survey each given database for the taxon / marker duo.
        # ntries determines the number of attempts
        for tool in self.survey_tools.values():
            tool.survey(ntries)
            if tool.warn:
                self.warn_report.at[tool.get_dbase()] = tool.attempt
        
        self.save_warn()
        
        # prelister compares the summary with the old_file (if present) and generates the acc_file
        self.prelister.pre_list()

#%% functions
def build_acc_lists(taxon, marker, databases, out_dir, ntries = 3):
    surveyors = {'BOLD':SurveyBOLD(taxon, marker, out_dir),
                 'ENA':SurveyENA(taxon, marker, out_dir),
                 'NCBI':SurveyNCBI(taxon, marker, out_dir)}

    for database in databases:
        logging.info(f'Surveying database {database} for {taxon} {marker}')
        surveyors[database].survey(ntries)
    
    out_files = {database:surveyors[database].out_file for database in databases}
    return out_files