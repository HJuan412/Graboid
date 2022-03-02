#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:12:47 2021

@author: hernan
"""

#%% libraries
from Bio import Entrez
import pandas as pd
import urllib3

#%% classes
# survey tools
class SurveyTool():
    # Template for Survey tool, each instance will handle a signle taxon - marker - database trio
    # Count download attempts and register success
    def __init__(self, taxon, marker, out_dir):
        self.taxon = taxon
        self.marker = marker
        self.generate_outfile(out_dir)
        self.attempt = 0
        self.done = False
    
    def generate_outfile(self, out_dir):
        # This generates the output file for the downloaded summary
        dbase = self.get_dbase()
        self.out_file = f'{out_dir}/{self.taxon}_{self.marker}_{dbase}.summ'
        return

class SurveyWAPI(SurveyTool):
    # Survey tool using an API
    def survey(self):
        # Connects to a database (BOLD or ENA) and downloads search results
        # update attempt count
        self.attempt += 1

        # connect to server & send request
        http = urllib3.PoolManager()
        apiurl = self.get_url()
        r = http.request('GET', apiurl, preload_content = False)
        
        # stream request and store in outfile
        try:
            with open(self.out_file, 'ab') as handle:
                for chunk in r.stream():
                    handle.write(chunk)
            # close connection
            r.release_conn()
            self.done = True # signal success
            return 0
        except:
            return 1

# Specific survey tools
# each of these uses a survey method to attempt to download a summary
class SurveyBOLD(SurveyWAPI):
    def get_dbase(self):
        return 'BOLD'

    def get_url(self):
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={self.taxon}&format=tsv'
        return apiurl

class SurveyENA(SurveyWAPI):
    def get_dbase(self):
        return 'ENA'

    def get_url(self):
        apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{self.taxon}%22%20AND%20%22{self.marker}%22'
        return apiurl

class SurveyNCBI(SurveyTool):
    # This surveyor uses the Entrez package instead of an API, defines its own survey method
    def get_dbase(self):
        return 'NCBI'

    def survey(self):
        # update attempt count
        self.attempt += 1
        try:
            search = Entrez.esearch(db='nucleotide', term = f'{self.taxon}[Organism] AND {self.marker}[all fields]', idtype = 'acc', retmax="100000")
            search_record = Entrez.read(search)
            with open(self.out_file, 'w') as handle:
                handle.write('\n'.join(search_record['IdList']))
            self.done = True # signal success
            return 0
        except:
            return 1

class Surveyor():
    # This class manages the download process for all taxon - marker - database trio
    def __init__(self, taxon, marker, type1tools, type2tools, out_dir, warn_dir):
        """
        Parameters
        ----------
        taxon : str
            Taxons to survey for.
        marker : list
            Markers to survey for.
        type1tools : list
            List type 1 survey tools. These tools only survey for taxons.
        type2tools : list
            List type 2 survey tools. These tools survey for taxon - marker duos.
        out_dir : str
            Path to the output directory.
        warn_dir : str
            Path to the warning directory.

        Returns
        -------
        None.

        """
        self.taxon = taxon
        self.marker = marker
        self.type1 = type1tools
        self.type2 = type2tools
        self.out_dir = out_dir
        self.warn_dir = warn_dir
        self.warn_report = pd.DataFrame(columns = ['Database', 'Taxon', 'Marker'])
    
    def download(self, tooltype, ntries):
        # Attempts to download perform a survey for a given taxon - marker pair with a given survey tool.
        # If download is not completed after the given number of attempts (ntries), store a warning.
        tool = tooltype(self.taxon, self.marker, self.out_dir)
        dbase = tool.get_dbase()
        print(f'Surveying {dbase} for {self.taxon} {self.marker}')
        while tool.attempt < ntries and not tool.done:
            tool.survey()
        if not tool.done:
            print(f'Failed to get summary from {dbase} after {ntries} attempts.')
            self.warn_report = self.warn_report.append([tool.get_dbase, self.taxon, self.marker])
        else:
            print(f'Done getting summary from {dbase} in {tool.attempt} attempts.')
    
    def save_warn(self):
        # Generate a warning file if a taxon - marker - database can't be downloaded.
        if len(self.warn_report) > 0:
            self.warn_report.to_csv('{self.warn_dir}/summary_warnings.csv')
        
    def survey(self, ntries = 3):
        # Survey each given database for the taxon / marker duo.
        # ntries determines the number of attempts
        for t1 in self.type1:
            self.attempt_dl(t1, ntries, self.taxon)

        for t2 in self.type2:
            self.attempt_dl(t2, ntries, self.taxon, self.marker)
        self.save_warn()