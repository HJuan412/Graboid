#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:05:58 2021

@author: hernan
Survey databases and download search summaries
"""

#%% libraries
from Bio import Entrez
import urllib3

#%% main function
def survey(summ_dir, taxons, markers, bold = True, ena = True, ncbi = True, verbose = True):
    """
    

    Parameters
    ----------
    summ_dir : str
        Path to the directory in which summaries will be stored.
    taxons : list
        Taxons to search for.
    markers : list
        Markers to search for.
    bold : bool, optional
        Survey the BOLD database. The default is True.
    ena : bool, optional
        Survey the ENA database. The default is True.
    ncbi : bool, optional
        Survey the NCBI database. The default is True.

    Returns
    -------
    None.

    """
    
    bold_survey = BOLD_surveyor(summ_dir, 'BOLD', bold, verbose)
    ena_survey = ENA_surveyor(summ_dir, 'ENA', ena, verbose)
    ncbi_survey = NCBI_surveyor(summ_dir, 'NCBI', ncbi, verbose)

    for taxon in taxons:
        bold_survey.survey(taxon)
        for marker in markers:
            ena_survey.survey(taxon, marker)
            ncbi_survey.survey(taxon, marker)

#%% classes
class Surveyor():
    def __init__(self, out_dir, dbase, active = False, verbose = False):
        self.out_dir = out_dir
        self.dbase = dbase
        self.active = active
        self.verbose = verbose
    
    def activate(self):
        self.active = True
    def deactivate(self):
        self.active = False
    
    def generate_filename(self, taxon, marker = ''):
        filename = f'{self.out_dir}/{taxon}_{marker}_{self.dbase}.summ'
        return filename

    def dl_and_save(self, apiurl, out_file):
        """
        Connects to a database (BOLD or ENA) and downloads search results
    
        Parameters
        ----------
        apiurl : str
            Generated instructions for the api.
        outfile : str
            Output file name.
    
        Returns
        -------
        None.
    
        """
        # connect to server & send request
        http = urllib3.PoolManager()
        r = http.request('GET', apiurl, preload_content = False)
        
        # stream request and store in outfile
        # TODO: handle interruptions
        with open(out_file, 'ab') as handle:
            for chunk in r.stream():
                handle.write(chunk)
        # close connection
        r.release_conn()
        return
    
    def anounce(self, taxon, marker = None):
        msg = f'Surveying {self.dbase} for {taxon}'
        if not marker is None:
            msg += f'{marker}'
        if self.verbose:
            print(msg)

    def survey(self, taxon, marker = None):
        if self.active:
            self.anounce(taxon, marker)
            out_file = self.generate_filename(taxon, marker)
            apiurl = self.get_url(taxon, marker)
            self.dl_and_save(apiurl, out_file)
            return
        return

class BOLD_surveyor(Surveyor):
    def get_url(self, taxon, marker = None):
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
        return apiurl

class ENA_surveyor(Surveyor):
    def get_url(self, taxon, marker):
        apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
        return apiurl

class NCBI_surveyor(Surveyor):
    
    def survey(self, taxon, marker):
        if self.active:
            self.anounce(taxon, marker)
            out_file = self.generate_filename(taxon, marker)
            search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
            search_record = Entrez.read(search)
            with open(out_file, 'w') as handle:
                handle.write('\n'.join(search_record['IdList']))
            return
        return