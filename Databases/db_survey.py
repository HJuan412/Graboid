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

#%% functions
# def dl_and_save(apiurl, outfile):
#     """
#     Connects to a database (BOLD or ENA) and downloads search results

#     Parameters
#     ----------
#     apiurl : str
#         Generated instructions for the api.
#     outfile : str
#         Output file name.

#     Returns
#     -------
#     None.

#     """
#     # connect to server & send request
#     http = urllib3.PoolManager()
#     r = http.request('GET', apiurl, preload_content = False)
    
#     # stream request and store in outfile
#     # TODO: handle interruptions
#     with open(outfile, 'ab') as handle:
#         for chunk in r.stream():
#             handle.write(chunk)
#     # close connection
#     r.release_conn()
#     return
        
# def generate_filename(tax, mark, db):
#     filename = f'{tax}_{mark}_{db}.summ'
#     return filename

# # url generators
# def get_bold_url(taxon):
#     apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
#     return apiurl

# def get_ena_url(taxon, marker):
#     apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
#     return apiurl

# # survey
# def survey_bold(taxon, outfile):
#     """
#     Performs a search using the BOLD api

#     Parameters
#     ----------
#     taxon : str
#         Taxon to search for.
#     outfile : str
#         Output file name.

#     Returns
#     -------
#     None.

#     """
#     apiurl = get_bold_url(taxon)
#     dl_and_save(apiurl, outfile)

# def survey_ena(taxon, marker, outfile):
#     """
#     Performs a search using the ENA api

#     Parameters
#     ----------
#     taxon : str
#         Taxon to search for.
#     marker : str
#         Marker to search for.
#     outfile : str
#         Output file name.

#     Returns
#     -------
#     None.

#     """
#     apiurl = get_ena_url(taxon, marker)
#     dl_and_save(apiurl, outfile)

# def survey_ncbi(taxon, marker, outfile):
#     """
#     Performs a search in the NCBI database using the Entrez api

#     Parameters
#     ----------
#     taxon : str
#         Taxon to search for.
#     marker : str
#         Marker to search for.
#     outfile : str
#         Output file name.

#     Returns
#     -------
#     None.

#     """
#     search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
#     search_record = Entrez.read(search)
#     with open(outfile, 'w') as handle:
#         handle.write('\n'.join(search_record['IdList']))

#%% main function
def survey(summ_dir, taxons, markers, bold = True, ena = True, ncbi = True):
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
    surv_list = []
    if bold:
        surv_list.append(BOLD_surveyor(summ_dir))
    if ena:
        surv_list.append(ENA_surveyor(summ_dir))
    if ncbi:
        surv_list.append(NCBI_surveyor(summ_dir))

    for taxon in taxons:
        for marker in markers:
            for surv in surv_list:
                surv.survey(taxon, marker)

#%% classes
class Surveyor():
    def __init__(self, out_dir):
        self.out_dir = out_dir
    
    def dl_and_save(self, apiurl, out_file):
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


    def survey(self, taxon, marker = None):
        out_file = self.generate_filename(taxon. marker)
        apiurl = self.get_url(taxon, marker)
        self.dl_and_save(apiurl, out_file)
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

    def survey(self, taxon, marker = None):
        out_file = self.generate_filename(taxon, marker)
        apiurl = self.get_url(taxon, marker)
        self.dl_and_save(apiurl, out_file)
        return

class BOLD_surveyor(Surveyor):
    def generate_filename(self, taxon, marker = None):
        filename = f'{self.out_dir}/{taxon}__BOLD.summ'
        return filename
    
    def get_url(self, taxon, marker = None):
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
        return apiurl

class ENA_surveyor(Surveyor):
    def get_url(self, taxon, marker):
        apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
        return apiurl

    def generate_filename(self, taxon, marker = None):
        filename = f'{self.out_dir}/{taxon}_{marker}_BOLD.summ'
        return filename

class NCBI_surveyor(Surveyor):
    def generate_filename(self, taxon, marker):
        filename = f'{taxon}_{marker}_NCBI.summ'
        return filename
    
    def survey(self, taxon, marker):
        out_file = self.generate_filename(taxon, marker)
        search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
        search_record = Entrez.read(search)
        with open(out_file, 'w') as handle:
            handle.write('\n'.join(search_record['IdList']))
        return