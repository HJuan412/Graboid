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

# TODO: remove this
# #%% variables
# Entrez.email = "hernan.juan@gmail.com"
# Entrez.api_key = "7c100b6ab050a287af30e37e893dc3d09008"

# #%% List current databases
# data = '/home/hernan/PROYECTOS/Maestria/Data/18S_Nematoda.fasta'
# def make_acclist(data):
#     with open(data, 'r') as handle:
#         data_parser = SeqIO.parse(handle, 'fasta')
#         acc_list = [record.id for record in data_parser]
#     return acc_list

# acc_list = make_acclist(data)

#%% functions
def dl_and_save(apiurl, outfile):
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
    with open(outfile, 'ab') as handle:
        for chunk in r.stream():
            handle.write(chunk)
    # close connection
    r.release_conn()
    return
        
def generate_filename(tax, mark, db):
    filename = f'{tax}_{mark}_{db}.summ'
    return filename

# url generators
def get_bold_url(taxon):
    apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
    return apiurl

def get_ena_url(taxon, marker):
    apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
    return apiurl

# survey
def survey_bold(taxon, outfile):
    """
    Performs a search using the BOLD api

    Parameters
    ----------
    taxon : str
        Taxon to search for.
    outfile : str
        Output file name.

    Returns
    -------
    None.

    """
    apiurl = get_bold_url(taxon)
    dl_and_save(apiurl, outfile)

def survey_ena(taxon, marker, outfile):
    """
    Performs a search using the ENA api

    Parameters
    ----------
    taxon : str
        Taxon to search for.
    marker : str
        Marker to search for.
    outfile : str
        Output file name.

    Returns
    -------
    None.

    """
    apiurl = get_ena_url(taxon, marker)
    dl_and_save(apiurl, outfile)

def survey_ncbi(taxon, marker, outfile):
    """
    Performs a search in the NCBI database using the Entrez api

    Parameters
    ----------
    taxon : str
        Taxon to search for.
    marker : str
        Marker to search for.
    outfile : str
        Output file name.

    Returns
    -------
    None.

    """
    search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
    search_record = Entrez.read(search)
    with open(outfile, 'w') as handle:
        handle.write('\n'.join(search_record['IdList']))

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
    for taxon in taxons:
        # BOLD search
        if bold:
            print(f'Surveying BOLD database for {taxon}...')
            apiurl = f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
            outfile = generate_filename(taxon, '', 'BOLD')
            dl_and_save(apiurl, f'{summ_dir}/{outfile}')
            # bold_tab = pd.read_csv('nem_BOLD_summ.tsv', sep = '\t', encoding = 'latin-1') # latin-1 to parse BOLD files
    
        for marker in markers:
            # ENA search
            if ena:
                print(f'Surveying ENA database for {taxon} and {marker}...')
                apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
                outfile = generate_filename(taxon, marker, 'ENA')
                dl_and_save(apiurl, f'{summ_dir}/{outfile}')
    
            # NCBI search
            if ncbi:
                print(f'Surveying NCBI database for {taxon} and {marker}...')
                outfile = generate_filename(taxon, marker, 'NCBI')
                survey_ncbi(taxon, marker, f'{summ_dir}/{outfile}')