#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:05:58 2021

@author: hernan
Build and maintain databases
"""

#%% libraries
from Bio import Entrez
from Bio import SeqIO
from datetime import datetime
import os
import urllib3

#%% variables
Entrez.email = "hernan.juan@gmail.com"
Entrez.api_key = "7c100b6ab050a287af30e37e893dc3d09008"

#%% List current databases
data = '/home/hernan/PROYECTOS/Maestria/Data/18S_Nematoda.fasta'
def make_acclist(data):
    with open(data, 'r') as handle:
        data_parser = SeqIO.parse(handle, 'fasta')
        acc_list = [record.id for record in data_parser]
    return acc_list

acc_list = make_acclist(data)

#%% functions
def dl_and_save(apiurl, outfile):
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

def ncbi_search(taxon, marker, outfile):
    search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
    search_record = Entrez.read(search)
    with open(outfile, 'w') as handle:
        handle.write('\n'.join(search_record['IdList']))
        
def generate_filename(tax, mark, db, timereg = True):
    t = datetime.now()
    filename = f'{tax}_{mark}_{db}_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}.summ'
    return filename

#%%
summary_dir = '/home/hernan/PROYECTOS/Graboid/Databases/Summaries'
taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']

bold = True
ena = False
ncbi = False

for taxon in taxons:
    # BOLD search
    if bold:
        apiurl =f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
        outfile = generate_filename(taxon, '', 'BOLD')
        dl_and_save(apiurl, f'{summary_dir}/{outfile}')
        # bold_tab = pd.read_csv('nem_BOLD_summ.tsv', sep = '\t', encoding = 'latin-1') # latin-1 to parse BOLD files

    for marker in markers:
        # ENA search
        if ena:
            apiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'
            outfile = generate_filename(taxon, marker, 'ENA')
            dl_and_save(apiurl, )

        # NCBI search
        if ncbi:
            outfile = generate_filename(taxon, marker, 'NCBI')
            ncbi_search(taxon, marker, f'{summary_dir}/{outfile}')