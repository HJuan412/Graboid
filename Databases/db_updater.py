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
import urllib3
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
    http = urllib3.PoolManager()
    r = http.request('GET', apiurl, preload_content = False)
    
    with open(outfile, 'ab') as handle:
        for chunk in r.stream():
            handle.write(chunk)
    r.release_conn()
#%% BOLD
# def get_summary_data(taxon, outfile):
taxon = 'Nematoda'
apiurl =f'http://www.boldsystems.org/index.php/API_Public/specimen?taxon={taxon}&format=tsv'
http = urllib3.PoolManager()

r = http.request('GET', apiurl, preload_content = False)
outfile = 'nem_BOLD_summ.tsv'
with open(outfile, 'ab') as handle:
    i = 0
    for chunk in r.stream():
        print(i)
        i += 1
        handle.write(chunk)

r.release_conn()
# get_summary_data('Nematoda', 'nem_BOLD_summ.tsv')
#%%
import pandas as pd
bold_tab = pd.read_csv('nem_BOLD_summ.tsv', sep = '\t', encoding = 'latin-1') # latin-1 to parse BOLD files
#%% ENA
marker = '18S'
enaapiurl = f'https://www.ebi.ac.uk/ena/browser/api/tsv/textsearch?domain=embl&result=sequence&query=%22{taxon}%22%20AND%20%22{marker}%22'

dl_and_save(enaapiurl, 'ena_summary.tsv')
#%% NCBI
ncbi_mark = '18S ribosomal rna'
Entrez.email = "hernan.juan@gmail.com"
Entrez.api_key = "7c100b6ab050a287af30e37e893dc3d09008"
search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
search_record = Entrez.read(search)

#%%
