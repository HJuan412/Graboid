#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:33:36 2021

@author: hernan
"""

#%% libraries
from glob import glob
from datetime import datetime
import pandas as pd
#%%
summary_dir = '/home/hernan/PROYECTOS/Graboid/Databases/Summaries'

summ_files = glob(f'{summary_dir}/*summ')

summ_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'File'])

for file in summ_files:
    split_file = file.split('/')[-1].split('_')
    row = {'Taxon':split_file[0],
           'Marker':split_file[1],
           'Database':split_file[2],
           'File':file}
    summ_tab = summ_tab.append(row, ignore_index=True)

#%%
taxons = ['Nematoda', 'Platyhelminthes']
markers = ['18S', '28S', 'COI']

acc_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'Accession'])

for taxon in taxons:
    tax_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'Accession'])
    for marker in markers:
        ena_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Marker'] == marker) & (summ_tab['Database'] == 'ENA'), 'File'].values[0]
        ncbi_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Marker'] == marker) & (summ_tab['Database'] == 'NCBI'), 'File'].values[0]

        # read ENA
        ena_tab = pd.read_csv(ena_file, sep ='\t')
        ena_accs = set(ena_tab['accession'])
        # read NCBI
        with open(ncbi_file, 'r') as handle:
            acc_list = handle.read().splitlines()
            ncbi_accs = set([acc.split('.')[0] for acc in acc_list])
        
        ncbi_tab = pd.DataFrame({'Taxon':taxon, 'Marker':marker, 'Database':'NCBI', 'Accession':list(ncbi_accs)})
        ena_tab = pd.DataFrame({'Taxon':taxon, 'Marker':marker, 'Database':'ENA', 'Accession':list(ena_accs.difference(ncbi_accs))})
        
        tax_tab = pd.concat([tax_tab, ncbi_tab, ena_tab])
    # read BOLD
    bold_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Database'] == 'BOLD'), 'File'].values[0]
    bold_tab = pd.read_csv(bold_file, sep = '\t',encoding = 'latin-1') # latin-1 to parse BOLD files
    bold_accs = set(bold_tab['sampleid'])
    core_accs = set(tax_tab.loc[tax_tab['Marker'] == 'COI', 'Accession'])
    bold_tab = pd.DataFrame({'Taxon':taxon, 'Marker':'COI', 'Database':'BOLD', 'Accession':list(bold_accs.difference(core_accs))})
    tax_tab = pd.concat([tax_tab, bold_tab])

    acc_tab = pd.concat([acc_tab, tax_tab])

#%% save results
out_dir = '/home/hernan/PROYECTOS/Graboid/Databases/Acc_lists'
t = datetime.now()
outfile = f'{out_dir}/accessions_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}.csv'
acc_tab.to_csv(outfile)