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
#%% variables
summary_dir = '/home/hernan/PROYECTOS/Graboid/Databases/22_9_2021-11_54_51/Summaries'
#%% functions
def list_summ_files(summ_dir):
    summ_files = glob(f'{summ_dir}/*summ')
    return summ_files

def build_summ_tab(summ_files):
    # get taxon, marker, database and path information for each file
    summ_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'File'])
    
    for file in summ_files:
        split_file = file.split('/')[-1].split('.summ')[0].split('_')
        row = {'Taxon':split_file[0],
               'Marker':split_file[1],
               'Database':split_file[2],
               'File':file}
        summ_tab = summ_tab.append(row, ignore_index=True)
    return summ_tab

def read_summ(summ_file, dbase):
    if dbase == 'BOLD':
        bold_tab = pd.read_csv(summ_file, sep = '\t',encoding = 'latin-1') # latin-1 to parse BOLD files
        accs = bold_tab['sampleid'].tolist()
    else:
        ncbi_tab = pd.read_csv(summ_file, sep = '\t') # this also loads ENA summaries
        accs = ncbi_tab.iloc[:,0].tolist()
    
    shortaccs = get_shortaccs(accs)
    acc_series = pd.Series(accs, index = shortaccs, name = dbase)
    return acc_series

def get_shortaccs(acclist):
    shortaccs = [acc.split('.')[0] for acc in acclist]
    return shortaccs

def make_shortacc_dict(acc_dict):
    db_shortaccs = {}
    for k,v in acc_dict.items():
        db_shortaccs[k] = get_shortaccs(v)
    
    return db_shortaccs

def fold_tab(db_tab):
    cols = db_tab.columns
    folded = pd.DataFrame(columns = ['Accesion', 'Database'])
    dbases = ['NCBI', 'ENA', 'BOLD']
    
#%%
def make_acc_tab(summ_tab):
    acc_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'Accession']) # will store accession numbers

    taxons = summ_tab['Taxon'].unique().tolist()
    for taxon in taxons:
        sub_tab1 = summ_tab.loc[summ_tab['Taxon'] == taxon]
    
        markers = sub_tab1['Marker'].unique().tolist()
        for marker in markers:
            sub_tab2 = sub_tab1.loc[sub_tab1['Marker'] == marker]
            
            sub_acc_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'Accession'])
            db_accs = []
            dbases = sub_tab2['Database'].unique().tolist()
            for dbase in dbases:
                file = sub_tab2.loc[sub_tab2['Database'] == dbase, 'File'].values[0]
                # db_accs[dbase] = read_summ(file, dbase)
                db_accs.append(read_summ(file, dbase))
            
            db_tab = pd.concat(db_accs, axis = 1)
            for db in ['NCBI', 'ENA', 'BOLD']:
                if db in db_tab.columns:
                    col = db_tab[db].dropna()
                    idx = set(col).difference(set(sub_acc_tab))
            return db_tab

    # for taxon in taxons:
    #     tax_tab = pd.DataFrame(columns = ['Taxon', 'Marker', 'Database', 'Accession']) # sub table containing entries for the given taxon, will be incorporated to acc_tab 
    #     for marker in markers:
    #         # TODO handle missing databases
    #         # retrieve summary files
    #         ena_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Marker'] == marker) & (summ_tab['Database'] == 'ENA'), 'File'].values[0]
    #         ncbi_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Marker'] == marker) & (summ_tab['Database'] == 'NCBI'), 'File'].values[0]
    
    #         # read ENA
    #         ena_tab = pd.read_csv(ena_file, sep ='\t')
    #         ena_accs = set(ena_tab['accession'])
    #         # read NCBI
    #         with open(ncbi_file, 'r') as handle:
    #             acc_list = handle.read().splitlines()
    #             ncbi_accs = set([acc.split('.')[0] for acc in acc_list])
            
    #         ncbi_tab = pd.DataFrame({'Taxon':taxon, 'Marker':marker, 'Database':'NCBI', 'Accession':list(ncbi_accs)})
    #         ena_tab = pd.DataFrame({'Taxon':taxon, 'Marker':marker, 'Database':'ENA', 'Accession':list(ena_accs.difference(ncbi_accs))})
            
    #         tax_tab = pd.concat([tax_tab, ncbi_tab, ena_tab])
    #     # read BOLD
        # bold_file = summ_tab.loc[(summ_tab['Taxon'] == taxon) & (summ_tab['Database'] == 'BOLD'), 'File'].values[0]
        # bold_tab = pd.read_csv(bold_file, sep = '\t',encoding = 'latin-1') # latin-1 to parse BOLD files
        # bold_accs = set(bold_tab['sampleid'])
        # core_accs = set(tax_tab.loc[tax_tab['Marker'] == 'COI', 'Accession'])
        # bold_tab = pd.DataFrame({'Taxon':taxon, 'Marker':'COI', 'Database':'BOLD', 'Accession':list(bold_accs.difference(core_accs))})
        # tax_tab = pd.concat([tax_tab, bold_tab])
    
    #     acc_tab = pd.concat([acc_tab, tax_tab])

#%% save results
out_dir = '/home/hernan/PROYECTOS/Graboid/Databases/Acc_lists'
t = datetime.now()
outfile = f'{out_dir}/accessions_{t.day}-{t.month}-{t.year}_{t.hour}-{t.minute}-{t.second}.csv'
acc_tab.to_csv(outfile)