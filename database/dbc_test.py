
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:10:36 2022

@author: hernan
"""

#%%

import os
import pandas as pd
import sys
sys.path.append('data_fetch/database_construction')

import surveyor
import lister
import fetcher
import taxonomist
import merger

fetcher.set_entrez()

#%% test params
taxon = 'platyhelminthes'
marker = '28s'
databases = ['NCBI', 'BOLD']

#%% make dirs
master_dir = f'{taxon}_{marker}'
out_dir = f'{master_dir}/out_dir'
tmp_dir = f'{master_dir}/tmp_dir'
warn_dir = f'{master_dir}/warn_dir'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(warn_dir, exist_ok=True)

#%% surveyor test
worker_surveyor = surveyor.Surveyor(taxon, marker, tmp_dir, warn_dir)
worker_surveyor.survey()

#% lister test
# TODO: fix lister
# worker_lister = lister.Lister(taxon, marker, tmp_dir, warn_dir)
# worker_lister.make_list()

#% fetcher test
# error en la descarga de ncbi, tuve que redefinir la tabla de accessos para omitir BOLD
worker_fetcher = fetcher.Fetcher2(taxon, marker
                                  , worker_surveyor.prelister.out_file, tmp_dir, warn_dir)
# worker_fetcher.load_accfile(worker_lister.out_file)
worker_fetcher.read_acclist(worker_surveyor.prelister.out_file)
worker_fetcher.fetch()

#% taxer test
worker_taxer = taxonomist.Taxonomist(taxon, marker, databases, tmp_dir, tmp_dir, warn_dir)
worker_taxer.taxing()

#% merger test
worker_merger = merger.Merger(taxon, marker, databases, tmp_dir, out_dir, warn_dir)
worker_merger.merge()

#TODO Remove gaps from bold sequences (blast doesn't like them), temporarly fixed with sed '/^>.*$/! s/\-//g' out_file.fasta > fixed_file