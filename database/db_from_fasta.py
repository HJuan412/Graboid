#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:45:20 2024

@author: hernan

Build Graboid database from a provided fasta file.
A taxonomy table must be provided 
"""

import numpy as np
import os
import pandas as pd
import shutil
import subprocess

shell_path = os.path.dirname(__file__) + '/get_taxdmp.sh'

# copy or move fasta file to the graboid DATA directory
# get NCBI taxonomy code for taxa presented in taxonomy table
    # generate warning file if any taxa is not present in genbank

# output files:
    # fasta file
    # Lineage table
    # Taxonomy table
    # names table

def move_fasta(fasta_file, destination, mv=False):
    if mv:
        shutil.move(fasta_file, destination)
    else:
        shutil.copy(fasta_file, destination)

def get_taxdmp(out_dir):
    """Retrieve and format the NCBI taxdmp files and store them in out_file"""
    subprocess.run([shell_path, out_dir])

def unfold_lineage(taxid, node_tab, *ranks):
    """
    Get the complete lineage for the specified ranks for the given taxid

    Parameters
    ----------
    taxid : int
        Taxid for which the lineage will be unfolded.
    node_tab : pandas.DataFrane
        Nodes dataframe.
    *ranks : str
        Ranks to be included in the lineage, all lowercase.

    Returns
    -------
    lineage : dict
        Lineage of the given taxid. Dict with rank:taxid key:value pairs.
        Missing ranks are not included

    """
    ranks = ranks[0]
    lineage = {}
    while len(lineage) < len(ranks) and node_tab.loc[taxid, 'Parent'] != 1:
        rk = node_tab.loc[taxid, 'Rank']
        if rk in ranks:
            lineage[rk] = taxid
        taxid = node_tab.loc[taxid, 'Parent']
    return lineage

def retrieve_lineages(taxids, node_tab, *ranks):
    """
    Retrieve the lineage for each record for the specified ranks

    Parameters
    ----------
    taxids : pandas.Series
        Series containing the Taxid from the lowest assigned taxon for each
        record.
    node_tab : pandas.DataFrane
        Nodes dataframe.
    *ranks : str
        Taxonomic ranks to include in the lineage (lower case).

    Returns
    -------
    lineages : pandas.DataFrame
        DataFrame containing the full lineage for all unique taxa represented
        in the records.

    """
    # for each located taxid, get the full lineage for the specified ranks
    lineages = pd.DataFrame(index=taxids, columns=ranks)
    for ln in lineages.index:
        lineages.loc[ln] = unfold_lineage(ln, node_tab, *ranks)
    return lineages

#%%
names_tab = pd.read_csv('test/names.tsv', sep='\t', header=None, index_col=0, names='sciName TaxId'.split())
names_tab.loc[0] = 'unknown'
nodes_tab = pd.read_csv('test/nodes.tsv', sep='\t', header=None, index_col=0, names='TaxId Parent Rank'.split())
tax_file = '/media/hernan/disco_viejo/PROYECTOS/Maestria/Silva/arb-silva.de_2024-01-22_id1299351/headers.csv'
sep=';'
records = []
with open(tax_file, 'r') as h:
    for row in h.read().splitlines():
        # get last 3 taxa from each record, should be enough to establish a valid lineage
        cols = row.split(sep)
        records.append([cols[0]] + cols[-3:])
tax_tab = pd.DataFrame(records).set_index(0)

lowest_taxa = pd.Series('', index=tax_tab.index)
for col in tax_tab.columns:
    lowest_taxa.loc[tax_tab[col].isin(names_tab.TaxId.values)] = tax_tab[col]
taxids = names_tab.reset_index().set_index('TaxId').loc[lowest_taxa.values].set_index(tax_tab.index).sciName
#%% build lineage table
ranks = 'phylum class order family genus species'.split()

lineages = [unfold_lineage(taxid, nodes_tab, ranks) for taxid in taxids.unique()]
lineages = pd.DataFrame(lineages, index = taxids.unique())[ranks]

real_taxids = pd.Series(index=lineages.index)
for col in lineages.columns:
    valid_idxs = ~lineages[col].isna().values
    real_taxids.loc[valid_idxs] = lineages.loc[valid_idxs, col].values

lineages.index = real_taxids.values.astype(int)
lineages.drop_duplicates(inplace=True)
lineages.sort_values(ranks[::-1], inplace=True)
lineages = lineages.fillna(0).astype(int)
for idx, rk in enumerate(ranks[:-1]):
    trail = ranks[:idx+1]
    for tax, subtab in lineages.groupby(rk):
        lineages.loc[tax, trail] = subtab[trail].iloc[0]
#%% build taxonomy table
taxonomy = real_taxids.loc[taxids].astype(int)
taxonomy.index = tax_tab.index
#%% build name table
names = names_tab.loc[np.unique(lineages), 'TaxId']
