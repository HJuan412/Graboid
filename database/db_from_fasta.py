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

#%% load taxonomy data
names_tab = pd.read_csv('test/names.tsv', sep='\t', header=None, index_col=0, names='sciName TaxId'.split())
names_tab.loc[0] = 'unknown'
nodes_tab = pd.read_csv('test/nodes.tsv', sep='\t', header=None, index_col=0, names='TaxId Parent Rank'.split())

def detect_separator(line):
    """
    Detect separator characters from the source taxonomy table. Expect it to be
    either ',', ';', '\t'. Raise exception if none of those was used.

    """
    # criterion, separator should be one of these characters, splitting a line usingg the incorrect separator should return a 1 element lise
    seps = [',', ';', '\t']
    separator = ''
    max_len = 0
    for s in seps:
        if len(line.split(s)) > max_len:
            separator = s
            max_len = len(line.split(s))
    if max_len == 1:
        raise Exception("Can't recognize separator character. Be kind and ensure the taxonomy table uses one of ',', ';', or '\t' as separator character]")
    return separator
    
def parse_source_tax(tax_file):
    """
    Load the provided taxonomy table. Get the last three taxa for each record.
    Assume the first column corresponds to the sequence accession.

    Parameters
    ----------
    tax_file : str
        Path to the taxonomy file.

    Returns
    -------
    source_tax : pandas.DataFrame
        Dataframe with index : accession codes. 3 columns, last 3 taxa for each record

    """
    records = []
    with open(tax_file, 'r') as src:
        lines = src.read().splitlines()
        sep = detect_separator(lines[0])
        for line in lines:
            cols = line.split(sep)
            # select accession & last 3 taxa
            records.append([cols[0]] + cols[-3:])
    source_tax = pd.DataFrame(records).set_index(0)
    return source_tax

def get_acc_taxids(src_tax, names_tab):
    # select the lowest valid taxon for each record
    lowest_taxa = pd.Series('', index=src_tax.index)
    for idx, col in src_tax.T.iterrows():
        lowest_taxa.loc[col.isin(names_tab.TaxId)] = col
    # assign TaxId for each record
    acc_taxids = names_tab.reset_index().set_index('TaxId').loc[lowest_taxa.values].set_index(src_tax.index).sciName
    return acc_taxids

tax_file = '/home/hernan/PROYECTOS/Maestria/Silva/arb-silva.de_2024-01-22_id1299351/headers.csv'
source_tax = parse_source_tax(tax_file)
acc_taxids = get_acc_taxids(source_tax, names_tab)
#%% build lineage table
def build_lineage_table(taxid_tab, nodes_tab, *ranks):
    """
    Rebuild the lineage for each taxon present in the records.
    Include lineages of upper taxa

    Parameters
    ----------
    taxid_tab : pandas.DataFrame
        DataFrame containing the 3 last taxa for each record.
    nodes_tab : pandas.DataFrame
        DataFrame containing each taxon's parent TaxId and rank.
    *ranks : str
        Ranks to be included in the lineage table.

    Returns
    -------
    lineages : TYPE
        DESCRIPTION.
    real_taxids : TYPE
        DESCRIPTION.

    """
    # unfold lineages
    lineages = [unfold_lineage(taxid, nodes_tab, *ranks) for taxid in taxid_tab.unique()]
    lineages = pd.DataFrame(lineages, index = taxid_tab.unique())[list(ranks)]
    
    # some records have an assigned taxonomy at a lower rank than the ones specified in ranks
    # we need to retrieve the real taxid for each record (the lowest taxa within the specified ranks)
    real_taxids = pd.Series(index=lineages.index) # this will contain the value that needs to replace each "false" taxid
    for col in lineages.columns:
        valid_idxs = ~lineages[col].isna().values
        real_taxids.loc[valid_idxs] = lineages.loc[valid_idxs, col].values
    
    # repeated records produced by records with lower rank taxonomic assignments, clear them
    lineages.index = real_taxids.values.astype(int)
    lineages.drop_duplicates(inplace=True)
    # add lineages of higher rank taxa
    for idx, rk in enumerate(ranks[:-1]):
        trail = list(ranks)[:idx+1]
        for tax, subtab in lineages.groupby(rk):
            lineages.loc[tax, trail] = subtab[trail].iloc[0]
    lineages.sort_values(list(ranks)[::-1], inplace=True)
    lineages = lineages.fillna(0).astype(int)
    return lineages, real_taxids.astype(int)

ranks = 'phylum class order family genus species'.split()
lineages, real_taxids = build_lineage_table(acc_taxids, nodes_tab, *ranks)

#%% build taxonomy table
taxonomy = real_taxids.loc[acc_taxids].astype(int)
taxonomy.index = source_tax.index
#%% build name table
names = names_tab.loc[np.unique(lineages), 'TaxId']
