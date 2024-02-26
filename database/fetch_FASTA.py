#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:45:20 2024

@author: hernan

Build Graboid database from a provided fasta file.
A taxonomy table must be provided 
"""

import logging
import numpy as np
import os
import pandas as pd
import shutil
from database.fetch_tools import unfold_lineage

logger = logging.getLogger('Graboid.database.FASTA')
#%%
def fetch(fasta_file, out_dir, mv=False):
    """
    Move or copy the source fasta file to the database destination.

    Parameters
    ----------
    fasta_file : str
        Path to the fasta file to be used in the database.
    out_dir : str
        Path to destination directory for the fasta file.
    mv : Bool, optional
        Set True to move the fasta file into its destination location instead of copying it. The default is False.

    Returns
    -------
    None.

    """
    os.makedirs(out_dir, exist_ok=True)
    out_seqs = f'{out_dir}/FASTA.seqs'
    if mv:
        shutil.move(fasta_file, out_seqs)
    else:
        shutil.copy(fasta_file, out_seqs)
    return out_seqs

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
    lineages : pandas.DataFrame
        DataFrame containing the lineages of all taxa present among the given records.
    real_taxids : pandas.Series
        Series containing the adjusted taxonomic Id for every record.
    
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

#%% build taxonomy table
def build_taxonomy_table(acc_taxids, real_taxids):
    taxonomy = real_taxids.loc[acc_taxids].astype(int)
    taxonomy.index = acc_taxids.index
    return taxonomy

#%% build name table
def build_name_table(lineage_tab, names_tab):
    names = names_tab.loc[np.unique(lineage_tab), 'TaxId']
    return names

def arrange_taxonomy(out_dir, tax_source, names_tab, nodes_tab, *ranks):
    # prepare output directory & files
    os.makedirs(out_dir, exist_ok=True)
    lineage_file=f'{out_dir}/FASTA.lineage'
    taxonomy_file=f'{out_dir}/FASTA.taxonomy'
    name_file=f'{out_dir}/FASTA.names'
    
    # parse taxonomy data
    source_taxonomy = parse_source_tax(tax_source)
    acc_taxids = get_acc_taxids(source_taxonomy, names_tab)
    
    # build taxonomy tables
    lineages, real_taxids = build_lineage_table(acc_taxids, nodes_tab, *ranks)
    taxonomy_table = build_taxonomy_table(acc_taxids, real_taxids)
    name_table = build_name_table(lineages, names_tab)
    
    # save files
    lineages.to_csv(lineage_file)
    taxonomy_table.to_csv(taxonomy_file)
    name_table.to_csv(name_file)
    
    return lineage_file, taxonomy_file, name_file

#%%
def retrieve_data(fasta_file, taxonomy_file, out_dir, names_tab, nodes_tab, *ranks):
    # retrieve data
    out_seqs = fetch(fasta_file, out_dir)
    
    # generate taxonomy files
    lineage_file, taxonomy_file, name_file = arrange_taxonomy(out_dir, taxonomy_file, names_tab, nodes_tab, ranks)
    return out_seqs, lineage_file, taxonomy_file, name_file
