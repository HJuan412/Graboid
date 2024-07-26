#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:00:36 2024

@author: hernan

Retrieve taxdmp files from the NCBI ftp server
"""

import os
import pandas as pd
import shutil
import subprocess
from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp

shell_path = os.path.dirname(__file__) + '/get_taxdmp.sh'
def get_taxdmp(out_dir):
    """Retrieve and format the NCBI taxdmp files and store them in out_file"""
    names_file = f'{out_dir}/names.tsv'
    nodes_file = f'{out_dir}/nodes.tsv'
    if not os.path.isfile(names_file) and not os.path.isfile(nodes_file):
        subprocess.run([shell_path, out_dir])
    return names_file, nodes_file

def read_taxdmp(names_tab, nodes_tab):
    names_tab = pd.read_csv(names_tab, sep='\t', names='TaxId SciName'.split(), index_col=0)
    names_tab.loc[0] = 'Unknown'
    nodes_tab = pd.read_csv(nodes_tab, sep='\t', names='TaxId Parent Rank'.split(), index_col=0)
    nodes_tab.loc[0] = [0, 'No_Rank']
    return names_tab, nodes_tab
#%%
def unfold_lineage(taxid, node_tab, ranks):
    """
    Get the complete lineage for the specified ranks for the given taxid

    Parameters
    ----------
    taxid : int
        Taxid for which the lineage will be unfolded.
    node_tab : pandas.DataFrane
        Nodes dataframe.
    ranks : str
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

def count_seqs(fasta_file):
    counts = 0
    with open(fasta_file, 'r') as f:
        for rec in sfp(f):
            counts += 1
    return counts

def merge_records(ncbi_seqs, bold_seqs, out_dir, db_name):
    os.makedirs(out_dir, exist_ok=True)
    out_file = f'{out_dir}/{db_name}.fasta'
    
    shutil.copy(ncbi_seqs, out_file)
    with open(bold_seqs, 'r') as bold_handle:
        bold_content = bold_handle.read()
        with open(out_file, 'a') as out_handle:
            out_handle.write(bold_content)
    nseqs = count_seqs(out_file)
    return out_file, nseqs

def merge_taxonomies(ncbi_tax, ncbi_lin, ncbi_nam, bold_tax, bold_lin, bold_nam, out_dir, db_name):
    os.makedirs(out_dir, exist_ok=True)
    out_tax = f'{out_dir}/{db_name}.taxonomy'
    out_lin = f'{out_dir}/{db_name}.lineage'
    out_nam = f'{out_dir}/{db_name}.names'
    
    merged_tax = pd.concat([pd.read_csv(ncbi_tax, index_col=0),
                            pd.read_csv(bold_tax, index_col=0)])
    merged_lin = pd.concat([pd.read_csv(ncbi_lin, index_col=0),
                            pd.read_csv(bold_lin, index_col=0)]).drop_duplicates()
    merged_nam = pd.concat([pd.read_csv(ncbi_nam, index_col=0),
                            pd.read_csv(bold_nam, index_col=0)]).drop_duplicates()
    
    merged_tax.to_csv(out_tax)
    merged_lin.to_csv(out_lin)
    merged_nam.to_csv(out_nam)
    return out_tax, out_lin, out_nam

def count_ranks(taxonomy_tab, lineage_tab):
    tax_tab = pd.read_csv(taxonomy_tab, index_col=0).iloc[:,0]
    lin_tab = pd.read_csv(lineage_tab, index_col=0).loc[tax_tab.values]
    rk_counts = {}
    for rk in lin_tab.columns:
        rk_counts[rk] = len(lin_tab.loc[lin_tab[rk] != 0][rk].unique())
    return rk_counts