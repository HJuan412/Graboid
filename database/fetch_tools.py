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

shell_path = os.path.dirname(__file__) + '/get_taxdmp.sh'
def get_taxdmp(out_dir):
    """Retrieve and format the NCBI taxdmp files and store them in out_file"""
    subprocess.run([shell_path, out_dir])
    names_file = f'{out_dir}/names.tsv'
    nodes_file = f'{out_dir}/nodes.tsv'
    return names_file, nodes_file

#%%
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

def merge_records(ncbi_seqs, bold_seqs, out_dir, db_name):
    os.makedirs(out_dir, exist_ok=True)
    out_file = f'{out_dir}/{db_name}.fasta'
    
    shutil.copy(ncbi_seqs, out_file)
    with open(bold_seqs, 'r') as bold_handle:
        bold_content = bold_handle.read()
        with open(out_file, 'a') as out_handle:
            out_handle.write(bold_content)
    return out_file

def merge_taxonomes(ncbi_tax, ncbi_lin, ncbi_nam, bold_tax, bold_lin, bold_nam, out_dir, db_name):
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