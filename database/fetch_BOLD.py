#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:31:26 2024

@author: hernan

Perform search, retrieve sequence and taxonomy data from NCBI
"""

import logging
import numpy as np
import os
import pandas as pd
import re
import time
import urllib3
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import fetch_tools

logger = logging.getLogger('Graboid.database.BOLD')

#%% Retrieve records from the BOLD database
"""
Retrieve records matching the cross-search for the given taxon-marker

Parameters
----------
taxon : str
    Taxon to look for.
marker : str
    Marker to look for.
out_dir : str
    Path to the directory to contain the retrieved files.
max_attempts : int, optional
    Number of attempts to perform. The default is 3.

Raises
------
Exception
    Abort operation after the given amount of failed requests.

Returns
-------
out_file : str
    Path to the file containing the retrieved records.

"""
def fetch_in(taxon, marker, out_file, max_attempts=3):
    
    attempt = 1
    while attempt <= max_attempts:
        # connect to server & send request
        http = urllib3.PoolManager()
        apiurl = f'http://www.boldsystems.org/index.php/API_Public/combined?taxon={taxon}&marker={marker}&format=tsv' # this line downloads sequences AND taxonomies
        r = http.request('GET', apiurl, preload_content = False)
        
        # stream request and store in outfile
        try:
            with open(out_file, 'wb') as handle:
                for chunk in r.stream():
                    handle.write(chunk)
            # close connection
            r.release_conn()
            return out_file
        except Exception:
            # update attempt count
            print(f'Download of {taxon} {marker} records from BOLD interrupted. {max_attempts - attempt} attempts remaining')
            attempt += 1
        time.sleep(3)
    raise Exception('Failed to retrieve records after {max_attempts} attempts.')

def postprocess_data(source_file, exclude):
    """
    Parse the retrieved BOLD database records. Extract sequence and taxonomic
    data.

    Parameters
    ----------
    source_file : str
        Path to the file of retrieved BOLD records.
    exclude : list
        List of accession codes to be excluded from the source file (already retrieved from NCBI).

    Returns
    -------
    records : list
        List of extracted sequence records.
    tax_subtab : pandas.DataFrame
        DataFrame containing the taxonomic data from the selected records.
    nseqs : int
        Number of records retrieved from the BOLD database.

    """
    
    # read table and get records in acc_list
    bold_tab = pd.read_csv(source_file, sep = '\t', encoding = 'latin-1', dtype = str) # latin-1 to parse BOLD files
    bold_tab.set_index('sampleid', inplace=True)
    # exclude already present records
    bold_tab = bold_tab.loc[bold_tab.index.difference(exclude)]
    
    # get sequences and taxonomy table
    seq_subtab = bold_tab['nucleotides']
    tax_subtab = bold_tab.loc[:, 'phylum_taxID':'subspecies_name']
    
    # save results
    records = []
    for acc, seq in seq_subtab.iteritems():
        records.append(SeqRecord(id=acc, seq = Seq(seq.replace('-', '')), description = ''))
    nseqs = len(seq_subtab)
    return records, tax_subtab, nseqs

"""
Retrieve sequence and taxonomy data from the BOLD repository.
Make two download attempts.
Generates 2 files:
    seq_file: contains the sequence data in fasta format.
    tax_file: contains a table with the taxonomic data of the selected records

Parameters
----------
taxon : str
    Taxon to look for.
marker : str
    Marker to look for.
out_dir : str
    Path to the directory to contain the generated files.
exclude : list
    List of accession codes to be excluded from the source file (already retrieved from NCBI).
tmp_dir : str, optional
    Path to the directory to contain the temporal files. If none is given, temporal files will be stored in the output directory.
warn_dir : str, optional
    Path to the directory to contain the warning file. If none is given, the warning file is stored in the output directory.
max_attempts : int, optional
    Number of attempts to perform. The default is 3.
rm_temp : bool, optional
    Delete temporary record file at the end of the task. The default is True.

Raises
------
Exception
    Raise exception if no record can be retrieved.

Returns
-------
out_seqs : str
    Path to the generated sequence file.
out_taxs : str
    Path to the generated taxonomy file.
nseqs : int
    Number of records retrieved from the BOLD database.

"""
def fetch(taxon, marker, out_dir, tmp_dir, warn_dir, exclude=None, max_attempts=3, rm_temp=True, db_name='BOLD'):
    out_seqs = f'{out_dir}/{db_name}.seqs'
    out_taxs = f'{out_dir}/{db_name}.taxs'
    records_file = f'{tmp_dir}/BOLD.records'
    # Make two download passes, only give up when a sequence chunk fails to be retrieved both times
    done=False
    # first pass
    for pss in ('first', 'second'):
        try:
            tmp_file = fetch_in(taxon, marker, records_file, max_attempts)
            done = True
            break
        except Exception as excp:
            logger.warning(f'BOLD {pss} pass. {excp}')
    if not done:
        raise Exception('Failed to retrieve sequences from BOLD database.')
    
    # extract sequence and taxonomic data
    records, tax_tab, nseqs = postprocess_data(records_file, exclude)
    logger.info(f'Retrieved {len(records)} sequence records from the BOLD database.')
    
    # save results
    with open(out_seqs, 'w') as out_handle:
        SeqIO.write(records, out_handle, 'fasta')
    tax_tab.to_csv(out_taxs)
    if rm_temp:
        os.remove(tmp_file)
        
    return out_seqs, out_taxs, nseqs

#%% Process taxonomy data
def parse_source_tax(tax_file, ranks):
    """
    Extract the relevant taxonomc data from the taxonomy table retrieved from
    the BOLD database.

    Parameters
    ----------
    tax_file : str
        Path to the file containing the retrieved taxonomy data.
    ranks : str
        Ranks to extract from the taxonomy table.

    Raises
    ------
    Exception
        Raise an exception it the provided file is empty or none of the
        requested ranks are present in the file.

    Returns
    -------
    source_tax : pandas.DataFrame
        DataFrame containing only the columns corresponded to the requested
        taxonomic ranks.

    """
    
    # load raw taxonomy data
    bold_tab = pd.read_csv(tax_file, index_col = 0)
    if len(bold_tab) == 0:
        # interrupt execution and let constructor know
        raise Exception(f'Taxonomy source file {tax_file} is empty')
    
    # filter columns
    valid_ranks = []
    invalid_ranks = []
    cols = [] # list columns to be extracted
    for rk in ranks:
        if rk+'_taxID' in bold_tab.columns:
            valid_ranks.append(rk)
            cols.append(f'{rk}_name')
        else:
            invalid_ranks.append(rk)
    source_tax = bold_tab[cols]
    
    if len(invalid_ranks) > 0:
        logger.warning(f'Ranks {" ".join(invalid_ranks)} are not found in the BOLD table and will be ommited')
    if len(valid_ranks) == 0:
        raise Exception('None of the given ranks ({" ".join(self.ranks)}) were found in the BOLD table')
    return source_tax

def get_acc_taxids(src_tax, names_tab):
    # select the lowest valid taxon for each BOLD record
    lowest_taxa = pd.Series('', index=src_tax.index)
    for idx, col in src_tax.T.iterrows():
        lowest_taxa.loc[col.isin(names_tab.TaxId)] = col
    # assign TaxId for each record
    acc_taxids = names_tab.reset_index().set_index('TaxId').loc[lowest_taxa.values].set_index(src_tax.index).sciName
    return acc_taxids

def build_lineage_table(taxid_tab, nodes_tab, ranks):
    """
    Rebuild the lineage for each taxon present in the records.
    Include lineages of upper taxa.
    The generated table uses the NCBI taxonomic IDs

    Parameters
    ----------
    taxid_tab : pandas.DataFrame
        DataFrame containing the 3 last taxa for each record.
    nodes_tab : pandas.DataFrame
        DataFrame containing each taxon's parent TaxId and rank.
    ranks : str
        Ranks to be included in the lineage table.
    
    Returns
    -------
    lineages : pandas.DataFrame
        DataFrame containing the lineages of all taxa present among the given records.
    real_taxids : pandas.Series
        Series containing the adjusted taxonomic Id for every record.
    
    """
    # unfold lineages
    lineages = [fetch_tools.unfold_lineage(taxid, nodes_tab, ranks) for taxid in taxid_tab.unique()]
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

def build_taxonomy_table(acc_taxids, real_taxids):
    # build the dataframe containing the lowest valid taxId for each record
    taxonomy = real_taxids.loc[acc_taxids].astype(int)
    taxonomy.index = acc_taxids.index
    return taxonomy

def build_name_table(lineage_tab, names_tab):
    # build the dataframe containing the TaxId-Scientific name pairs
    names = names_tab.loc[np.unique(lineage_tab)]
    return names

def arrange_taxonomy(out_dir, tax_source, names_tab, nodes_tab, ranks):
    """
    Generate the taxonomy, lineage, and names tables for the generated records.

    Parameters
    ----------
    out_dir : str
        Path to the directory to contain the generated files.
    tax_source : str
        Path to the file containing the retrieved record taxonomic IDs.
    names_tab : str
        Path to the file containing the NCBI TaxId:Scientific name pairs.
    nodes_tab : str
        Path to the file containing the cladistic information for each taxon.
    ranks : str
        Ranks to be included in the lineage table.

    Returns
    -------
    lineage_file : str
        Path to the file containing the generated lineage table.
    taxonomy_file : str
        Path to the file containing the accession-TaxId mapping.
    name_file : str
        Path to the file containing the TaxId-Scientific name mapping.

    """
    # prepare output directory & files
    os.makedirs(out_dir, exist_ok=True)
    lineage_file=f'{out_dir}/BOLD.lineage'
    taxonomy_file=f'{out_dir}/BOLD.taxonomy'
    name_file=f'{out_dir}/BOLD.names'
    
    # load names and nodes tables
    names_tab, nodes_tab = fetch_tools.read_taxdmp(names_tab, nodes_tab)
    
    # parse retrieved taxonomy data
    source_taxonomy = parse_source_tax(tax_source, ranks)
    acc_taxids = get_acc_taxids(source_taxonomy, names_tab)
    
    # build taxonomy tables
    lineages, real_taxids = build_lineage_table(acc_taxids, nodes_tab, ranks)
    taxonomy_table = build_taxonomy_table(acc_taxids, real_taxids)
    name_table = build_name_table(lineages, names_tab)
    
    # save files
    lineages.to_csv(lineage_file)
    taxonomy_table.to_csv(taxonomy_file)
    name_table.to_csv(name_file)
    
    return lineage_file, taxonomy_file, name_file

#%%
"""
Retrieve sequence and taxonomy data from the BOLD database.

Parameters
----------
taxon : str
    Taxon to look for.
marker : str
    Marker to look for.
out_dir : str
    Path to the directory to contain the generated files.
names_tab : str
    Path to the file containing the NCBI TaxId:Scientific name pairs.
nodes_tab : str
    Path to the file containing the cladistic information for each taxon.
exclude : list
    List of accession codes to be excluded from the source file (already retrieved from NCBI).
tmp_dir : str, optional
    Path to the directory to contain the temporal files. If none is given, temporal files will be stored in the output directory.
warn_dir : str, optional
    Path to the directory to contain the warning file. If none is given, the warning file is stored in the output directory.
max_attempts : int, optional
    Number of attempts to perform. The default is 3.
rm_temp : bool, optional
    Delete temporary record file at the end of the task. The default is True.
ranks : str
    Ranks to extract from the taxonomy table.

Returns
-------
out_seqs : str
    Path to the generated sequence file.
out_taxs : str
    Path to the generated taxonomy file.
lineage_file : str
    Path to the file containing the generated lineage table.
taxonomy_file : str
    Path to the file containing the accession-TaxId mapping.
name_file : str
    Path to the file containing the TaxId-Scientific name mapping.
nseqs : int
    Number of records retrieved from the BOLD database.

"""
def retrieve_data(taxon,
                  marker,
                  out_dir,
                  names_tab,
                  nodes_tab,
                  tmp_dir,
                  warn_dir,
                  exclude=None,
                  max_attempts=3,
                  rm_temp=True,
                  ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
                  db_name='BOLD'):
    # ensure directory names are properly formatted
    out_dir = re.sub('/$', '', out_dir)
    tmp_dir = re.sub('/$', '', tmp_dir)
    warn_dir = re.sub('/$', '', warn_dir)
    # retrieve data
    out_seqs, out_taxs, nseqs = fetch(taxon, marker, out_dir, exclude, tmp_dir, warn_dir, max_attempts, rm_temp, db_name)
    
    # generate taxonomy files
    lineage_file, taxonomy_file, name_file = arrange_taxonomy(out_dir, out_taxs, names_tab, nodes_tab, ranks)
    return out_seqs, out_taxs, lineage_file, taxonomy_file, name_file, nseqs