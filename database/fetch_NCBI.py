#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:41:43 2024

@author: hernan

Perform search, retrieve sequence and taxonomy data from BOLD
"""

import concurrent.futures
import logging
import numpy as np
import os
import pandas as pd
import re
import time
from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import fetch_tools

logger = logging.getLogger('Graboid.database.fetch_NCBI')

#%% retrieve records from the NCBI database
def survey(taxon, marker, max_attempts=3):
    """
    Perform a cross-term search for the given taxon-marker pair in the NCBI
    database.

    Parameters
    ----------
    taxon : str
        Taxon to look for.
    marker : str
        Marker to look for.
    max_attempts : int, optional
        Number of attempts to perform. The default is 3.

    Raises
    ------
    Exception
        Abort the search after the specified number of failed attempts.

    Returns
    -------
    search_record : list
        List of accession codes returned from the search.

    """
    attempt = 1
    while attempt <= max_attempts:
        # use entrez API to download summary
        try:
            search = Entrez.esearch(db='nucleotide', term = f'{taxon}[Organism] AND {marker}[all fields]', idtype = 'acc', retmax="100000")
            search_record = Entrez.read(search)['IdList']
            return search_record
        except Exception as excp:
            logger.warning(f'Download of {taxon} {marker} interrupted (Exception: {excp}), {max_attempts - attempt} attempts remaining')
            attempt += 1
            time.sleep(3)
    raise Exception('Failed to perform survey after {max_attempts} attempts.')

def acc_slicer(acc_list, chunksize):
    # slice the list of accessions into chunks
    n_seqs = len(acc_list)
    for i in np.arange(0, n_seqs, chunksize):
        yield acc_list[i:i+chunksize]

def retrieve(accs, tries=3, tag=0):
    # retrieve a single chunk data (<accs>), perform up to <tries> attempts before giving up
    # tag is used to illustrate the retrieval progress to the user
    failed = accs # empty this list if chunk is succesfully retrieved
    for t in range(1, tries+1):
        print(f'Retrieving chunk {tag}...')
        try:
            # contact NCBI and attempt download
            seq_handle = Entrez.efetch(db = 'nucleotide', id = accs, rettype = 'fasta', retmode = 'xml')
            seq_recs = Entrez.read(seq_handle)
            failed = [] # empty failed list!
            break
        except:
            print(f'Chunk {tag}: attempt {t} of {tries} failed')
            time.sleep(3)
            continue
    if len(failed) > 0:
        logger.warning(f'Failed to retreive chunk {tag} after {tries} attempts.')
    
    # generate seq records and retrieve taxids
    seqs = []
    taxids = []
    for acc, seq in zip(accs, seq_recs):
        seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence']), description = ''))
        taxids.append(seq['TSeq_taxid'])
    taxids = pd.Series(taxids, index=accs)
    return seqs, taxids, failed

def retrieve_pass(acc_list, out_seqs, out_taxs, chunk_size=500, max_attempts=3, workers=1):
    # perform a retrieval pass, returns list of failed records
    # downloads are done in parallel
    failed_accs = []
    n_chunks = np.ceil(len(acc_list) / chunk_size)
    # attempt to retrieve data
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(retrieve, chunk, max_attempts, f'{idx+1} of {n_chunks}') for idx, chunk in enumerate(acc_slicer(acc_list, chunk_size))]
        for future in concurrent.futures.as_completed(futures):
            seqs, taxids, failed = future.result()
            # save seq recods to fasta
            with open(out_seqs, 'a') as out_handle0:
                SeqIO.write(seqs, out_handle0, 'fasta')
            # save tax table to csv
            taxids.to_csv(out_taxs, header=False, mode='a')
            failed_accs += failed
    return failed_accs

def fetch(acc_list, out_dir, tmp_dir, warn_dir, chunk_size=500, max_attempts=3, workers=1, db_name='NCBI'):
    # perform two passes over the accession list, second pass attempts to retrieve the failed records
    # store list of records that fail to be retrieved after two passes
    
    # generate output file names
    out_seqs = f'{out_dir}/{db_name}.seqs'
    out_taxs = f'{tmp_dir}/{db_name}.taxs' # taxs file goes into the temporal directory because it is not the final taxonomy file
    warn_failed = f'{warn_dir}/NCBI_failed.lis'
    
    failed = []
    # first pass
    failed0 = retrieve_pass(acc_list, out_seqs, out_taxs, chunk_size, max_attempts, workers)
    if len(failed0) > 0:
        # second pass, skipped if there are no failed downloads
        failed = retrieve_pass(failed0, out_seqs, out_taxs, chunk_size. max_attempts, workers)
    nseqs = len(acc_list) - len(failed)
    # generate a save file containing the failed sequences (if any)
    if len(failed) > 0:
        with open(warn_failed, 'w') as w:
            w.write('\n'.join(failed))
    
    exclusion = list(set(acc_list).difference(set(failed)))
    return out_seqs, out_taxs, warn_failed, nseqs, exclusion

"""
Retrieve the designated records returned from the NCBI survey. The
accession list is split into smaller pieces that are retrieved
sequentially.
Skip and record every piece that fails to be downloaded a given amount of
times.
Succesfully retrieved records are stored to two files:
    seq_file: stores the retrieved sequences in fasta format
    tax_file: 2 column table containing accession codes : TaxIds

Parameters
----------
acc_list : list
    List of accession codes to be retrieved.
out_seqs : str
    Path to the file to store the retrieved sequences.
out_taxs : str
    Path to the file to store the retrieved taxonomy.
chunk_size : int, optional
    Size of the pieces the accession list will be split into. The default is 500.
max_attempts : int, optional
    Number of retrieval attempts to be made for each piece. The default is 3.

Returns
-------
failed : list
    List of accession codes that failed to be retrieved.

"""

"""
Retrieve records designated by the NCBI database survey. Perform two passes
and record accession codes that fail be retrieved.
Up to three files are generated:
    seq_file: stores the retrieved sequences in fasta format
    tax_file: 2 column table containing accession codes : TaxIds
    warning_file:   stores a list of accession codes that fail to be
                    retrieved after the two passes.

Parameters
----------
acc_list : list
    List of accession codes to be retrieved.
out_dir : str
    Path to the directory to contain the generated files.
tmp_dir : str, optional
    Path to the directory to contain the temporal files. If none is given, temporal files will be stored in the output directory.
warn_dir : str, optional
    Path to the directory to contain the warning file. If none is given, the warning file is stored in the output directory.
chunk_size : int, optional
    Size of the pieces the accession list will be split into. The default is 500.
max_attempts : int, optional
    Number of retrieval attempts to be made for each piece. The default is 3.

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
warn_failed : str
    Path to the generated warning file.
nseqs : int
    Number of records succesfully retrieved from the NCBI database.

"""

#%% Process taxonomy data
def build_lineage_table(tax_file, nodes_tab, ranks):
    """
    Rebuild the lineage for each taxon present in the records.
    Include lineages of upper taxa

    Parameters
    ----------
    tax_file : str
        Path to the taxonomy file generated by the NCBI fetch function.
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
    # open the generated taxonomy file
    taxid_tab = pd.read_csv(tax_file, index_col=0, header=None, names='accesion TaxId'.split()).TaxId
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

def build_taxonomy_table(tax_file, real_taxids):
    # build the dataframe containing the lowest valid taxId for each record
    taxid_tab = pd.read_csv(tax_file, index_col=0, header=None, names='accesion TaxId'.split()).TaxId
    taxonomy = real_taxids.loc[taxid_tab].astype(int)
    taxonomy.index = taxid_tab.index
    return taxonomy

def build_name_table(lineage_tab, names_tab):
    # build the dataframe containing the TaxId-Scientific name pairs
    names = names_tab.loc[np.unique(lineage_tab)]
    return names

def arrange_taxonomy(out_dir, tax_source, names_tab, nodes_tab, ranks, db_name='NCBI'):
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
    lineage_file=f'{out_dir}/{db_name}.lineage'
    taxonomy_file=f'{out_dir}/{db_name}.taxonomy'
    name_file=f'{out_dir}/{db_name}.names'
    
    # load names and nodes tables
    names_tab, nodes_tab = fetch_tools.read_taxdmp(names_tab, nodes_tab)
    
    # build taxonomy tables
    lineages, real_taxids = build_lineage_table(tax_source, nodes_tab, ranks)
    taxonomy_table = build_taxonomy_table(tax_source, real_taxids)
    name_table = build_name_table(lineages, names_tab)
    
    # save files
    lineages.to_csv(lineage_file)
    taxonomy_table.to_csv(taxonomy_file)
    name_table.to_csv(name_file)
    
    return lineage_file, taxonomy_file, name_file

#%% main retrieval function
"""
Retrieve sequence and taxonomy data from the NCBI database.

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
tmp_dir : str, optional
    Path to the directory to contain the temporal files. If none is given, temporal files will be stored in the output directory.
warn_dir : str, optional
    Path to the directory to contain the warning file. If none is given, the warning file is stored in the output directory.
chunk_size : TYPE, optional
    DESCRIPTION. The default is 500.
max_attempts : int, optional
    Number of download attempts to perform. The default is 3.
ranks : str
    Ranks to be included in the lineage table.

Returns
-------
out_seqs : str
    Path to the generated sequence file.
out_taxs : str
    Path to the generated taxonomy file.
warn_failed : str
    Path to the generated warning file.
lineage_file : str
    Path to the file containing the generated lineage table.
taxonomy_file : str
    Path to the file containing the accession-TaxId mapping.
name_file : str
    Path to the file containing the TaxId-Scientific name mapping.
nseqs : int
    Number of sequences successfuly retrieved from the NCBI database.

"""
def retrieve_data(taxon,
                  marker,
                  out_dir,
                  names_tab,
                  nodes_tab,
                  tmp_dir,
                  warn_dir,
                  chunk_size=500,
                  max_attempts=3,
                  ranks=['phylum', 'class', 'order', 'family', 'genus', 'species'],
                  workers=1,
                  db_name='NCBI'):
    # ensure directory names are properly formatted
    out_dir = re.sub('/$', '', out_dir)
    tmp_dir = re.sub('/$', '', tmp_dir)
    warn_dir = re.sub('/$', '', warn_dir)
    
    # retrieve data
    print(f'Searching NCBI for records matching the terms "{taxon}" and "{marker}"...')
    acc_list = survey(taxon, marker, max_attempts)
    if len(acc_list) == 0:
        raise Exception(f'No matching records found for terms "{taxon}" and "{marker}"')
    print(f'Found {len(acc_list)} records. Beginning retrieval operation...')
    out_seqs, out_taxs, warn_failed, nseqs, exclusion = fetch(acc_list, out_dir, tmp_dir, warn_dir, chunk_size=chunk_size, max_attempts=max_attempts, workers=workers, db_name=db_name)
    
    # generate taxonomy files
    lineage_file, taxonomy_file, name_file = arrange_taxonomy(out_dir, out_taxs, names_tab, nodes_tab, ranks)
    return out_seqs, out_taxs, warn_failed, lineage_file, taxonomy_file, name_file, nseqs, exclusion