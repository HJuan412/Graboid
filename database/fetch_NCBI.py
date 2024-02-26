#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:41:43 2024

@author: hernan

Perform search, retrieve sequence and taxonomy data from BOLD
"""

import http
import logging
import numpy as np
import os
import pandas as pd
import time
from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from database.fetch_tools import unfold_lineage

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
        
def fetch_in(acc_list, seq_file, tax_file, chunk_size=500, max_attempts=3):
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
    
    # chunks that can't be downloaded are returned in the failed list
    failed = []
    
    # split acc_list
    chunks = acc_slicer(acc_list, chunk_size)
    n_chunks = int(np.ceil(len(acc_list)/chunk_size))
    
    print(f'Downloading {len(acc_list)} records from NCBI...')
    for chunk_n, chunk in enumerate(chunks):
        print(f'Downloading chunk {chunk_n + 1} of {n_chunks}')
        seq_recs = []
        for attempt in range(max_attempts):
            # try to retrieve the sequence records up to max_attempt times per chunk
            try:
                # contact NCBI and attempt download
                logger.debug(f'Starting chunk download (size: {len(chunk)})')
                seq_handle = Entrez.efetch(db = 'nucleotide', id = chunk, rettype = 'fasta', retmode = 'xml')
                seq_recs = Entrez.read(seq_handle)
                logger.debug(f'Done retrieving {len(seq_recs)} records')
                break
            except IOError:
                print(f'Chunk {chunk_n} download interrupted. {max_attempts - attempt} attempts remaining')
                continue
            except http.client.IncompleteRead:
                print(f'Chunk {chunk_n} download interrupted. {max_attempts - attempt} attempts remaining')
                continue
            except KeyboardInterrupt:
                print('Manually aborted')
                return
            time.sleep(3)
        if len(seq_recs) != len(chunk):
            failed += chunk
            continue
        
        # generate seq records and retrieve taxids
        seqs = []
        taxids = []
        for acc, seq in zip(chunk, seq_recs):
            seqs.append(SeqRecord(id=acc, seq = Seq(seq['TSeq_sequence']), description = ''))
            taxids.append(seq['TSeq_taxid'])
        
        # save seq recods to fasta
        with open(seq_file, 'a') as out_handle0:
            SeqIO.write(seqs, out_handle0, 'fasta')
        # save tax table to csv
        taxs = pd.DataFrame({'Accession':chunk, 'TaxId':taxids})
        taxs.to_csv(tax_file, header = chunk_n == 0, index = False, mode='a')
    return failed

def fetch(acc_list, out_dir, warn_dir=None, chunk_size=500, max_attempts=3):
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

    """
    
    # prepare output directory & files
    os.makedirs(out_dir, exist_ok=True)
    out_seqs = f'{out_dir}/NCBI.seqs'
    out_taxs = f'{out_dir}/NCBI.tax'
    if warn_dir is None:
        warn_dir = out_dir
    else:
        os.makedirs(warn_dir, exist_ok=True)
    warn_failed = f'{warn_dir}/NCBI_failed.lis'
    
    # Make two download passes, only give up when a sequence chunk fails to be retrieved both times
    failed = [] # here we store accessions that failed to download
    # first pass
    failed0 = fetch_in(acc_list, out_seqs, out_taxs, chunk_size, max_attempts)
    # do a second pass if any sequence couldn't be downloaded
    if len(failed0) > 0:
        logger.warning(f'NCBI first pass. Failed to retrieve {len(failed0)} of {len(acc_list)} sequence records.')
        time.sleep(10)
        failed = fetch_in(failed0, out_seqs, out_taxs, chunk_size, max_attempts)
    if len(failed) > 0:
        logger.warning(f'NCBI second pass. Failed to retrieve {len(failed)} of {len(failed0)} sequence records.')
    
    # no records could be retrieved
    if len(failed) == len(acc_list):
        raise Exception('Failed to retrieve sequences from NCBI database.')
    
    # report results
    logger.info(f'Retrieved {len(acc_list) - len(failed)} of {len(acc_list)} sequence records from the NCBI database.')
    
    # generate a save file containing the failed sequences (if any)
    if len(failed) > 0:
        with open(warn_failed, 'w') as w:
            w.write('\n'.join(failed))
        logger.info(f'{len(failed)} failed accession codes saved to {warn_failed}')
    else:
        warn_failed = None
    
    return out_seqs, out_taxs, warn_failed

#%% Process taxonomy data
def build_lineage_table(tax_file, nodes_tab, *ranks):
    """
    Rebuild the lineage for each taxon present in the records.
    Include lineages of upper taxa

    Parameters
    ----------
    tax_file : str
        Path to the taxonomy file generated by the NCBI fetch function.
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
    # open the generated taxonomy file
    taxid_tab = pd.read_csv(tax_file, index_col=0).TaxId
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

def build_taxonomy_table(tax_file, real_taxids):
    # build the dataframe containing the lowest valid taxId for each record
    taxid_tab = pd.read_csv(tax_file, index_col=0).TaxId
    taxonomy = real_taxids.loc[taxid_tab].astype(int)
    taxonomy.index = taxid_tab.index
    return taxonomy

def build_name_table(lineage_tab, names_tab):
    # build the dataframe containing the TaxId-Scientific name pairs
    names = names_tab.loc[np.unique(lineage_tab), 'TaxId']
    return names

def arrange_taxonomy(out_dir, tax_source, names_tab, nodes_tab, *ranks):
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
    *ranks : str
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
    lineage_file=f'{out_dir}/NCBI.lineage'
    taxonomy_file=f'{out_dir}/NCBI.taxonomy'
    name_file=f'{out_dir}/NCBI.names'
    
    # build taxonomy tables
    lineages, real_taxids = build_lineage_table(tax_source, nodes_tab, *ranks)
    taxonomy_table = build_taxonomy_table(tax_source, real_taxids)
    name_table = build_name_table(lineages, names_tab)
    
    # save files
    lineages.to_csv(lineage_file)
    taxonomy_table.to_csv(taxonomy_file)
    name_table.to_csv(name_file)
    
    return lineage_file, taxonomy_file, name_file

#%% main retrieval function
def retrieve_data(taxon, marker, out_dir, names_tab, nodes_tab, warn_dir=None, chunk_size=500, max_attempts=3, *ranks):
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
    warn_dir : str, optional
        Path to the directory to contain the warning file. If none is given, the warning file is stored in the output directory.
    chunk_size : TYPE, optional
        DESCRIPTION. The default is 500.
    max_attempts : int, optional
        Number of download attempts to perform. The default is 3.
    *ranks : str
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

    """
    # retrieve data
    acc_list = survey(taxon, marker, max_attempts)
    out_seqs, out_taxs, warn_failed = fetch(acc_list, out_dir, warn_dir, chunk_size, max_attempts)
    
    # generate taxonomy files
    lineage_file, taxonomy_file, name_file = arrange_taxonomy(out_dir, out_taxs, names_tab, nodes_tab, *ranks)
    return out_seqs, out_taxs, warn_failed, lineage_file, taxonomy_file, name_file