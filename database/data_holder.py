#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:25:41 2024

@author: hernan

This script contains the data holder class used to load graboid databases and query files
"""

#%% libraries
import json
import numpy as np
import os
import pandas as pd

# graboid libraries
from mapping import mapping as mpp
from preprocess import sequence_collapse, consensus_taxonomy
#%% functions
def load_map(map_file, acc_file):
    # map_file: __map.npz file
    # acc_file: __map.acc file
    
    # load a map file and the corresponding accession file
    # from npz file, extract: alignment map, bounds array, coverage array
    # calculate normalized coverage
    # TODO: could incorporate accession list to npz file?
    map_ = np.load(map_file)
    matrix = map_['matrix']
    bounds = map_['bounds']
    coverage = map_['coverage']
    coverage_norm = coverage / coverage.max()
    # retrieve accession list
    with open(acc_file, 'r') as acc_:
        accs = acc_.read().splitlines()
    
    return matrix, accs, bounds, coverage, coverage_norm

#%% classes
class Data:
    """
    Parent class for the reference and query data classes, includes methods
    for:
        site filtering by coverage
        region selection
        effective sequence collapsing
    """        
    @property
    def shape(self):
        if hasattr(self, 'map'):
            return self.map.shape
        return None
    
    def filter_sites(self, min_coverage=.0):
        """
        Select sites in the alignment according to the minimum coverage threshold

        Parameters
        ----------
        min_coverage : float, optional
            Minimum coverage threshold, expressed as a fraction from 0 to 1.
            The default is .0.

        Returns
        -------
        None.

        """
        self.min_coverage = max(.0, min(min_coverage, 1))
        self.filtered = self.coverage_norm >= min_coverage
        
        
        # record that filter has changed and region selection & collapsing are outdated
        self.to_date_region = False
        self.to_date_collapsed = False
    
    def select_region(self, start=None, end=None, indexes=None):
        """
        Select a specific region or a set of positions of the alignment map, retaining the previously
        filtered sites

        Parameters
        ----------
        start : int, optional
            Starting position for the selected region. The default is None.
        end : int, optional
            End position for the selected region. The default is None.
        indexes : numpy.array, optional
            Array containing a specific set of indexes to select. The default is None.

        Returns
        -------
        None.

        """
        
        self.region = np.full(self.map.shape[1], False)
        if not indexes is None:
            self.region[indexes] = True
        else:
            self.region[start:end] = True
        self.region = self.region & self.filtered
        
        # get indexes of selected sites
        self.site_indexes = np.arange(self.shape[1])[self.region]
        self.start = self.site_indexes.min()
        self.end = self.site_indexes.max()
        
        # record that region selection has changed and that collapsing is outdated
        self.to_date_region = True
        self.to_date_collapsed = False
    
    def collapse(self, max_unk_thresh=.2):
        """
        Collapse the selected alignment region. Apply a different postprocessing
        method depending on the child class.

        Parameters
        ----------
        max_unk_thresh : float, optional
            Maximum unknown sites allowed for a sequence to be included.
            The default is .2.

        Returns
        -------
        None.

        """
        # attribute valid indicates selected sequences:
            # For the reference dataset these are the sequences with known classification at the required rank
            # For the query dataset, all sequences are valid (attribute is kept for compatibility)
        filtered_map = self.map[self.valid][:, self.region]
        
        # only collapse if region selection is up to date and collapsing is outdated
        if self.to_date_region:
            
            if not self.to_date_collapsed:
                self.collapsed, self.branches = sequence_collapse.sequence_collapse(filtered_map, max_unk_thresh)
                
                # postprocess (different method for R and Q)
                self.collapse_postprocess()
                
                # record that collapsing is up to date
                self.to_date_collapsed = True
            else:
                # include log announcing that no collapsing was performed
                print('Collapsing up to date, skipping')
                pass
        else:
            raise Exception('Region selection is not up to date, call select_region method and try again')
    
class R(Data):
    """
    Child of Data, meant to hold the reference dataset.
    """
    def load(self, ref_dir, min_coverage=.0, required_rank='family'):
        """
        Load reference database. Apply minimum coverage filter and select sequences
        by required rank classification

        Parameters
        ----------
        ref_dir : str
            Path to the directory containing the database components.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .0.
        required_rank : str, optional
            Rank at which classification is required. The default is 'family'.

        Raises
        ------
        Exception
            If any database component is missing, the loading process is interrupted.

        Returns
        -------
        None.

        """

        try:
            with open(f'{ref_dir}/meta.json', 'r') as handle:
                meta = json.load(handle)
        except FileNotFoundError:
            raise Exception('Meta file not found, verify that the given reference directory is a graboid database.')

        # load map files
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(meta['map_mat_file'], meta['map_acc_file'])
        
        # load taxonomy data
        ref_tax = pd.read_csv(meta['tax_file'], names=['Accession', 'TaxId'], skiprows=[0])
        self.y = ref_tax.set_index('Accession').loc[self.accs, 'TaxId'].to_numpy()
        self.lineage_tab = pd.read_csv(meta['lineages_file'], index_col=0)
        self.names_tab = pd.read_csv(meta['names_file'], index_col=0)['SciName']
        
        self.lineage = self.lineage_tab.loc[self.y] # subsection of lineage_tab corresponding to the reference instances
        
        # load guide blast reference
        self.blast_db = meta['guide_db']
        
        # apply coverage threshold
        self.filter_sites(min_coverage)
        
        # filter by known taxon at the required rank
        self.get_valid_sequences(required_rank)
        
    def get_valid_sequences(self, required_rank='family'):
        """
        Select sequences with known taxonomic classification at the required rank

        Parameters
        ----------
        required_rank : str, optional
            Rank at which classification is required. The default is 'family'.

        Returns
        -------
        None.

        """
        self.required_rank = required_rank
        self.valid = self.lineage[required_rank].to_numpy() != 0
        
        self.to_date_region = False
        self.to_date_collapsed = False
    
    def collapse_postprocess(self):
        """
        Reference dataset specific postprocessing. Get consensus taxa for the
        collapsed effective sequences

        Returns
        -------
        None.

        """
        # collapse reference taxonomy data
        filtered_accs = self.y[self.valid]
        self.y_collapsed = consensus_taxonomy.collapse_taxonomies(self.branches, filtered_accs, self.lineage_tab)
        self.lineage_collapsed = self.lineage_tab.loc[self.y_collapsed].reset_index(drop=True)

class Q(Data):
    """
    Child of Data, meant to hold the query dataset.
    """
    @property
    def valid(self):
        return np.full(self.map.shape[0], True)
    
    def load(self,
             qry_file,
             qry_dir,
             blast_db,
             evalue=0.0005,
             dropoff=0.05,
             min_height=0.1,
             min_width=2,
             threads=1,
             qry_name='QUERY',
             min_coverage=.95):
        """
        Load query dataset, build alignment, and apply minimum coverage filter.

        Parameters
        ----------
        qry_file : str
            Path to the query sequence file (expected to be in Fasta format).
        qry_dir : str
            Path to the output files generated during alignment building.
        blast_db : str
            Path to the reference dataset BLAST guide.
        evalue : float, optional
            Maximum e-value threshold for the aligned sequences. The default is 0.0005.
        dropoff : float, optional
            Maximum dropoff required for mesa designation. The default is 0.05.
        min_height : float, optional
            Minimum height (coverage) required for mesa designation. The default is 0.1.
        min_width : int, optional
            Minimum width meant for mesa designation. The default is 2.
        threads : int, optional
            Number of processors to be used during map construction. The default is 1.
        qry_name : str, optional
            Prefix for the generated files. The default is 'QUERY'.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .95.

        Raises
        ------
        Exception
            Query file is expected to be a Fasta file, if another format is
            presented the loading process is interrupted.

        Returns
        -------
        None.

        """
        map_prefix = f'{qry_dir}/{qry_name}'
        os.makedirs(qry_dir, exist_ok=True)
        if mpp.check_fasta(qry_file) == 0:
            raise Exception(f'Error: Query file {qry_file} is not a valid fasta file')
        
        qry_map_file, qry_acc_file, nrows, ncols = mpp.build_map(qry_file, blast_db, map_prefix, threads=threads, clip=False)
        
        # load map files
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(qry_map_file, qry_acc_file)
        
        # apply coverage threshold
        self.filter_sites(min_coverage)
    
    def load_quick(self, qry_map_file, qry_acc_file, min_coverage=.95):
        """
        Load preexisting query alignment map.

        Parameters
        ----------
        qry_map_file : str
            Path to the query map file.
        qry_acc_file : str
            Path to the query accessions file.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .95.

        Returns
        -------
        None.

        """
        # shorter version of load query, load pre-generated query map files
        # load query dataset
        self.map, self.accs, self.bounds, self.coverage, self.coverage_norm = load_map(qry_map_file, qry_acc_file)
        
        # apply coverage threshold
        self.filter_sites(min_coverage)
    
    def collapse_postprocess(self):
        """
        Generate a map indicating the effective sequence branch to which each
        query sequence belongs.

        Returns
        -------
        None.

        """
        # build expanded query map, dataframe mapping each query sequence to its branch
        branch_map = pd.DataFrame(-1, index=pd.Index(self.accs, name='Query'), columns=['Branch'])
        for br_idx, branch in enumerate(self.branches):
            branch_map.iloc[branch, 0] = br_idx
        self.branch_map = branch_map

class DataHolder:
    """
    This class contains both reference and query datasets. Handles data loading,
    filtering, selection and collapsing.
    Reference dataset attributes used in analysis:
        R.collapsed : Collapsed sub-alignment of valid sequences and filtered, selected sites
        R.y_collapsed : Consensus taxa for the collapsed effective sequences
        R.lineage_collapsed : Extended lineage for the collapsed effective sequences
    Query dataset attributes used in analysis:
        Q.collapsed : Collapsed sub-alignment of valid sequences and filtered, selected sites
        Q.branch_map : map identifying member sequences of each collapsed branch
    """
    @property
    def shape(self):
        if hasattr(self, 'R'):
            return self.R.shape
        return None
    
    def load_reference(self, ref_dir, min_coverage=.0, required_rank='family'):
        """
        Load reference database. Apply minimum coverage filter and select sequences
        by required rank classification

        Parameters
        ----------
        ref_dir : str
            Path to the directory containing the database components.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .0.
        required_rank : str, optional
            Rank at which classification is required. The default is 'family'.
        
        Returns
        -------
        None.

        """
        self.R = R()
        self.R.load(ref_dir, min_coverage, required_rank)
    
    def load_query(self,
                   qry_file,
                   qry_dir='.',
                   evalue=0.0005,
                   dropoff=0.05,
                   min_height=0.1,
                   min_width=2,
                   threads=1,
                   qry_name='QUERY',
                   min_coverage=.95):
        """
        Load query dataset, build alignment, and apply minimum coverage filter.

        Parameters
        ----------
        qry_file : str
            Path to the query sequence file (expected to be in Fasta format).
        qry_dir : str
            Path to the output files generated during alignment building.
        blast_db : str
            Path to the reference dataset BLAST guide.
        evalue : float, optional
            Maximum e-value threshold for the aligned sequences. The default is 0.0005.
        dropoff : float, optional
            Maximum dropoff required for mesa designation. The default is 0.05.
        min_height : float, optional
            Minimum height (coverage) required for mesa designation. The default is 0.1.
        min_width : int, optional
            Minimum width meant for mesa designation. The default is 2.
        threads : int, optional
            Number of processors to be used during map construction. The default is 1.
        qry_name : str, optional
            Prefix for the generated files. The default is 'QUERY'.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .95.

        Returns
        -------
        None.

        """
        
        self.Q = Q()
        self.Q.load(qry_file,
                    qry_dir,
                    self.R.blast_db,
                    evalue=evalue,
                    dropoff=dropoff,
                    min_height=min_height,
                    min_width=min_width,
                    threads=threads,
                    qry_name=qry_name,
                    min_coverage=min_coverage)
        
    def load_query_quick(self, qry_map_file, qry_acc_file, min_coverage=.95):
        """
        Load preexisting query alignment map.

        Parameters
        ----------
        qry_map_file : str
            Path to the query map file.
        qry_acc_file : str
            Path to the query accessions file.
        min_coverage : float, optional
            Minimum coverage threshold to apply during site filtering.
            The default is .95.

        Returns
        -------
        None.

        """
        self.Q = Q()
        self.Q.load_quick(qry_map_file, qry_acc_file, min_coverage)
    
    def set_required_rank(self, required_rank='family'):
        """
        Reset the reference dataset's required rank filter

        Parameters
        ----------
        required_rank : str, optional
            Rank at which classification is required. The default is 'family'.

        Returns
        -------
        None.

        """
        self.R.get_valid_sequences(required_rank)
    
    def set_filters(self, min_cov_R=None, min_cov_Q=None):
        """
        Reset minimum coverage filters for reference or query datasets. Only perform
        the change if a new value is given.

        Parameters
        ----------
        min_cov_R : float, optional
            New minimum coverage threshold for the REFERENCE dataset. The default is None.
        min_cov_Q : float, optional
            New minimum coverage threshold for the REFERENCE dataset. The default is None.

        Returns
        -------
        None.

        """
        if not min_cov_R is None:
            self.R.filter_sites(min_cov_R)
        if not min_cov_Q is None:
            self.Q.filter_sites(min_cov_Q)
    
    def get_overlap(self, min_cov_R=None, min_cov_Q=None, start=0, end=-1, collapse=True, max_unk_thresh=.2):
        """
        Select overlapping region between the reference and query datasets.
        Optionally reapply coverage thresholds or bound the selected region.

        Parameters
        ----------
        min_cov_R : float, optional
            New minimum coverage threshold for the REFERENCE dataset. The default is None.
        min_cov_Q : float, optional
            New minimum coverage threshold for the REFERENCE dataset. The default is None.
        start : TYPE, optional
            DESCRIPTION. The default is 0.
        end : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """
        self.set_filters(min_cov_R, min_cov_Q)
        overlap = np.arange(self.shape[1])[self.R.filtered & self.Q.filtered]
        end = self.shape[1] if end < 1 else end
        overlap = overlap[(overlap >= start) & (overlap <= end + 1)]

        self.R.select_region(indexes=overlap)
        self.Q.select_region(indexes=overlap)
        if collapse:
            self.collapse(max_unk_thresh)
    
    def select_region(self, start=0, end=-1, max_unk_thresh=.2):
        """
        Select and collapse a specific region of the reference alignment.

        Parameters
        ----------
        start : int, optional
            Starting position of the selected region. The default is 0.
        end : int, optional
            Ending position of the selected region. The default is -1.
        max_unk_thresh : float, optional
            Maximum unknown threshold for sequence collapsing. The default is .2.

        Returns
        -------
        None.

        """
        end = self.shape[1] if end < 1 else end
        
        self.R.select_region(start, end)
        self.R.collapse(max_unk_thresh)
        
    def collapse(self, max_unk_thresh=.2):
        """
        Collapse filtered and selected sequences of reference and query datasets.

        Parameters
        ----------
        max_unk_thresh : float, optional
            Maximum unknown threshold for sequence collapsing. The default is .2.

        Returns
        -------
        None.

        """
        self.R.collapse(max_unk_thresh)
        if hasattr(self, 'Q'):
            self.Q.collapse()
