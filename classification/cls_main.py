#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:25:00 2023

@author: hernan

Classifier class, handles steps: database loading, query blasting, custom calibration, parameter selection, classification and reporting
"""

#%% libraries
from datetime import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import shutil
# Graboid libraries
from calibration import cal_main
from calibration import cal_metrics
from classification import cost_matrix
from classification import cls_classify
from classification import cls_distance
from classification import cls_neighbours
from classification import cls_parameters
from classification import cls_preprocess
from classification import cls_report

from DATA import DATA
from mapping import director as mp

#%% set logger
logger = logging.getLogger('Graboid.Classification')
logger.setLevel(logging.DEBUG)

#%% functions
def map_query(out_dir,
              warn_dir,
              fasta_file,
              db_dir,
              evalue=0.005,
              dropoff=0.05,
              min_height=0.1,
              min_width=2,
              threads=1):
    """
    Map the query file against the database reference sequence and define
    mesas.

    Parameters
    ----------
    out_dir : str
        Output directory.
    warn_dir : str
        Warnings directory.
    fasta_file : str
        Query sequence file.
    db_dir : str
        Graboid database director.
    evalue : float, optional
        Evalue threshold for the BLAST alignment. The default is 0.005.
    dropoff : float, optional
        Coverage dropoff threshold used to detect the edge of a mesa. The default is 0.05.
    min_height : float, optional
        Minimum fraction of the max coverage for a region of the alignment to be included in a mesa. The default is 0.1.
    min_width : int, optional
        Minimum amount of sites for a region of the alignment to be included in a mesa. The default is 2.
    threads : int, optional
        Number of processors to use. The default is 1.

    Returns
    -------
    map_file : str
        Path to the resulting map file.
    acc_file : str
        Path to the file of accepted accession codes.
    blast_report : str
        Path to the generated blast report.
    acc_list : list
        list of accepted accesion codes.
    bounds : tuple
        Edge coordinates of the query alignment over the reference sequence.
    matrix : numpy.array
        Generated alignment in the form of a 2d numpy array.
    coverage : numpy.array
        Coverage of the generated alignment.
    mesas : numpy.array
        Information about the generated coverage mesas.
        
    """
    
    map_director = mp.Director(out_dir, warn_dir, logger)
    map_director.direct(fasta_file = fasta_file,
                        db_dir = db_dir,
                        evalue = evalue,
                        dropoff = dropoff,
                        min_height = min_height,
                        min_width = min_width,
                        threads = threads,
                        keep = True)
    map_file = os.path.abspath(map_director.mat_file)
    acc_file = os.path.abspath(map_director.acc_file)
    blast_report = os.path.abspath(map_director.blast_report)
    acc_list = map_director.accs
    bounds = map_director.bounds
    matrix = map_director.matrix
    coverage = map_director.coverage
    mesas = map_director.mesas
    return map_file, acc_file, blast_report, acc_list, bounds, matrix, coverage, mesas

def get_mesas_overlap(ref_mesas, qry_mesas, min_width=10):
    """
    Locate the overlapping regions between the query and reference alignments

    Parameters
    ----------
    ref_mesas : numpy.array
        Information matrix for the reference mesas.
    qry_mesas : numpy.array
        Information matrix for the query mesas.
    min_width : int, optional
        Minimum overlap threshold. The default is 10.

    Returns
    -------
    numpy.array
        2d Array with columns: overlap_start, overlap_end, overlap_width, ref_coverage, qry_coverage.
        Each row represent a distinct overlapping region

    """

    mesas_overlap = []
    for q_mesa in qry_mesas:
        # get reference mesas that overlap with the current query mesa: (ref mesa start < qry mesa end) & (ref mesa end > qry mesa start)
        overlap_idxs = (ref_mesas[:,0] <= q_mesa[1]) & (ref_mesas[:,1] >= q_mesa[0])
        overlapping_mesas = np.clip(ref_mesas[overlap_idxs, :2], q_mesa[0], q_mesa[1])
        overlapping_widths = overlapping_mesas[:,1] - overlapping_mesas[:,0]
        # build overlapping matrix for current q_mesa
        q_overlap = np.zeros((len(overlapping_mesas), 5))
        q_overlap[:, :2] = overlapping_mesas
        q_overlap[:, 2] = overlapping_widths
        q_overlap[:, 3] = ref_mesas[overlap_idxs, 3]
        q_overlap[:, 4] = q_mesa[3]
        
        mesas_overlap.append(q_overlap[q_overlap[:,2] >= min_width]) # append only overlaps over the specified minimum width
    return np.concatenate(mesas_overlap, 0)

## parameter selection
def get_params_ce(report, ranks):
    """
    Select the best parameter combination for each taxonomic rank using the
    cross entropy metric.
    Cross entropy ranges from 0 to 10. Lower values are better.

    Parameters
    ----------
    report : pandas.DataFrame
        Cross Entropy report. Columns: window, w_start, w_end, n, k, method, rank0, rank1, ...
    ranks : list
        List of ranks. Must match the ranks present in the report columns

    Yields
    ------
    best_params : pandas.DataFrame
        Best parameters report. Columns: rank, window, w_start, w_end, n, k, method, cross_entropy
    []
        Empty list, kept for compatibility with get_params_met.

    """ 
    
    selected_params = []
    for rk in ranks:
        # get the best (minimum) score for rk, retrieve parameter combinations that yield it
        min_ce = report[rk].min()
        params_subtab = report.loc[report[rk] == min_ce, ['window', 'w_start', 'w_end', 'n', 'k', 'method', rk]].copy()
        params_subtab.rename(columns={rk:'cross_entropy'}, inplace=True)
        params_subtab['rank'] = rk
        selected_params.append(params_subtab)
    
    selected_params = pd.concat(selected_params).reset_index(drop=True)
    
    # filter params
    # the basal rank will usually score 0 loss for all param combinations, select only combinations that yield good scores in lower ranks
    score0_tab = selected_params.loc[selected_params.cross_entropy == 0].reset_index().set_index(['window', 'n', 'k', 'method']) # all combinations with 0 entropy
    next_best_tab = selected_params.loc[selected_params.cross_entropy > 0] # all parameter combinations with cross entropy greater than 0
    
    filtered_idxs = []
    for params, params_subtab in next_best_tab.groupby(['window', 'n', 'k', 'method']):
        try:
            filtered_idxs.append(score0_tab.loc[params, 'index'])
        except KeyError:
            continue
        
    best_params = pd.concat((selected_params.loc[filtered_idxs], next_best_tab))[['rank', 'window', 'w_start', 'w_end', 'n', 'k', 'method', 'cross_entropy']] # reorganize columns
    return best_params, [] # empty list used for compatibility with get_params_met

def get_params_met(taxa, report):
    """
    Select the best parameter combinations for each taxon in taxa using the
    given metric report.
    Scores range from 0 to 1. Higher values are better.

    Parameters
    ----------
    taxa : list
        List of taxa to search for.
    report : pandas.DataFrame
        Metric report. Columns: rank, taxon, taxID, window, w_start, w_end, n, k, method, score

    Returns
    -------
    best_params : pandas.DataFrame
        Best parameters report. Columns: taxon, window, w_start, w_end, n, k, method, score
    warnings : list
        List of generated warnings.

    """
    
    best_params = []
    warnings = []
    
    for tax in taxa:
        # locate occurrences of tax in the report. Generate a warning if tax is absent or its best score is 0
        tax_subtab = report.loc[report.taxon == tax]
        if tax_subtab.shape[0] == 0:
            warnings.append(f'{tax} not found in the given report')
            continue
        best_score = tax_subtab.score.max()
        if best_score == 0:
            warnings.append(f'{tax} had a null score. Cannot be detected in the current window.')
            continue
        best_params.append(tax_subtab.loc[tax_subtab.score == best_score, ['taxon', 'window', 'w_start', 'w_end', 'n', 'k', 'method', 'score']])
    
    best_params = pd.concat(best_params)
    return best_params, warnings

def report_params(params, warnings, report_file, metric, *taxa):
    # build parameter report
    met_names = {'acc' : 'accuracy',
                 'prc' : 'precision',
                 'rec' : 'recall',
                 'f1' : 'F1 score',
                 'ce' : 'Cross entropy'}
    metric = met_names[metric]
    header = f'Best parameter combinations determined by {metric}'
    if len(taxa) > 0:
        header += '\nAnalyzed taxa: ' + ', '.join(taxa)
    header += '\n\n'
    
    params = params.rename(columns = {'score':metric}).set_index(params.columns[0])
    
    with open(report_file, 'a') as handle:
        handle.write(header)
        handle.write(repr(params))
        handle.write('\n\n')
        if len(warnings) > 0:
            handle.write('Warnings:\n')
            for warn in warnings.values():
                handle.write(warn + '\n')
            handle.write('\n')
        handle.write('#' * 40 + '\n\n')
    return

def collapse_params(params):
    """
    Select unique parameter combinations

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame generated using either get_params_ce or get_params_met.

    Returns
    -------
    collapsed_params : dict
        Dictionary of key:values -> (window start, window end, n, k, m):[taxa/ranks].

    """
    
    params = params.rename(columns={'taxon':'name', 'rank':'name'})
    collapsed_params = {}
    for (w, n, k, m), param_subtab in params.groupby(['window', 'n', 'k', 'method']):
        ws = param_subtab.w_start.values[0]
        we = param_subtab.w_end.values[0]
        collapsed_params[(ws, we, n, k, m)] = param_subtab.name.values.tolist()
    return collapsed_params

#%% classes
class ClassifierBase:
    def __init__(self):
        self.__db = None
        self.__last_calibration = None
        self.__query_file = None
        self.__query_map_file = None
        self.__query_acc_file = None
        self.__transition = None
        self.__transversion = None
        self.__auto_start = None
        self.__auto_end = None
        self.__auto_n = None
        self.__auto_k = None
        self.__auto_mth = None
        
        # self.out_dir = None
        # self.calibration_dir = None
        # self.classif_dir = None
        # self.query_dir = None
        # self.warn_dir = None

        # self.cost_matrix = None

        # self.db_dir = None
        # self.db_reffile = None
        # self.db_refpath = None
        # self.db_refdir = None

        # self.guide = None
        # self.tax_ext = None
        # self.ranks = None

        # self.ref_matrix = None
        # self.ref_mesas = None
        # self.ref_coverage = None
        # self.max_pos = None

        # self.ref_accs = None
        # self.tax_tab = None

        # self.query_blast_report = None
        # self.query_accs = None
        # self.query_bounds = None
        # self.query_matrix = None
        # self.query_coverage = None
        # self.query_mesas = None

        # self.collapsed_params = None
    
    @property
    def meta(self):
        return {'db':self.db,
                'query_file':self.query_file,
                'query_map_file':self.query_map_file,
                'query_acc_file':self.query_acc_file,
                'last_calibration':self.last_calibration,
                'transition':self.transition,
                'transversion':self.transversion,
                'auto_start':self.auto_start,
                'auto_end':self.auto_end,
                'auto_n':self.auto_n,
                'auto_k':self.auto_k,
                'auto_mth':self.auto_mth}
    
    @property
    def db(self):
        return self.__db
    @db.setter
    def db(self, db):
        self.__db = db
        self.update_meta()
    
    @property
    def query_file(self):
        return self.__query_file
    @query_file.setter
    def query_file(self, query_file):
        self.__query_file = query_file
        self.update_meta()
    
    @property
    def query_map_file(self):
        return self.__query_map_file
    @query_map_file.setter
    def query_map_file(self, query_map_file):
        self.__query_map_file = query_map_file
        self.update_meta()
    
    @property
    def query_acc_file(self):
        return self.__query_acc_file
    @query_acc_file.setter
    def query_acc_file(self, query_acc_file):
        self.__query_acc_file = query_acc_file
        self.update_meta()
    
    @property
    def last_calibration(self):
        return self.__last_calibration
    @last_calibration.setter
    def last_calibration(self, last_calibration):
        self.__last_calibration = last_calibration
        self.update_meta()
    
    @property
    def transition(self):
        return self.__transition
    @transition.setter
    def transition(self, transition):
        self.__transition = transition
        self.update_meta()
    
    @property
    def transversion(self):
        return self.__transversion
    @transversion.setter
    def transversion(self, transversion):
        self.__transversion = transversion
        self.update_meta()
    
    # automatic parameters
    @property
    def auto_start(self):
        return self.__auto_start
    @auto_start.setter
    def auto_start(self, start):
        self.__auto_start = start
        self.update_meta()
    @property
    def auto_end(self):
        return self.__auto_end
    @auto_end.setter
    def auto_end(self, end):
        self.__auto_end = end
        self.update_meta()
    @property
    def auto_n(self):
        return self.__auto_n
    @auto_n.setter
    def auto_n(self, n):
        self.__auto_n = n
        self.update_meta()
    @property
    def auto_k(self):
        return self.__auto_k
    @auto_k.setter
    def auto_k(self, k):
        self.__auto_k = k
        self.update_meta()
    @property
    def auto_mth(self):
        return self.__auto_mth
    @auto_mth.setter
    def auto_mth(self, mth):
        self.__auto_mth = mth
        self.update_meta()
    
    def update_meta(self):
        with open(self.out_dir + '/meta.json', 'w') as handle:
            json.dump(self.meta, handle)
        
class Classifier(ClassifierBase):    
    def set_outdir(self, out_dir, overwrite=False):
        self.out_dir = out_dir
        self.calibration_dir = out_dir + '/calibration'
        self.classif_dir = out_dir + '/classification'
        self.query_dir = out_dir + '/query'
        self.warn_dir = out_dir + '/warnings'
        
        if os.path.isdir(out_dir):
            # out dir already exists
            if overwrite:
                shutil.rmtree(out_dir)
            else:
                try:
                    with open(out_dir + '/meta.json', 'r') as handle:
                        meta = json.load(handle)
                        self.set_database(meta['db'])
                        self.set_cost_matrix(meta['transition'], meta['transversion'])
                        self.last_calibration = meta['last_calibration']
                        self.query_file = meta['query_file']
                        self.query_map_file = meta['query_map_file']
                        self.query_acc_file = meta['query_acc_file']
                        self.load_query()
                except FileNotFoundError:
                    raise Exception('Specified output directory exists but cannot be verified as a Graboid classification directory. Recommend overwrtiting it or using a different name')
                    # TODO: maybe include the option of verifying if it is a classif dir with a damaged/mising meta file
        # create directories (if absent or set to overwrite)
        os.makedirs(self.calibration_dir, exist_ok=True)
        os.makedirs(self.classif_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.warn_dir, exist_ok=True)
        
        warn_handler = logging.FileHandler(self.warn_dir + '/warnings.log')
        warn_handler.setLevel(logging.WARNING)
        logger.addHandler(warn_handler)
    
    def set_cost_matrix(self, transition, transversion):
        self.transition = transition
        self.transversion = transversion
        self.cost_matrix = cost_matrix.cost_matrix(transition, transversion)
    
    def set_database(self, database=None):
        # verify that given database is valid and there isn't another database already set
        if database is None:
            # no database provided, do nothing
            return
        if not self.db is None and self.db != database:
            raise Exception(f'Working directory {self.out_dir} is set to use the graboid database {self.db}. Conflict with proposed database {database}. To use a different database, select a different working directory or overwrite the current one (This will delete all Calibration and Classification data in the current directory).')
        try:
            self.db_dir = DATA.get_database(database)
        except Exception as excp:
            raise excp
        
        # log database
        self.db = database
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        
        # get database reference sequence
        self.db_reffile = db_meta['reference']
        self.db_refpath = db_meta['ref_file']
        self.db_refdir = re.sub('ref.fasta', '', self.db_refpath) # get the directory containing the blast database files
        # load taxonomy guides
        self.guide = pd.read_csv(db_meta['guide_file'], index_col=0)
        self.tax_ext = pd.read_csv(db_meta['expguide_file'], index_col=0)
        self.ranks = self.tax_ext.columns.tolist()
        
        # load matrix & accession codes
        map_npz = np.load(db_meta['mat_file'])
        self.ref_matrix = map_npz['matrix']
        self.ref_mesas = map_npz['mesas']
        self.ref_coverage = map_npz['coverage']
        self.max_pos = self.ref_matrix.shape[1]
        with open(db_meta['acc_file'], 'r') as handle:
            self.ref_accs = handle.read().splitlines()
        
        # build extended taxonomy
        tax_tab = pd.read_csv(db_meta['tax_file'], index_col=0).loc[self.ref_accs]
        # the tax_tab attribute is the extended taxonomy for each record
        self.tax_tab = self.tax_ext.loc[tax_tab.TaxID.values]
        self.tax_tab.index = tax_tab.index
    
    # load and map query file
    def set_query(self, query_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
        if query_file is None:
            return
        if self.db is None:
            raise Exception('You must set a Graboid databse before loading a query file.')
        
        # map query to the same reference sequence of the database
        map_file, acc_file, blast_report, acc_list, bounds, matrix, coverage, mesas = map_query(self.query_dir,
                                                                                                self.warn_dir,
                                                                                                query_file,
                                                                                                self.db_refdir,
                                                                                                evalue,
                                                                                                dropoff,
                                                                                                min_height,
                                                                                                min_width,
                                                                                                threads)
        self.query_file = query_file
        self.query_map_file = map_file
        self.query_acc_file = acc_file
        self.query_blast_report = blast_report
        self.query_accs = acc_list
        self.query_bounds = bounds
        self.query_matrix = matrix
        self.query_coverage = coverage
        self.query_mesas = mesas
    
    def load_query(self):
        if self.query_file is None:
            return
        query_npz = np.load(self.query_map_file)
        self.query_bounds = query_npz['bounds']
        self.query_matrix = query_npz['matrix']
        self.query_coverage = query_npz['coverage']
        self.query_mesas = query_npz['mesas']
        with open(self.query_acc_file, 'r') as handle:
            self.query_accs = handle.read().splitlines()
        
    # locate overlapping regions between reference and query maps
    def get_overlaps(self, min_width=10):
        self.overlaps = get_mesas_overlap(self.ref_mesas, self.query_mesas, min_width)
    
    # custom calibrate
    def custom_calibrate(self,
                         max_n,
                         step_n,
                         max_k,
                         step_k,
                         row_thresh,
                         col_thresh,
                         min_seqs,
                         rank,
                         min_n,
                         min_k,
                         criterion,
                         threads=1,
                         **kwargs):
        """Perform custom calibration for the reference database, parameters
        are the same as those used to direct the grid search.
        If the user provides w_starts and w_ends coordinates as kwargs, use
        those. Otherwise, use the selected overlaps.
        Calibration results are stored to a subfolder inside the calibration
        directory in the working dir, by default named using datetime, unless
        the user provides cal_dir kwarg as an alternative name.
        Updates last_calibration parameter with the output directory"""
        # set calibrator
        calibrator = cal_main.Calibrator()
        calibrator.set_database(self.db)
        
        if 'w_starts' in kwargs.keys() and 'w_ends' in kwargs.keys():
            calibrator.set_custom_windows(kwargs['w_starts'], kwargs['w_ends'])
        elif hasattr(self, 'overlaps'):
            calibrator.set_custom_windows(self.overlaps[:,0], self.overlaps[:,1])
        else:
            raise Exception('Missing parameters to set calibration windows. Run the get_overlapps method to get overlapping sections of query and reference data or provide matching sets of custom start and end positions')
        # set calibration directory
        try:
            cal_dir = kwargs['cal_dir']
        except KeyError:
            cal_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibrator.set_outdir(self.calibration_dir + '/' + cal_dir)
        
        calibrator.grid_search(max_n,
                               step_n,
                               max_k,
                               step_k,
                               self.cost_matrix,
                               row_thresh,
                               col_thresh,
                               min_seqs,
                               rank,
                               min_n,
                               min_k,
                               criterion,
                               collapse_hm=True,
                               threads=threads)
        self.last_calibration = calibrator.out_dir
    
    # select parameters
    def select_parameters(self, calibration_dir, w_idx, w_start, w_end, metric, report, *taxa):
        """Generate a parameter report, print to screen"""
        # work dir is the CALIBRATION DIRECTORY containing the full reports
        # window is an int indicating the window to select the parameters from
        # metric is a single capital letter indicating the metric to base the parameter selection on
        # taxa is an optional list of taxa to select the best parameters for
        
        # verify calibration directory
        if self.last_calibration is None:
            raise Exception('No calibration has been performed, run the calibration step before attempting to select parameters')
        
        if calibration_dir is None:
            # no calibration directory provided, default to the last calibration
            calibration_dir = self.last_calibration
        else:
            # calibration directory was provided, verify before selecting
            cal_dirs = os.listdir(self.calibration_dir)
            if not calibration_dir in cal_dirs:
                print(f'Calibration directory {calibration_dir} not found among the calibration runs')
                print('Available calibration directories include:')
                for cdir in cal_dirs:
                    print('\t' + cdir)
                raise Exception('No calibration directory')
                    
        # get report files
        reports = {'A': cal_metrics.read_full_report_tab(calibration_dir + '/report_accuracy.csv'),
                   'P': cal_metrics.read_full_report_tab(calibration_dir + '/report_precision.csv'),
                   'R': cal_metrics.read_full_report_tab(calibration_dir + '/report_recall.csv'),
                   'F': cal_metrics.read_full_report_tab(calibration_dir + '/report_f1.csv'),
                   'C': cal_metrics.read_full_report_tab(calibration_dir + '/report__cross_entropy.csv')}
        
        # get window identifier
        window_table = pd.read_csv(calibration_dir + '/windows.csv', index_col=0)
        if not w_idx is None:
            try:
                win_coors = window_table.loc[w_idx].to_numpy()
                window = w_idx
            except KeyError:
                raise Exception(f'Given window index not found, available indexes are: {window_table.index.values}')
        elif not w_start is None and not w_end is None:
            window_row = window_table.loc[(window_table.Start == w_start) & (window_table.End == w_end)]
            if window_row.shape[0] != 1:
                raise Exception('Invalid window coordinates! Too many or too few selected')
            window = window_row.index[0]
            win_coors = window_row.to_numpy()
        else:
            raise Exception('No valid window index or coordinates given, aborting')
            
        try:
            table = reports[metric]
        except KeyError:
            raise Exception(f'Given metric {metric} is not valid. Must be A, P, R, F or C')
        general_params, general_auto, taxa_params, taxa_auto, taxa_collapsed, taxa_diff = cls_parameters.report_parameters(table, window, metric == 'C', *taxa)
        
        def report():
            selection_metric = {'A':'accuracy',
                                'P':'precision',
                                'R':'recall',
                                'F':'f1',
                                'C':'cross entropy'}[metric]
            print(f'# Parameter selection for window {window} [{win_coors[0]} - {win_coors[1]}]')
            print(f'# Selection based on metric: {selection_metric}')
            print('\n### General parameters')
            print(general_params.sort_values('Mean', ascending = metric == 'C'))
            
            if not taxa_params is None:
                print('\n### Taxon specific parameters')
                print(taxa_params)
                print('\n### Scores for all parameter combinations')
                print(taxa_collapsed.T)
            
            if len(taxa_diff) > 0:
                print('\nThe following taxa were not present in the given reports:')
                for tax in taxa_diff:
                    print(tax)
                    
            methods = ['unweighted', 'wknn', 'dwknn']
            print('\nAutomatically selected parameters:')
            print(f'In general:\t n: {general_auto[0]}, k: {general_auto[1]}, method: {methods[general_auto[2]]}')
            if not taxa_auto is None:
                print(f'For the given taxa:\t n: {taxa_auto[0]}, k: {taxa_auto[1]}, method: {methods[taxa_auto[2]]}')
        if report:
            report()
        
        self.auto_start = win_coors[0]
        self.auto_end = win_coors[1]
        self.auto_n = general_auto[1]
        self.auto_k = general_auto[2]
        self.auto_mth = ['unweighted', 'wknn', 'dwknn'][general_auto[3]]
        if not taxa_auto is None:
            self.auto_n = taxa_auto[1]
            self.auto_k = taxa_auto[2]
            self.auto_mth = ['unweighted', 'wknn', 'dwknn'][taxa_auto[3]]
    
    # classify using different parameter combinations, register which parameter 
    def classify(self,
                 w_start,
                 w_end,
                 n,
                 k,
                 rank,
                 row_thresh,
                 col_thresh,
                 min_seqs,
                 criterion='orbit',
                 method='wknn',
                 save=True,
                 save_dir=''):
        method = method.lower()
        methods = {'unweighted':cls_classify.unweighted,
                   'wknn':cls_classify.wknn,
                   'dwknn':cls_classify.dwknn,
                   'rara':cls_classify.wknn_rara}
        
        # verify that method is valid
        try:
            classif_method = methods[method.lower()]
        except KeyError:
            raise Exception(f'Invalid method: {method}. Must be one of: "unweighted", "wknn", "dwknn"')
        # collapse reference and query matrices for the given window coordinates w_start & w_end, selecting the n most informative sites for the given rank
        ref_window, qry_window, qry_branches, win_tax, sites = cls_preprocess.collapse(self,
                                                                                       w_start = w_start,
                                                                                       w_end = w_end,
                                                                                       n = n,
                                                                                       rank = rank,
                                                                                       row_thresh = row_thresh,
                                                                                       col_thresh = col_thresh,
                                                                                       min_seqs = min_seqs)
        q_seqs = qry_window.shape[0]
        seqs_per_branch = np.array([len(br) for br in qry_branches])
        ref_mat = ref_window.window[:, sites]
        window_tax = self.tax_ext.loc[win_tax.index]
        
        # calculate distances
        distances = cls_distance.get_distances(qry_window, ref_mat, self.cost_matrix)
        
        # sort distances and get k nearest neighbours
        sorted_idxs = np.argsort(distances, axis=1)
        sorted_dists = np.sort(distances, axis=1)
        compressed = [np.unique(dist, return_index=True, return_counts = True) for dist in sorted_dists] # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
        # get k nearest orbital or orbital containing the kth neighbour
        if criterion == 'orbit':
            k_dists, k_positions, k_counts = cls_neighbours.get_k_nearest_orbit_V(compressed, k)
        else:
            k_dists, k_positions, k_counts = cls_neighbours.get_k_nearest_neigh_V(compressed, k)
        
        # assign classifications
        # clasif_id is a 2d array containing columns: query_idx, rank_idx, tax_id
        # classif_data is a 2d array containing columns: total_neighbours, mean_distances, std_distances, total_support and softmax_support
        classif_id, classif_data = cls_classify.classify_V(k_dists, k_positions, k_counts, sorted_idxs, window_tax, classif_method)
        
        #build reports
        pre_report = cls_report.build_prereport_V(classif_id, classif_data, seqs_per_branch)
        report = cls_report.build_report(pre_report, self.guide, q_seqs, seqs_per_branch) # TODO: remember to remove LinCode columns
        characterization = cls_report.characterize_sample(report)
        designation = cls_report.designate_branch_seqs(qry_branches, self.query_accs)
        
        # replace tax codes in pre report for their real names
        for rep in pre_report.values():
            rep['tax'] = self.guide.loc[rep.tax.values, 'SciName'].values
        
        if save:
            # save results to files
            # generate output directory
            out_dir = self.classif_dir + '/' + datetime.now().strftime("%Y%m%d_%H%M%S") if save_dir == '' else save_dir
            os.mkdir(out_dir)
            # save pre reports
            for rk, rk_prereport in pre_report.items():
                rk_prereport.to_csv(out_dir + f'/pre_report_{rk}.csv')
            report.to_csv(out_dir + '/report.csv')
            characterization.to_csv(out_dir + '/sample_characterization.csv')
            designation.to_csv(out_dir + '/sequence_designation.csv')
        else:
            return pre_report, report, characterization, designation
