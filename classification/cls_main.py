
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:25:00 2023

@author: hernan

Classifier class, handles steps: database loading, query blasting, custom calibration, parameter selection, classification and reporting
"""

#%% libraries
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import shutil
import time
# Graboid libraries
from calibration import cal_main
from calibration import cal_metrics
from classification import cost_matrix
from classification import cls_classify
from classification import cls_distance
from classification import cls_neighbours
from classification import cls_parameters
from classification import cls_plots
from classification import cls_preprocess
from classification import cls_report

from DATA import DATA
from mapping import director as mp

#%% set logger
logger = logging.getLogger('Graboid.Classification')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

#%% functions
def make_working_dir(work_dir):
    calibration_dir = work_dir + '/calibration'
    classif_dir = work_dir + '/classification'
    query_dir = work_dir + '/query'
    warn_dir = work_dir + '/warnings'
    metadata = work_dir + '/meta.json'
    
    os.makedirs(calibration_dir)
    os.makedirs(classif_dir)
    os.makedirs(query_dir)
    os.makedirs(warn_dir)
        
    return calibration_dir, classif_dir, query_dir, warn_dir, metadata

def check_work_dir(work_dir):
    # proposed directory must contain only the expected contents (or be empty)
    exp_contents = {'calibration', 'classification', 'query', 'warnings', 'meta.json'}
    try:
        dir_contents = set(os.listdir(work_dir))
    except FileNotFoundError:
        raise Exception(f'Error: Directory {work_dir} does not exist')
    if len(dir_contents) == 0:
        raise Exception(f'Error: Directory {work_dir} exists but it is empty')
    missing = exp_contents - dir_contents
    if len(missing) > 0:
        raise Exception(f'Error: Missing expected elements: {missing}')
    return f'{work_dir}/calibration/', f'{work_dir}/classification/', f'{work_dir}/query/', f'{work_dir}/warnings/', f'{work_dir}/meta.json'

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
    query_meta = {'query_file':fasta_file,
                  'mat_file':os.path.abspath(map_director.mat_file),
                  'acc_file':os.path.abspath(map_director.acc_file),
                  'blast_report':os.path.abspath(map_director.blast_report),
                  'bounds':map_director.bounds,
                  'mesas':map_director.mesas,
                  'evalue':evalue,
                  'dropoff':dropoff,
                  'min_height':min_height,
                  'min_width':min_width}
    return query_meta

def get_mesas_overlap(ref_map, qry_map, min_width=10):
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
    
    ref_npz = np.load(ref_map)
    ref_mesas = ref_npz['mesas']
    qry_npz = np.load(qry_map)
    qry_mesas = qry_npz['mesas']
    
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

# get param metrics (report calibration metrics and confusion matrix for the utilize dparameters)
def build_params_tuple(work_dir, w_start, w_end, n, k, method):
    # retrieve window data from the calibration directory
    if work_dir is None:
        raise Exception('No parameter selection was made, cannot report calibration metrics for used parameters')
    params_dict = np.load(work_dir + '/params.npz')
    
    missing_params = []
    # check that used parameters exist in the calibration run
    if not n in params_dict['n']:
        missing_params.append(f'n value {n} not present in calibration reports (existing values{params_dict["n"]})')
    if not k in params_dict['k']:
        missing_params.append(f'n value {n} not present in calibration reports (existing values{params_dict["n"]})')
    
    # check that selected window exists in the calibration run
    win_tab = pd.read_csv(work_dir + '/windows.csv', index_col=0)
    window_idx = win_tab.loc[(win_tab.Start == w_start) & (win_tab.End == w_end)].index.values
    if len(window_idx) == 0:
        # given coordinates don't match the calibration windows # maybe could do a quick calib, user shouldn't use an uncalibrated window
        missing_params.append(f'Used window coordinates (w_start: {w_start}, w_end: {w_end}) not found in calibration')
    
    # any params are missing from calibration
    if len(missing_params) > 0:
        raise Exception('\n'.join(missing_params))
        
    window_idx = window_idx[0]
    # build parameters tuple
    mth_idxs = {'u':0, 'w':1, 'd':2} # method indexes, used to turn method name to numeric value
    return (window_idx, n, k, mth_idxs[method[0]])

def report_param_metrics(work_dir, params):
    """Build metrics report for the used parameters"""
    # returns a pandas dataframe with index (Rank, Taxon) and columns Accuracy, Precision, Recall, f1, Cross_entropy
    
    # load calibration reports
    reports = {'Accuracy': cal_metrics.read_full_report_tab(work_dir + '/report_accuracy.csv'),
               'Precision': cal_metrics.read_full_report_tab(work_dir + '/report_precision.csv'),
               'Recall': cal_metrics.read_full_report_tab(work_dir + '/report_recall.csv'),
               'f1': cal_metrics.read_full_report_tab(work_dir + '/report_f1.csv'),
               'Cross_entropy': cal_metrics.read_full_report_tab(work_dir + '/report__cross_entropy.csv')}
    
    # build and populate results table
    param_metrics = pd.DataFrame(index=reports['Accuracy'].index, columns = reports.keys())
    for metric, tab in reports.items():
        met_scores = cls_parameters.get_params_scores(tab, params)[0]
        param_metrics[metric] = met_scores
    return param_metrics

def build_param_confusion(work_dir, params, guide):
    """Build the confusion matrix for the used parameters. Rows: real values, columns: predicted values"""
    
    def get_total(matrix, index):
        """Get sum of real/predicted values"""
        results = pd.Series(0, index=index, dtype=int)
        for rk in matrix.T:
            taxa, counts = np.unique(rk[rk >= 0], return_counts=True)
            results.loc[taxa] = counts
        return results
    
    # retrieve real and predicted taxa for the utilized parameters
    classifs_file = work_dir + '/classifs.npz'
    win_taxa = np.load(work_dir + '/win_taxa.npz')
    key = '_'.join(np.array(params).astype(str))
    try:
        pred = np.load(classifs_file)[key]
        real = win_taxa[key[0]]
    except KeyError:
        raise Exception(f'Parameters {key} not found among the calibration predictions')
    
    # build confusion matrix
    confusion = cal_metrics.build_confusion(pred, real)
    # add total roes
    total_real = get_total(real, confusion.index)
    total_missed = total_real - confusion.sum(1)
    total_pred = get_total(pred, confusion.index)
    total_pred['Total_real'] = total_pred.sum()
    total_pred['Missed'] = total_missed.sum()
    confusion['Total_real'] = total_real
    confusion['Missed'] = total_missed
    confusion.loc['Total_pred'] = total_pred
    
    # update indexes
    guide = guide.copy()
    guide.loc['Total_real', 'Rank'] = 'Total_real'
    guide.loc['Missed', 'Rank'] = 'Missed'
    guide.loc['Total_pred', 'Rank'] = 'Total_pred'
    confusion.index = pd.MultiIndex.from_frame(guide.loc[confusion.index, ['Rank', 'SciName']], names = ['Rank', 'Taxon'])
    confusion.columns = pd.MultiIndex.from_frame(guide.loc[confusion.columns, ['Rank', 'SciName']], names = ['Rank', 'Taxon'])
    
    return confusion

def summary_report(date,
                   run_time,
                   database,
                   w_start,
                   w_end,
                   n,
                   sites,
                   k,
                   mth,
                   criterion,
                   row_thresh,
                   col_thresh,
                   min_seqs,
                   rank,
                   ref_seqs,
                   ref_taxa,
                   qry_branches,
                   report,
                   designation,
                   ranks,
                   files_pre,
                   file_classif,
                   file_chara,
                   file_assign,
                   file_parammetric,
                   file_paramconf):
    sep = '#' * 40
    
    # get reference taxa per rank
    taxa_per_rk = pd.Series(index=ref_taxa.columns, dtype=int)
    for rk, col in ref_taxa.T.iterrows():
        taxa_per_rk[rk] = len(col.unique())
    
    # get assigned branches per rank
    branches = len(qry_branches)
    qry_seqs = len(np.concatenate(qry_branches))
    seqs_p_branch = designation.branch.value_counts()
    
    assigned = pd.DataFrame(index = ranks, columns=['Branches', '% branches', 'Sequences', '% sequences'])
    for rk in ranks:
        nnull_support = ~report[(rk, 'support')].isna()
        nnull_support = nnull_support[nnull_support] # get only branches with non null support
        assigned.loc[rk, 'Branches'] = nnull_support.sum()
        assigned.loc[rk, 'Sequences'] = seqs_p_branch[nnull_support.index].sum()
    assigned['% branches'] = ((1 - ((branches - assigned['Branches']) / branches)) * 100).apply(lambda x: round(x, 3))
    assigned['% sequences'] = ((1 - ((qry_seqs - assigned['Sequences']) / qry_seqs)) * 100).apply(lambda x: round(x, 3))
    
    lines = ['Classification summary',
             sep,
             f'Date: {date}',
             f'Run time: {run_time:.2f} seconds',
             '',
             sep,
             'Parameters\n',
             f'Database: {database}',
             '',
             f'Window start: {w_start}',
             f'Window end: {w_end}',
             '',
             f'n sites: {n}',
             f'Selected sites: {sites}',
             f'k neighbours: {k}',
             f'Weighting method: {mth}',
             f'Neighbour criterion: {criterion}',
             '',
             f'Max unknowns per sequence: {row_thresh * 100} %',
             f'Max unknowns per site: {col_thresh * 100} %',
             f'Min non-redundant sequences: {min_seqs}',
             f'Rank used for site selection: {rank}',
             '',
             sep,
             'Sequences:\n',
             'Reference',
             f'Reference sequences: {ref_seqs}',
             'Reference taxa per rank in window:',
             repr(taxa_per_rk.to_frame(name='Taxa')),
             '',
             'Query',
             f'Query_sequences: {qry_seqs}',
             f'Total branches: {branches}',
             '',
             sep,
             'Results:\n',
             'Assigned branches:',
             repr(assigned),
             '',
             'Result files:',
             'Pre reports:',
             '\n'.join([f'\t{pre}' for pre in files_pre]),
             f'Classification report: {file_classif}',
             f'Characterization report: {file_chara}',
             f'Branch designation report: {file_assign}',
             f'Parameter calibration report: {file_parammetric}',
             f'Parameter confusion matrix: {file_paramconf}']
    return '\n'.join(lines)

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
        self.__active_calibration = None
        
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
                'auto_mth':self.auto_mth,
                'active_calibration':self.active_calibration}
    
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
    def active_calibration(self):
        return self.__active_calibration
    @active_calibration.setter
    def active_calibration(self, active):
        self.__active_calibration = active
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
        try:
            self.__auto_start = int(start)
            self.update_meta()
        except TypeError:
            pass
    @property
    def auto_end(self):
        return self.__auto_end
    @auto_end.setter
    def auto_end(self, end):
        try:
            self.__auto_end = int(end)
            self.update_meta()
        except TypeError:
            pass
    @property
    def auto_n(self):
        return self.__auto_n
    @auto_n.setter
    def auto_n(self, n):
        try:
            self.__auto_n = int(n)
            self.update_meta()
        except TypeError:
            pass
    @property
    def auto_k(self):
        return self.__auto_k
    @auto_k.setter
    def auto_k(self, k):
        try:
            self.__auto_k = int(k)
            self.update_meta()
        except TypeError:
            pass
    @property
    def auto_mth(self):
        return self.__auto_mth
    @auto_mth.setter
    def auto_mth(self, mth):
        try:
            self.__auto_mth = mth
            self.update_meta()
        except TypeError:
            pass
    
    def update_meta(self):
        with open(self.out_dir + '/meta.json', 'w') as handle:
            json.dump(self.meta, handle)
        
class Classifier(ClassifierBase):
    def read_meta(self):
        with open(self.work_dir + '/meta.json', 'r') as handle:
            meta = {'db':None,
                    'query_file':None,
                    'query_map_file':None,
                    'query_acc_file':None,
                    'last_calibration':None,
                    'transition':None,
                    'transversion':None}
    def set_work_dir(self, work_dir):
        # verify that given_work dir is valid, raise exception if it isn't
        try:
            calibration_dir, classification_dir, query_dir, warnings_dir, metadata = check_work_dir(work_dir)
        except:
            raise
        self.work_dir = work_dir
        self.calibration_dir = calibration_dir
        self.classification_dir = classification_dir
        self.query_dir = query_dir,
        self.warnings_dir = warnings_dir
        self.meta = metadata
        
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
                        self.auto_start = meta['auto_start']
                        self.auto_end = meta['auto_end']
                        self.auto_n = meta['auto_n']
                        self.auto_k = meta['auto_k']
                        self.auto_mth = meta['auto_mth']
                        self.active_calibration = meta['active_calibration']
                        self.load_query()
                except FileNotFoundError:
                    raise Exception('Specified output directory exists but cannot be verified as a Graboid classification directory. Recommend overwrtiting it or using a different name')
                    # TODO: maybe include the option of verifying if it is a classif dir with a damaged/mising meta file
        # create directories (if absent or set to overwrite)
        os.makedirs(self.calibration_dir, exist_ok=True)
        os.makedirs(self.classif_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.warn_dir, exist_ok=True)
        
        fh = logging.FileHandler(self.out_dir + '/classification.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        warn_handler = logging.FileHandler(self.warn_dir + '/warnings.log')
        warn_handler.setLevel(logging.WARNING)
        logger.addHandler(warn_handler)
    
    def set_cost_matrix(self, transition, transversion):
        self.transition = transition
        self.transversion = transversion
        self.cost_matrix = cost_matrix.cost_matrix(transition, transversion)
    
    def set_database2(self, database):
        self.db = database
        self.db_meta = DATA.DBASE_INFO[database]
        
    def set_database(self, database):
        # this method should only be done when creating a working dir
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
            cal_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    def select_parameters(self, calibration_dir, w_idx, w_start, w_end, metric, show, *taxa):
        """Generate a parameter report, print to screen"""
        # work dir is the CALIBRATION DIRECTORY containing the full reports
        # window is an int indicating the window to select the parameters from
        # metric is a single capital letter indicating the metric to base the parameter selection on
        # show: boolean, print selection summary
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
            calibration_dir = self.calibration_dir + '/' + calibration_dir
        self.active_calibration = calibration_dir
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
        if show:
            report()
        
        self.auto_start = win_coors[0]
        self.auto_end = win_coors[1]
        self.auto_n = general_auto[0]
        self.auto_k = general_auto[1]
        self.auto_mth = ['unweighted', 'wknn', 'dwknn'][general_auto[2]]
        if not taxa_auto is None:
            self.auto_n = taxa_auto[0]
            self.auto_k = taxa_auto[1]
            self.auto_mth = ['unweighted', 'wknn', 'dwknn'][taxa_auto[2]]
    
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
        # mark intial time
        t0 = time.time()
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
        logger.info('Collapsing sequences...')
        t_collapse_0 = time.time()
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
        t_collapse_1 = time.time()
        logger.info(f'Finished collapsing in {t_collapse_1 - t_collapse_0:.2f} seconds')
        
        # calculate distances
        logger.info('Calculating distances...')
        t_distances_0 = time.time()
        distances = cls_distance.get_distances(qry_window, ref_mat, self.cost_matrix)
        t_distances_1 = time.time()
        logger.info(f'Finished calculating distances in {t_distances_1 - t_distances_0:.2f} seconds')
        
        # sort distances and get k nearest neighbours
        logger.info('Sorting and counting neighbours...')
        t_neighbours_0 = time.time()
        sorted_idxs = np.argsort(distances, axis=1)
        sorted_dists = np.sort(distances, axis=1)
        compressed = [np.unique(dist, return_index=True, return_counts = True) for dist in sorted_dists] # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
        # get k nearest orbital or orbital containing the kth neighbour
        if criterion == 'orbit':
            k_nearest = cls_neighbours.get_knn_orbit(compressed, k)
        else:
            k_nearest = cls_neighbours.get_knn_neigh(compressed, k)
        k_dists = np.array([row[0] for row in k_nearest])
        k_counts = np.array([row[2] for row in k_nearest])
        k_positions = k_counts.sum(1)
        t_neighbours_1 = time.time()
        logger.info(f'Finished sorting neighbours in {t_neighbours_1 - t_neighbours_0:.2f} seconds')
        
        # assign classifications
        # clasif_id is a 2d array containing columns: query_idx, rank_idx, tax_id
        # classif_data is a 2d array containing columns: total_neighbours, mean_distances, std_distances, total_support and softmax_support
        logger.info('Classifying...')
        t_classif_0 = time.time()
        classif_id, classif_data = cls_classify.classify_V(k_dists, k_positions, k_counts, sorted_idxs, window_tax.to_numpy(), classif_method)
        t_classif_1 = time.time()
        logger.info(f'Finished classsifying sequences in {t_classif_1 - t_classif_0:.2f} seconds')
        
        #build reports
        logger.info('Building reports...')
        t_reports_0 = time.time()
        pre_report = cls_report.build_prereport_V(classif_id, classif_data, seqs_per_branch, self.ranks)
        report = cls_report.build_report(pre_report, q_seqs, seqs_per_branch, self.guide)
        characterization = cls_report.characterize_sample(report)
        designation = cls_report.designate_branch_seqs(qry_branches, self.query_accs)
        
        # replace tax codes in pre report for their real names
        for rep in pre_report.values():
            rep['tax'] = self.guide.loc[rep.tax_id.values, 'SciName'].values
        t_reports_1 = time.time()
        logger.info(f'Finished building reports in {t_reports_1 - t_reports_0:.2f} seconds')
                
        if save:
            # save results to files
            # generate output directory
            out_dir = self.classif_dir + '/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if save_dir == '' else save_dir
            os.mkdir(out_dir)
            # save pre reports
            for rk, rk_prereport in pre_report.items():
                rk_prereport.to_csv(out_dir + f'/pre_report_{rk}.csv')
                rk_figure = cls_plots.plot_pre_report(rk_prereport, rk) # Remember that fig_width and tax_height are adjustable
                rk_figure.savefig(out_dir + f'results_{rk}.png')
            report.to_csv(out_dir + '/report.csv')
            characterization.to_csv(out_dir + '/sample_characterization.csv')
            designation.to_csv(out_dir + '/sequence_designation.csv')
            
            # report param metrics
            try:
                logger.info('Attempting to retrieve metrics for the used parameters...')
                t_metrics_0 = time.time()
                params = build_params_tuple(self.active_calibration, w_start, w_end, n, k, method)
                param_metrics = report_param_metrics(self.active_calibration, params)
                param_confusion = build_param_confusion(self.active_calibration, params, self.guide)
                
                param_metrics.to_csv(out_dir + '/calibration_metrics.csv', sep='\t')
                param_confusion.to_csv(out_dir + '/confusion.csv')
                t_metrics_1 = time.time()
                logger.info(f'Finished retrieving metrics in {t_metrics_1 - t_metrics_0:.2f} seconds')
                # TODO: tell the user how to generate the calibration reports
            except Exception as excp:
                logger.warning(excp)
        t1 = time.time()
        logger.info(f'Finished classification in {t1 - t0:.2f} seconds')
        # write summary
        date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        summ_report = summary_report(date,
                                     run_time = t1 - t0,
                                     database = self.db,
                                     w_start = w_start,
                                     w_end = w_end,
                                     n = n,
                                     sites = sites,
                                     k = k,
                                     mth = method,
                                     criterion = criterion,
                                     row_thresh = row_thresh,
                                     col_thresh = col_thresh,
                                     min_seqs = min_seqs,
                                     rank = rank,
                                     ref_seqs = ref_window.window.shape[0],
                                     ref_taxa = self.tax_ext.loc[ref_window.taxonomy],
                                     qry_branches = qry_branches,
                                     report = report,
                                     designation = designation,
                                     ranks = self.ranks,
                                     files_pre = [out_dir + f'/pre_report_{rk}.csv' for rk in pre_report.keys()],
                                     file_classif = out_dir + '/report.csv',
                                     file_chara = out_dir + '/sample_characterization.csv',
                                     file_assign = out_dir + '/sequence_designation.csv',
                                     file_parammetric = out_dir + '/calibration_metrics.csv',
                                     file_paramconf = out_dir + '/confusion.csv')
        with open(out_dir + '/classification_summary.txt', 'w') as handle:
            handle.write(summ_report)
        return pre_report, report, characterization, designation

#%%
{'db':None,
'query':None,
'last_calibration':None,
'transition':None,
'transversion':None}

class Classifier2:
    @property
    def meta(self):
        return {'db':self.db,
                'query':self.query,
                'transition':self.transition,
                'transversion':self.transversion,
                'calibrations':self.calibrations}
    
    def __init__(self,
                 work_dir,
                 database=None,
                 query=None,
                 qry_evalue=0.005,
                 qry_dropoff=0.05, 
                 qry_min_height=0.1,
                 qry_min_width=2,
                 transition=1,
                 transversion=1,
                 threads=1):
        
        self.set_work_dir(work_dir)
        self.set_database(database)
        self.set_query(query, qry_evalue, qry_dropoff, qry_min_height, qry_min_width, threads)
        if self.transition is None:
            self.transition = transition
        if self.transversion is None:
            self.transversion = transversion
        self.update_meta()
    
    def update_meta(self):
        with open(self.meta_file, 'w') as handle:
            json.dump(self.meta, handle, indent=2)
            
    def set_work_dir(self, work_dir):
        # if work dir doesn't exist, create it, else check that existing dir is valid
        if not os.path.isdir(work_dir):
            calibration_dir, classification_dir, query_dir, warnings_dir, metadata = make_working_dir(work_dir)
            self.db = None
            self.query = None
            self.transition = None
            self.transversion = None
            self.calibrations = []
        else:
            # verify that given_work dir is valid, raise exception if it isn't
            try:
                calibration_dir, classification_dir, query_dir, warnings_dir, metadata = check_work_dir(work_dir)
                # update parameters
                with open(metadata, 'r') as handle:
                    meta = json.load(handle)
                    for attr, val in meta.items():
                        setattr(self, attr, val)
            except:
                raise
        self.work_dir = work_dir
        self.calibration_dir = calibration_dir
        self.classification_dir = classification_dir
        self.query_dir = query_dir,
        self.warnings_dir = warnings_dir
        self.meta_file = metadata
        
        # set logger
        fh = logging.FileHandler(self.work_dir + '/classification.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        warn_handler = logging.FileHandler(self.warn_dir + '/warnings.log')
        warn_handler.setLevel(logging.WARNING)
        logger.addHandler(warn_handler)
    
    def set_database(self, database):
        if database is None:
            # no database given
            return
        if self.db is None:
            if not DATA.database_exists(database):
                raise Exception(f'Error: Database {database} not found among [{" ".join(DATA.DBASES)}]')
            self.db = DATA.DBASE_INFO[database]
            return
        if self.db['name'] == database:
            # database is already set
            logger.warning(f'Attempted to set {database} as working graboid database when {self.db["name"]} is already set')
        return
        
    # load and map query file
    def set_query(self, query_file, evalue=0.005, dropoff=0.05, min_height=0.1, min_width=2, threads=1):
        if query_file is None:
            return
        if self.db is None:
            raise Exception('You must set a Graboid databse before loading a query file.')
        if mp.check_fasta(query_file) == 0:
            raise Exception(f'Error: Query file {query_file} is not a valid fasta file')
        if self.query is None:
            # map query to the same reference sequence of the database
            self.query = map_query(self.query_dir,
                                   self.warn_dir,
                                   query_file,
                                   self.db_refdir,
                                   evalue,
                                   dropoff,
                                   min_height,
                                   min_width,
                                   threads)
            return
        if query_file == self.query['query_file']:
            logger.warning(f'Attempted to set {query_file} as query file when {self.query["query_file"]} is already set')
        return
    
    # locate overlapping regions between reference and query maps
    def get_overlaps(self, min_width=10):
        self.overlaps = get_mesas_overlap(self.db['mat_file'], self.query['mat_file'], min_width)
    
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
        """
        Perform custom calibration for the reference database, parameters
        are the same as those used to direct the grid search.
        If the user provides w_starts and w_ends coordinates as kwargs, use
        those. Otherwise, use the selected overlaps.
        Calibration results are stored to a subfolder inside the calibration
        directory in the working dir, by default named using datetime, unless
        the user provides cal_dir kwarg as an alternative name.
        Updates calibrations attribute with the output directory
        """
        # set calibration directory
        try:
            cal_dir = kwargs['cal_dir']
        except KeyError:
            cal_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # set calibrator
        calibrator = cal_main.Calibrator(self.calibration_dir + '/' + cal_dir)
        calibrator.set_database(self.db['name'])
        
        if 'w_starts' in kwargs.keys() and 'w_ends' in kwargs.keys():
            calibrator.set_custom_windows(kwargs['w_starts'], kwargs['w_ends'])
        elif hasattr(self, 'overlaps'):
            calibrator.set_custom_windows(self.overlaps[:,0], self.overlaps[:,1])
        else:
            raise Exception('Missing parameters to set calibration windows. Run the get_overlapps method to get overlapping sections of query and reference data or provide matching sets of custom start and end positions')
        
        # generate cost matrix
        cost_mat = cost_matrix.cost_matrix(self.transition, self.transversion)
        
        calibrator.grid_search(max_n,
                               step_n,
                               max_k,
                               step_k,
                               cost_mat,
                               row_thresh,
                               col_thresh,
                               min_seqs,
                               rank,
                               min_n,
                               min_k,
                               criterion,
                               collapse_hm=True,
                               threads=threads)
        self.calibrations.append(cal_dir)
    
    # select parameters
    def select_parameters(self, calibration_dir, w_idx, w_start, w_end, metric, show, *taxa):
        """Generate a parameter report, print to screen"""
        # work dir is the CALIBRATION DIRECTORY containing the full reports
        # window is an int indicating the window to select the parameters from
        # metric is a single capital letter indicating the metric to base the parameter selection on
        # show: boolean, print selection summary
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
            calibration_dir = self.calibration_dir + '/' + calibration_dir
        self.active_calibration = calibration_dir
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
        if show:
            report()
        
        self.auto_start = win_coors[0]
        self.auto_end = win_coors[1]
        self.auto_n = general_auto[0]
        self.auto_k = general_auto[1]
        self.auto_mth = ['unweighted', 'wknn', 'dwknn'][general_auto[2]]
        if not taxa_auto is None:
            self.auto_n = taxa_auto[0]
            self.auto_k = taxa_auto[1]
            self.auto_mth = ['unweighted', 'wknn', 'dwknn'][taxa_auto[2]]
    
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
        # mark intial time
        t0 = time.time()
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
        logger.info('Collapsing sequences...')
        t_collapse_0 = time.time()
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
        t_collapse_1 = time.time()
        logger.info(f'Finished collapsing in {t_collapse_1 - t_collapse_0:.2f} seconds')
        
        # calculate distances
        logger.info('Calculating distances...')
        t_distances_0 = time.time()
        distances = cls_distance.get_distances(qry_window, ref_mat, self.cost_matrix)
        t_distances_1 = time.time()
        logger.info(f'Finished calculating distances in {t_distances_1 - t_distances_0:.2f} seconds')
        
        # sort distances and get k nearest neighbours
        logger.info('Sorting and counting neighbours...')
        t_neighbours_0 = time.time()
        sorted_idxs = np.argsort(distances, axis=1)
        sorted_dists = np.sort(distances, axis=1)
        compressed = [np.unique(dist, return_index=True, return_counts = True) for dist in sorted_dists] # for each qry_sequence, get distance groups, as well as the index where each group begins and the count for each group
        # get k nearest orbital or orbital containing the kth neighbour
        if criterion == 'orbit':
            k_nearest = cls_neighbours.get_knn_orbit(compressed, k)
        else:
            k_nearest = cls_neighbours.get_knn_neigh(compressed, k)
        k_dists = np.array([row[0] for row in k_nearest])
        k_counts = np.array([row[2] for row in k_nearest])
        k_positions = k_counts.sum(1)
        t_neighbours_1 = time.time()
        logger.info(f'Finished sorting neighbours in {t_neighbours_1 - t_neighbours_0:.2f} seconds')
        
        # assign classifications
        # clasif_id is a 2d array containing columns: query_idx, rank_idx, tax_id
        # classif_data is a 2d array containing columns: total_neighbours, mean_distances, std_distances, total_support and softmax_support
        logger.info('Classifying...')
        t_classif_0 = time.time()
        classif_id, classif_data = cls_classify.classify_V(k_dists, k_positions, k_counts, sorted_idxs, window_tax.to_numpy(), classif_method)
        t_classif_1 = time.time()
        logger.info(f'Finished classsifying sequences in {t_classif_1 - t_classif_0:.2f} seconds')
        
        #build reports
        logger.info('Building reports...')
        t_reports_0 = time.time()
        pre_report = cls_report.build_prereport_V(classif_id, classif_data, seqs_per_branch, self.ranks)
        report = cls_report.build_report(pre_report, q_seqs, seqs_per_branch, self.guide)
        characterization = cls_report.characterize_sample(report)
        designation = cls_report.designate_branch_seqs(qry_branches, self.query_accs)
        
        # replace tax codes in pre report for their real names
        for rep in pre_report.values():
            rep['tax'] = self.guide.loc[rep.tax_id.values, 'SciName'].values
        t_reports_1 = time.time()
        logger.info(f'Finished building reports in {t_reports_1 - t_reports_0:.2f} seconds')
                
        if save:
            # save results to files
            # generate output directory
            out_dir = self.classif_dir + '/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if save_dir == '' else save_dir
            os.mkdir(out_dir)
            # save pre reports
            for rk, rk_prereport in pre_report.items():
                rk_prereport.to_csv(out_dir + f'/pre_report_{rk}.csv')
                rk_figure = cls_plots.plot_pre_report(rk_prereport, rk) # Remember that fig_width and tax_height are adjustable
                rk_figure.savefig(out_dir + f'results_{rk}.png')
            report.to_csv(out_dir + '/report.csv')
            characterization.to_csv(out_dir + '/sample_characterization.csv')
            designation.to_csv(out_dir + '/sequence_designation.csv')
            
            # report param metrics
            try:
                logger.info('Attempting to retrieve metrics for the used parameters...')
                t_metrics_0 = time.time()
                params = build_params_tuple(self.active_calibration, w_start, w_end, n, k, method)
                param_metrics = report_param_metrics(self.active_calibration, params)
                param_confusion = build_param_confusion(self.active_calibration, params, self.guide)
                
                param_metrics.to_csv(out_dir + '/calibration_metrics.csv', sep='\t')
                param_confusion.to_csv(out_dir + '/confusion.csv')
                t_metrics_1 = time.time()
                logger.info(f'Finished retrieving metrics in {t_metrics_1 - t_metrics_0:.2f} seconds')
                # TODO: tell the user how to generate the calibration reports
            except Exception as excp:
                logger.warning(excp)
        t1 = time.time()
        logger.info(f'Finished classification in {t1 - t0:.2f} seconds')
        # write summary
        date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        summ_report = summary_report(date,
                                     run_time = t1 - t0,
                                     database = self.db,
                                     w_start = w_start,
                                     w_end = w_end,
                                     n = n,
                                     sites = sites,
                                     k = k,
                                     mth = method,
                                     criterion = criterion,
                                     row_thresh = row_thresh,
                                     col_thresh = col_thresh,
                                     min_seqs = min_seqs,
                                     rank = rank,
                                     ref_seqs = ref_window.window.shape[0],
                                     ref_taxa = self.tax_ext.loc[ref_window.taxonomy],
                                     qry_branches = qry_branches,
                                     report = report,
                                     designation = designation,
                                     ranks = self.ranks,
                                     files_pre = [out_dir + f'/pre_report_{rk}.csv' for rk in pre_report.keys()],
                                     file_classif = out_dir + '/report.csv',
                                     file_chara = out_dir + '/sample_characterization.csv',
                                     file_assign = out_dir + '/sequence_designation.csv',
                                     file_parammetric = out_dir + '/calibration_metrics.csv',
                                     file_paramconf = out_dir + '/confusion.csv')
        with open(out_dir + '/classification_summary.txt', 'w') as handle:
            handle.write(summ_report)
        return pre_report, report, characterization, designation