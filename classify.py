#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:57:24 2023

@author: hernan
Classification director. Handles custom calibration and classification of query files
"""

#%% libraries
import argparse
import os
import shutil
# graboid libraries
from classification import cls_main
from classification import cls_plots
#%%
### classification steps
## preparations
# set working directory
# set database
# set query
## operations
# custom calibration
#   get overlapping regions
# classification

# initialize classifier, set working directory
# set database (if not already set, else skip this step): NOTE. For simplicity, database cannot be changed for a working directory
# set query, report overlapping regions
# perform custom calibration of overlapping reions
# classify query

def is_graboid_dir(path):
    # proposed directory must contain only the expected contents
    exp_contents = {'calibration', 'classification', 'query', 'meta.json'}
    dir_contents = set(os.path.listdir(path))
    not_found = exp_contents.difference(dir_contents)
    not_belong = dir_contents.difference(exp_contents)
    check = True
    warn_text = 'Inconsistencies found: '
    if len(not_found) > 0:
        warn_text += f'Missing expected elements: {not_found}.'
        check = False
    if len(not_belong) > 0:
        warn_text += f'Found extraneous elements: {not_belong}.'
        check = False
    
    if check:
        return
    raise Exception(warn_text)

def preparation(out_dir,
                transition,
                transversion,
                database,
                query,
                qry_evalue=0.005,
                qry_dropoff=0.05,
                qry_min_height=0.1,
                qry_min_width=2,
                min_overlap_width=10,
                overwrite=False,
                overwrite_force=False,
                threads=1):
    
    # set up graboid database and query file
    ## Working directory
    # check overwrite (also check that potential directory is a graboid dir)
    if os.path.isdir(out_dir):
        try:
            is_graboid_dir(out_dir)
        except Exception as excp:
            if overwrite_force:
                shutil.rmtree(out_dir)
                overwrite = True # just in case
            else:
                raise excp + '\nProposed directory cannot be assertained as agraboid directory. Overwrite at your own risk with option --OVERWRITE'
        
        if overwrite:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            raise Exception('Proposed working directory already exists. Overwrite with option --overwrite. WARNING: this will remove all existing calibration and classification data.')
    
    classifier = cls_main.Classifier()
    classifier.set_outdir(out_dir, overwrite)
    try:
        classifier.set_database(database)
    except Exception as excp:
        # invalid database remove generated riectories
        shutil.rmtree(out_dir)
        raise excp
    classifier.set_cost_matrix(transition, transversion)
    
    # Query
    classifier.set_query(query, qry_evalue, qry_dropoff, qry_min_height, qry_min_width, threads)
    classifier.get_overlaps(min_overlap_width)
    cls_plots.plot_ref_v_qry(classifier.ref_coverage,
                             classifier.ref_mesas,
                             classifier.qry_coverage,
                             classifier.qry_mesas,
                             classifier.overlaps,
                             ref_title = classifier.db,
                             qry_title = classifier.query_file,
                             out_file = classifier.query_dir + '/coverage.png')

def operation_calibrate(classifier, min_overlap_width, max_n, step_n, max_k, step_k, row_thresh, col_thresh, min_seqs, rank, min_n, min_k, criterion, threads, **kwargs):
    classifier.get_overlaps(min_overlap_width)
    classifier.custom_calibrate(max_n,
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
                                threads,
                                **kwargs)

def operation_classify(classifier, w_start, w_end, n, k, rank, row_thresh, col_thresh, min_seqs, criterion, method):
    classifier.classify(w_start, w_end, n, k, rank, row_thresh, col_thresh, min_seqs, criterion, method)

#%%
parser = argparse.ArgumentParser(prog='Graboid CLASSIFY',
                                 description='Common parameters for GRABOID CLASSIFY operations:')
parser.add_argument('out_dir',
                    help='Working directory for the classification run',
                    type=str)

# common arguments
parser.add_argument('--criterion',
                    help='Criterion for neighbour selection. Default: orbit',
                    type=str,
                    default='orbit')

subparsers = parser.add_subparsers(title='Operations')
prp_parser = subparsers.add_parser('prep',
                                   help='Prepare graboid working directory',
                                   description='Set up working directory, select graboid database, load and map query file')
cal_parser = subparsers.add_parser('calibrate',
                                   prog='graboid CLASSIFY [common options]',
                                   help='Perform a custom calibration operation',
                                   description='Parameters specific to the calibration operation')
cls_parser = subparsers.add_parser('classify',
                                   prog='graboid CLASSIFY [common options]',
                                   help='Classify the provided query',
                                   description='Parameters specific to the classification operation')

# preparation arguments
prp_parser.set_defaults(mode='prep')
prp_parser.add_argument('-s', '--transition',
                        default=1,
                        help='Cost for transition-type substitutions (A <-> T, C <-> G)',
                        type=float)
prp_parser.add_argument('-v', '--transversion',
                        default=1,
                        help='Cost for transversion-type substitutions (A/T <-> C/G)',
                        type=float)
prp_parser.add_argument('database',
                        help='Database to be used in the classification',
                        type=str)
prp_parser.add_argument('query',
                        help='Query file in FASTA format',
                        type=str)
prp_parser.add_argument('--evalue',
                        default=0.005,
                        help='E-value threshold for the BLAST matches when mapping the query file. Default: 0.005',
                        type=float)
prp_parser.add_argument('--dropoff',
                        help='Percentage of mesa height drop to determine a border. Default: 0.05',
                        type=float,
                        default=0.05)
prp_parser.add_argument('--min_height',
                        help='Minimum sequence coverage to consider for a mesa candidate (percentage of maximum coverage). Default: 0.1',
                        type=float,
                        default=0.1)
prp_parser.add_argument('--min_width',
                        help='Minimum width needed for a mesa candidate to register. Default: 2',
                        type=int,
                        default=2)
prp_parser.add_argument('-ow', '--min_overlap_width',
                        default=10,
                        help='Minimum overlap width betwen reference and query mesas. Default: 10',
                        type=int)
prp_parser.add_argument('--overwrite',
                        help='Overwrite provided working directory (if present). All existing calibration and classification data will be lost',
                        action='store_true')
prp_parser.add_argument('--force_overwrite',
                        help='Overwrite working directory, even if it is not a graboid directory. WARNING: the established directory will be deleted, do this at your own risk')
prp_parser.add_argument('--threads',
                        help='Threads to use when building the alignment. Default: 1',
                        type=int,
                        default=1)

prp_parser.add_argument('-rk', '--rank',
                        default='genus',
                        help='Taxonomic rank to be used for feature selection. Default: genus',
                        type=str)
prp_parser.add_argument('-rt', '--row_thresh',
                        default=0.2,
                        help='Empty row threshold. Default: 0.2',
                        type=float)
prp_parser.add_argument('-ct', '--col_thresh',
                        default=0.2,
                        help='Empty column threshold. Default: 0.2',
                        type=float)
prp_parser.add_argument('-ms', '--min_seqs',
                        default=10,
                        help='Minimum number of sequences allowed per taxon. Default: 10',
                        type=int)


# calibration arguments
cal_parser.set_defaults(mode='calibrate')
cal_parser.add_argument('-ow', '--min_overlap_width',
                        default=10,
                        help='Minimum overlap width betwen reference and query mesas. Default: 10',
                        type=int)
cal_parser.add_argument('-mn', '--max_n',
                        default=30,
                        help='Max value of n. Default: 30',
                        type=int)
cal_parser.add_argument('-sn', '--step_n',
                        default=5,
                        help='Rate of increase of n. Default: 5',
                        type=int)
cal_parser.add_argument('-mk', '--max_k',
                        default=15,
                        help='Max value of K. Default: 15',
                        type=int)
cal_parser.add_argument('-sk', '--step_k',
                        default=2,
                        help='Rate of increase of K. Default: 2',
                        type=int)
cal_parser.add_argument('-rt', '--row_thresh',
                        default=0.2,
                        help='Empty row threshold. Default: 0.2',
                        type=float)
cal_parser.add_argument('-ct', '--col_thresh',
                        default=0.2,
                        help='Empty column threshold. Default: 0.2',
                        type=float)
cal_parser.add_argument('-ms', '--min_seqs',
                        default=10,
                        help='Minimum number of sequences allowed per taxon. Default: 10',
                        type=int)
cal_parser.add_argument('-rk', '--rank',
                        default='genus',
                        help='Taxonomic rank to be used for feature selection. Default: genus',
                        type=str)
cal_parser.add_argument('-nk', '--min_k',
                        default=1,
                        help='Min value of K. Default: 1',
                        type=int)
cal_parser.add_argument('-nn', '--min_n',
                        default=5,
                        help='Min value of n. Default: 5',
                        type=int)
cal_parser.add_argument('--criterion',
                        help='Criterion for neighbour selection. Default: orbit',
                        type=str,
                        default='orbit')
prp_parser.add_argument('--threads',
                        help='Threads to use when performing the calibration. Default: 1',
                        type=int,
                        default=1)

# classification arguments
cls_parser.set_defaults(mode='classify')
cls_parser.add_argument('-ws', '--w_start',
                        help='Start coordinate for the classification window',
                        type=int)
cls_parser.add_argument('-we', '--w_end',
                        help='End coordinate for the classification window',
                        type=int)
cls_parser.add_argument('-n',
                        help='Number of informative sites to be used in the classification',
                        type=int)
cls_parser.add_argument('-k',
                        help='Number of neighbours to include in the classification',
                        type=int)
cls_parser.add_argument('-rk', '--rank',
                        default='genus',
                        help='Taxonomic rank to be used for feature selection. Default: genus',
                        type=str)
cls_parser.add_argument('-rt', '--row_thresh',
                        default=0.2,
                        help='Empty row threshold. Default: 0.2',
                        type=float)
cls_parser.add_argument('-ct', '--col_thresh',
                        default=0.2,
                        help='Empty column threshold. Default: 0.2',
                        type=float)
cls_parser.add_argument('-ms', '--min_seqs',
                        default=10,
                        help='Minimum number of sequences allowed per taxon. Default: 10',
                        type=int)
cls_parser.add_argument('--criterion',
                        help='Criterion for neighbour selection. Default: orbit',
                        type=str,
                        default='orbit')
cls_parser.add_argument('--method',
                        help='Weighting method. Default: unweighted',
                        type=str,
                        default='unweighted')

#%% classify
if __name__ == '__main__':
    args, unk = parser.parse_known_args()    
    # operation: preparation
    if args.mode == 'prep':
        preparation(args.out_dir,
                    args.transition,
                    args.transversion,
                    args.database,
                    args.query,
                    args.evalue,
                    args.dropoff,
                    args.min_height,
                    args.min_width,
                    args.min_overlap_width,
                    args.overwrite,
                    args.force_overwrite,
                    args.threads)
    else:
        # initialize classifier
        classifier = cls_main.Classifier()
        classifier.set_outdir(args.out_dir, overwrite=False)
        # operation: calibration
        if args.mode == 'calibrate':
            operation_calibrate(classifier,
                                min_overlap_width = args.min_overlap_width,
                                max_n = args.max_n,
                                step_n = args.step_n,
                                max_k = args.max_k,
                                step_k = args.step_k,
                                row_thresh = args.row_thresh,
                                col_thresh = args.col_thresh,
                                min_seqs = args.min_seqs,
                                rank = args.rank,
                                min_n = args.min_n,
                                min_k = args.min_k,
                                criterion = args.criterion,
                                threads = args.threads)
        
        # operation: classification
        if args.mode == 'classify':
            operation_classify(classifier,
                               w_start = args.w_start,
                               w_end = args.w_end,
                               n = args.n,
                               k = args.k,
                               rank = args.rank,
                               row_thresh = args.row_thresh,
                               col_thresh = args.col_thresh,
                               min_seqs = args.min_seqs,
                               criterion = args.criterion,
                               method = args.method)
