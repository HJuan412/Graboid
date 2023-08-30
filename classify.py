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
from classification import cls_main, cls_parameters, cls_plots

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
    # proposed directory must contain only the expected contents (or be empty)
    exp_contents = {'calibration', 'classification', 'query', 'meta.json', 'warnings'}
    dir_contents = set(os.listdir(path))
    if len(dir_contents) == 0:
        # directory exists but it's empty
        return
    not_found = exp_contents.difference(dir_contents)
    not_belong = dir_contents.difference(exp_contents)
    check = True
    warn_text = 'Inconsistencies found:\n'
    if len(not_found) > 0:
        warn_text += f'Missing expected elements: {not_found}.\n'
        check = False
    if len(not_belong) > 0:
        warn_text += f'Found extraneous elements: {not_belong}.'
        check = False
    
    if check:
        return
    raise Exception(warn_text)

def preparation(out_dir,
                database,
                query,
                transition,
                transversion,
                evalue=0.005,
                dropoff=0.05,
                min_height=0.1,
                min_width=2,
                min_overlap_width=10,
                overwrite=False,
                threads=1):
    
    # set up graboid database and query file
    ## Working directory
    # check overwrite (also check that potential directory is a graboid dir)
    if os.path.isdir(out_dir):
        try:
            is_graboid_dir(out_dir)
        except Exception as excp:
            raise 'Proposed directory cannot be assertained as a graboid directory\n' + excp
        
        if overwrite:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            raise Exception('Proposed working directory already exists. Overwrite with option --overwrite. WARNING: this will remove all data in the proposed directory.')
    
    classifier = cls_main.Classifier()
    classifier.set_outdir(out_dir, overwrite)
    try:
        classifier.set_database(database)
    except Exception as excp:
        # invalid database remove generated directories
        shutil.rmtree(out_dir)
        raise excp
    classifier.set_cost_matrix(transition, transversion)
    
    # Query
    classifier.set_query(query,
                         evalue,
                         dropoff,
                         min_height,
                         min_width,
                         threads)
    classifier.get_overlaps(min_overlap_width)
    cls_plots.plot_ref_v_qry(classifier.ref_coverage,
                             classifier.ref_mesas,
                             classifier.query_coverage,
                             classifier.query_mesas,
                             classifier.overlaps,
                             ref_title = classifier.db,
                             qry_title = classifier.query_file,
                             out_file = classifier.query_dir + '/coverage.png')

#%%
parser = argparse.ArgumentParser(prog='Graboid CLASSIFY',
                                 description='Common parameters for GRABOID CLASSIFY operations:')

subparsers = parser.add_subparsers(title='Operations')
prp_parser = subparsers.add_parser('prep',
                                   help='Prepare graboid working directory',
                                   description='Set up working directory, select graboid database, load and map query file',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cal_parser = subparsers.add_parser('calibrate',
                                   prog='graboid CLASSIFY [common options]',
                                   help='Perform a custom calibration operation',
                                   description='Parameters specific to the calibration operation',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
par_parser = subparsers.add_parser('params',
                                   prog='graboid CLASSIFY [common options]',
                                   help='Select classification parameters',
                                   description='Parameters specific to the parameter selection operation',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cls_parser = subparsers.add_parser('classify',
                                   prog='graboid CLASSIFY [common options]',
                                   help='Classify the provided query',
                                   description='Parameters specific to the classification operation',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# preparation arguments
prp_parser.set_defaults(mode='prep')
prp_parser.add_argument('out_dir',
                        help='Working directory',
                        type=str)
prp_parser.add_argument('database',
                        help='Database to be used in the classification',
                        type=str)
prp_parser.add_argument('query',
                        help='Query file in FASTA format',
                        type=str)
prp_parser.add_argument('-s', '--transition',
                        default=1,
                        help='Cost for transition-type substitutions (A <-> T, C <-> G)',
                        type=float)
prp_parser.add_argument('-v', '--transversion',
                        default=1,
                        help='Cost for transversion-type substitutions (A/T <-> C/G)',
                        type=float)
prp_parser.add_argument('--evalue',
                        default=0.005,
                        help='E-value threshold for the BLAST matches when mapping the query file',
                        type=float)
prp_parser.add_argument('--dropoff',
                        help='Maximum coverage drop threshold for mesa candidates (percentage of maximum coverage)',
                        type=float,
                        default=0.05)
prp_parser.add_argument('--min_height',
                        help='Minimum coverage threshold for mesa candidates (percentage of maximum coverage)',
                        type=float,
                        default=0.1)
prp_parser.add_argument('--min_width',
                        help='Minimum width threshold for mesa candidates',
                        type=int,
                        default=2)
prp_parser.add_argument('--min_overlap_width',
                        default=10,
                        help='Minimum overlap width betwen reference and query mesas',
                        type=int)
prp_parser.add_argument('--overwrite',
                        help='Overwrite provided working directory (if present). WARNING: All existing calibration and classification data will be lost',
                        action='store_true')
prp_parser.add_argument('--threads',
                        help='Threads to use when building the alignment',
                        type=int,
                        default=1)

# calibration arguments
cal_parser.set_defaults(mode='calibrate')
cal_parser.add_argument('out_dir',
                        help='Working directory',
                        type=str)
cal_parser.add_argument('-ow', '--min_overlap_width',
                        default=10,
                        help='Minimum overlap width betwen reference and query mesas',
                        type=int)
cal_parser.add_argument('-mn', '--max_n',
                        default=30,
                        help='Max value of n',
                        type=int)
cal_parser.add_argument('-sn', '--step_n',
                        default=5,
                        help='Rate of increase of n',
                        type=int)
cal_parser.add_argument('-mk', '--max_k',
                        default=15,
                        help='Max value of K',
                        type=int)
cal_parser.add_argument('-sk', '--step_k',
                        default=2,
                        help='Rate of increase of K',
                        type=int)
cal_parser.add_argument('-rt', '--row_thresh',
                        default=0.1,
                        help='Maximum empty row threshold',
                        type=float)
cal_parser.add_argument('-ct', '--col_thresh',
                        default=0.1,
                        help='Maximum empty column threshold',
                        type=float)
cal_parser.add_argument('-ms', '--min_seqs',
                        default=10,
                        help='Minimum number of sequences allowed per taxon',
                        type=int)
cal_parser.add_argument('-rk', '--rank',
                        default='genus',
                        help='Taxonomic rank to be used for feature selection',
                        type=str)
cal_parser.add_argument('-nk', '--min_k',
                        default=1,
                        help='Min value of K',
                        type=int)
cal_parser.add_argument('-nn', '--min_n',
                        default=5,
                        help='Min value of n',
                        type=int)
cal_parser.add_argument('--criterion',
                        choices=['orbit', 'neighbour'],
                        help='Criterion for neighbour sampling',
                        type=str,
                        default='orbit')
cal_parser.add_argument('--threads',
                        help='Threads to use when performing the calibration',
                        type=int,
                        default=1)

# param selection arguments
par_parser.set_defaults(mode='params')
par_parser.add_argument('out_dir',
                        help='Working directory',
                        type=str)
par_parser.add_argmuent('window',
                        help='Calibration window index',
                        type=int)
par_parser.add_argument('--metric',
                        help='Calibration metric used for parameter selection',
                        choices=['acc', 'prc', 'rec', 'f1', 'ce'],
                        default='f1')
par_parser.add_argument('--cal_dir',
                        help='Calibration directory to be used in parameter selection. If none is provided the latest run will be used',
                        type=str)
par_parser.add_argument('--taxa',
                        help='Select parameters that optimize selection for the given taxa',
                        type=str,
                        nargs='*')

# classification arguments
cls_parser.set_defaults(mode='classify')
cls_parser.add_argument('out_dir',
                        help='Working directory',
                        type=str)
# automatic parameter selection
cls_parser.add_argument('--auto',
                        help='Automatically select parameters',
                        action='store_true')
cls_parser.add_argument('-w', '--win_idx',
                        help='Window index, use alternatively to w_start and w_end',
                        type=int)
cls_parser.add_argument('--metric',
                        help='Calibration metric used for parameter selection',
                        choices=['acc', 'prc', 'rec', 'f1', 'ce'])
cls_parser.add_argument('--cal_dir',
                        help='Calibration directory to be used in parameter selection. If none is provided the latest run will be used',
                        type=str)
cls_parser.add_argument('--taxa',
                        help='Select parameters that optimize selection for the given taxa',
                        type=str,
                        nargs='*')
# classification parameters
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
                        help='Taxonomic rank to be used for feature selection',
                        type=str)
cls_parser.add_argument('-rt', '--row_thresh',
                        default=0.1,
                        help='Maximum empty row threshold',
                        type=float)
cls_parser.add_argument('-ct', '--col_thresh',
                        default=0.1,
                        help='Maximum empty column threshold',
                        type=float)
cls_parser.add_argument('-ms', '--min_seqs',
                        default=10,
                        help='Minimum number of sequences allowed per taxon',
                        type=int)
cls_parser.add_argument('--criterion',
                        choices=['orbit', 'neighbour'],
                        help='Criterion for neighbour sampling',
                        type=str,
                        default='orbit')
cls_parser.add_argument('--method',
                        choices=['unweighted', 'wknn', 'dwknn'],
                        help='Weighting method',
                        type=str,
                        default='unweighted')

#%% classify
if __name__ == '__main__':
    args, unk = parser.parse_known_args()
    # operation: preparation
    if args.mode == 'prep':
        try:
            preparation(args.out_dir,
                        args.database,
                        args.query,
                        args.transition,
                        args.transversion,
                        args.evalue,
                        args.dropoff,
                        args.min_height,
                        args.min_width,
                        args.min_overlap_width,
                        args.overwrite,
                        args.threads)
        except Exception as excp:
            print(excp)
            quit()
    else:
        # initialize classifier
        classifier = cls_main.Classifier()
        classifier.set_outdir(args.out_dir, overwrite=False)
        # operation: calibration
        if args.mode == 'calibrate':
            classifier.get_overlaps(args.min_overlap_width)
            classifier.custom_calibrate(args.max_n,
                                        args.step_n,
                                        args.max_k,
                                        args.step_k,
                                        args.row_thresh,
                                        args.col_thresh,
                                        args.min_seqs,
                                        args.rank,
                                        args.min_n,
                                        args.min_k,
                                        args.criterion,
                                        args.threads)
        
        if args.mode == 'param':
            metric = args.metric[0].upper()
            classifier.select_parameters(args.cal_dir, args.window, None, None, metric, True, *args.taxa)
        
        # operation: classification
        if args.mode == 'classify':
            start = args.w_start
            end = args.w_end
            n = args.n
            k = args.k
            mth = args.method
            if args.auto:
                # no previous selection or user has given selection parameters, perform a new selection
                if classifier.auto_start is None or (not args.win_idx is None and not args.metric is None):
                    classifier.select_parameters(args.cal_dir, args.win_idx, args.w_start, args.w_end, args.metric, False, args.taxa)
                start = classifier.auto_start
                end = classifier.auto_end
                n = classifier.auto_n
                k = classifier.auto_k
                mth = classifier.auto_mth
                
            classifier.classify(start,
                                end,
                                n,
                                k,
                                args.rank,
                                args.row_thresh,
                                args.col_thresh,
                                args.min_seqs,
                                args.criterion,
                                mth)
