#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:57:24 2023

@author: hernan
Classification director. Handles custom calibration and classification of query files
"""

#%% libraries
import argparse
# graboid libraries
from classification import cls_main
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

def preparation(classifier, database=None, query=None, qry_evalue=0.005, qry_dropoff=0.05, qry_min_height=0.1, qry_min_width=2, threads=1):
    # preparation steps, set database and query
    if not database is None:
        classifier.set_database(database)
    if not query is None:
        # this step will raise an exception if database isnt already set
        classifier.set_query(query,
                             evalue = qry_evalue,
                             dropoff = qry_dropoff,
                             min_height = qry_min_height,
                             min_width = qry_min_width,
                             threads = threads)

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

def operation_classify(classifier, w_start, w_end, n, k, rank, row_thresh, col_thresh, min_seqs, mat_code, criterion, method):
    classifier.classify(w_start = w_start,
                        w_end = w_end,
                        n = n,
                        k = k,
                        rank = rank,
                        row_thresh = row_thresh,
                        col_thresh = col_thresh,
                        min_seqs = min_seqs,
                        mat_code = mat_code,
                        criterion = criterion,
                        method = method)

#%%
parser = argparse.ArgumentParser(prog='Graboid CLASSIFY',
                                 usage='%(prog)s MODE_ARGS [-h]',
                                 description='Graboid CLASSIFY loads a query fasta file, checks overlapps to a reference sequence, and classifies')
parser.add_argument('out_dir',
                    help='Working directory for the classification run',
                    type=str)
parser.add_argument('--overwrite',
                    help='Overwrite provided working directory (if present)',
                    action='store_true')
parser.add_argument('--calibrate',
                    help='Perform a custom calibration operation',
                    action='store_true')
parser.add_argument('--classify',
                    help='Classify the provided query',
                    action='store_true')
# preparation arguments
parser.add_argument('-db', '--database',
                    help='Database to be used in the classification',
                    type=str)
parser.add_argument('-q', '--query',
                    help='Query file in FASTA format',
                    type=str)
parser.add_argument('--evalue',
                    default=0.005,
                    help='E-value threshold for the BLAST matches when mapping the query file. Default: 0.005',
                    type=float)
parser.add_argument('--dropoff',
                    help='Percentage of mesa height drop to determine a border. Default: 0.05',
                    type=float,
                    default=0.05)
parser.add_argument('--min_height',
                    help='Minimum sequence coverage to consider for a mesa candidate (percentage of maximum coverage). Default: 0.1',
                    type=float,
                    default=0.1)
parser.add_argument('--min_width',
                    help='Minimum width needed for a mesa candidate to register. Default: 2',
                    type=int,
                    default=2)

# common arguments
parser.add_argument('-dm', '--dist_mat',
                    default='id',
                    help='Distance matrix to be used for distance calculation. Valid codes: "id" and "s<int>v<int>". Default: id',
                    type=str)
parser.add_argument('-rk', '--rank',
                    default='genus',
                    help='Rank to be used for feature selection. Default: genus',
                    type=str)
parser.add_argument('-rt', '--row_thresh',
                    default=0.2,
                    help='Empty row threshold. Default: 0.2',
                    type=float)
parser.add_argument('-ct', '--col_thresh',
                    default=0.2,
                    help='Empty column threshold. Default: 0.2',
                    type=float)
parser.add_argument('-ms', '--min_seqs',
                    default=10,
                    help='Minimum number of sequences allowed per taxon. Default: 10',
                    type=int)
parser.add_argument('--threads',
                    help='Threads to use when building the alignment. Default: 1',
                    type=int,
                    default=1)
parser.add_argument('--criterion',
                    help='Criterion for neighbour selection. Default: orbit',
                    type=str,
                    default='orbit')

# calibration arguments
parser.add_argument('-ow', '--min_overlap_width',
                    default=10,
                    help='Minimum overlap width betwen reference and query mesas. Default: 10',
                    type=int)
parser.add_argument('-mn', '--max_n',
                    default=30,
                    help='Max value of n. Default: 30',
                    type=int)
parser.add_argument('-sn', '--step_n',
                    default=5,
                    help='Rate of increase of n. Default: 5',
                    type=int)
parser.add_argument('-mk', '--max_k',
                    default=15,
                    help='Max value of K. Default: 15',
                    type=int)
parser.add_argument('-sk', '--step_k',
                    default=2,
                    help='Rate of increase of K. Default: 2',
                    type=int)
parser.add_argument('-nk', '--min_k',
                    default=1,
                    help='Min value of K. Default: 1',
                    type=int)
parser.add_argument('-nn', '--min_n',
                    default=5,
                    help='Min value of n. Default: 5',
                    type=int)

# classification arguments
parser.add_argument('-ws', '--w_start',
                    help='Start coordinate for the classification window',
                    type=int)
parser.add_argument('-we', '--w_end',
                    help='End coordinate for the classification window',
                    type=int)
parser.add_argument('-n',
                    help='Number of informative sites to be used in the classification',
                    type=int)
parser.add_argument('-k',
                    help='Number of neighbours to include in the classification',
                    type=int)
parser.add_argument('--method',
                    help='Weighting method. Default: unweighted',
                    type=str,
                    default='unweighted')

#%% classify
if __name__ == '__main__':
    args = parser.parse_args()
    # initialize classifier
    classifier = cls_main.Classifier(args.out_dir, args.overwrite)
    
    # preparation
    preparation(classifier,
                args.database,
                args.query,
                qry_evalue = args.evalue,
                qry_dropoff = args.dropoff,
                qry_min_height = args.min_height,
                qry_min_width = args.min_width,
                threads = args.threads)
    
    # operation: calibration
    if args.calibrate:
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
    if args.classify:
        operation_classify(classifier,
                           w_start = args.w_start,
                           w_end = args.w_end,
                           n = args.n,
                           k = args.k,
                           rank = args.rank,
                           row_thresh = args.row_thresh,
                           col_thresh = args.col_thresh,
                           min_seqs = args.min_seqs,
                           mat_code = args.dist_mat,
                           criterion = args.criterion,
                           method = args.method)
