#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:57:24 2023

@author: hernan
Classification director. Handles custom calibration and classification of query files
"""

#%% libraries
import os
import shutil


# graboid libraries
from classification import cls_main, cls_plots
from parsers import cls_parser as parser

#%%
### classification operations
# preparations
# query calibration
# parameter selection
# classification run

# initialize classifier, set working directory
# set database (if not already set, else skip this step): NOTE. For simplicity, database cannot be changed for a working directory
# set query, report overlapping regions
# perform custom calibration of overlapping reions
# classify query

def is_graboid_dir(path):
    # proposed directory must contain only the expected contents (or be empty)
    exp_contents = {'calibration', 'classification', 'query', 'meta.json', 'warnings', 'classification.log'}
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
            raise Exception('Proposed directory cannot be assertained as a graboid directory\n' + str(excp))
        
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
def main(args):
    if args.operation in ('pre', 'prep', 'preparation'):
        # check out_dir is available
        if os.path.isdir(args.out_dir):
            print(f'Error: Selected working directory {args.out_dir} already exists')
            return
        try:
            # set working directory, database and query file
            cls_main.Classifier2(work_dir = args.out_dir,
                                 database = args.database,
                                 query = args.query,
                                 qry_evalue = args.evalue,
                                 qry_droppoff = args.dropoff,
                                 qry_min_height = args.min_height,
                                 qry_min_width = args.min_width,
                                 transition = args.transition,
                                 transversion = args.transversion,
                                 threads = args.threads)
        except Exception as excp:
            print(excp)
        return
    
    classifier = cls_main.Classifier2(work_dir = args.out_dir)
    if args.operation in ('cal', 'calibrate', 'calibration'):
        classifier.custom_calibrate(max_n = args.max_n,
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
                                    threads = args.threads,
                                    w_starts = args.w_start,
                                    w_ends = args.w_end)
    elif args.operation in ('par', 'params', 'parameters'):
        # TODO: fix param selection: each calibration run performed on a single window has its own calibration directory in the work_dir, parameters should then be: calibration_dir, metric, taxa (and show)
        classifier.select_parameters(calibration_dir, w_idx, w_start, w_end, metric, show, taxa)
    elif args.operation == 'run':
        classifier.classify(w_start = args.w_start,
                            w_end = args.w_end,
                            n = args.n,
                            k = args.k,
                            rank = args.rank,
                            row_thresh = args.row_thresh,
                            col_thresh = args.col_thresh,
                            min_seqs = args.min_seqs)
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
        
        if args.mode == 'params':
            metric = args.metric[0].upper()
            taxa = args.taxa
            if taxa is None:
                taxa = []
            classifier.select_parameters(args.cal_dir, args.window, None, None, metric, True, *taxa)
        
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
