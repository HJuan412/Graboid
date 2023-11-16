#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:57:24 2023

@author: hernan
Classification director. Handles custom calibration and classification of query files
"""

#%% libraries
import os
# graboid libraries
from classification import cls_main
from parsers import cls_parser as parser

#%%
    # cls_plots.plot_ref_v_qry(classifier.ref_coverage,
    #                          classifier.ref_mesas,
    #                          classifier.query_coverage,
    #                          classifier.query_mesas,
    #                          classifier.overlaps,
    #                          ref_title = classifier.db,
    #                          qry_title = classifier.query_file,
    #                          out_file = classifier.query_dir + '/coverage.png')

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
        if args.list:
            classifier.list_calibrations()
            return
        # TODO: fix param selection: each calibration run performed on a single window has its own calibration directory in the work_dir, parameters should then be: calibration_dir, metric, taxa (and show)
        classifier.select_parameters(calibration_dir = args.calibration_dir,
                                     metric = args.metric,
                                     taxa = args.taxa)
    elif args.operation == 'run':
        classifier.classify(w_start = args.w_start,
                            w_end = args.w_end,
                            n = args.n,
                            k = args.k,
                            rank = args.rank,
                            row_thresh = args.row_thresh,
                            col_thresh = args.col_thresh,
                            min_seqs = args.min_seqs,
                            criterion = args.criterion,
                            method = args.method,
                            save_dir = args.save_dir)
#%% classify
if __name__ == '__main__':
    args, unk = parser.parse_known_args()
    main(args)
