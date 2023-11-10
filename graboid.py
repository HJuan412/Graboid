#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:26:03 2023

@author: hernan

Main graboid script
"""

import mkdatabase
import calibrate
import classify

from parsers import parser

if __name__ == '__main__':
    args, unk = parser.parse_known_args()
    print(args)
    if args.task in ('gdb', 'database'):
        mkdatabase.main(args)
    elif args.task in ('cal', 'calibrate', 'calibration'):
        calibrate.main(database = args.database,
                       out_dir = args.out_dir,
                       row_thresh = args.row_thresh,
                       col_thresh = args.col_thresh,
                       min_seqs = args.min_seqs,
                       rank = args.rank,
                       mat_code = args.dist_mat,
                       w_size = args.w_size,
                       w_step = args.w_step,
                       max_k = args.max_k,
                       step_k = args.min_k,
                       max_n = args.max_n,
                       step_n = args.step_n,
                       min_k = args.min_k,
                       min_n = args.min_n,
                       criterion = args.criterion,
                       threads = args.threads,
                       keep = args.keep)
    elif args.task in ('cls', 'classify', 'classification'):
        if args.operation == 'preparation':
            pass
        elif args.operation == 'calibration':
            pass
        elif args.operation == 'params':
            pass
        elif args.operation == 'classify':
            pass
        