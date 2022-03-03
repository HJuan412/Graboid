#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:56:45 2022

@author: hernan
"""

#%% libraries
import argparse
import os
import sys
#%% functions
#%% main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'make-database', description='Create or update a sequence database for Graboid.')
    parser.add_argument('-o', '--out-dir', nargs='?', default='Database', help='Path to which the database will be stored')
    parser.add_argument('-t', '--taxons', nargs='*', default='*', help='Taxons to include in the database')
    parser.add_argument('-m', '--markers', nargs='*', default='*', help='Markers to include in the database')
    parser.add_argument('-d', '--databases', nargs='*', default='NCBI', choices = ['NCBI', 'BOLD'], help='Public databases to be surveyed (NCBI and/or BOLD)')
    args = parser.parse_args()
    
    for taxon in args.taxons:
        for marker in args.markers:
            db_path = f'{args.outdir}/{taxon}/{marker}'
            tmp_path = f'{args.outdir}/tmp/{taxon}/{marker}'
            warn_path = f'{args.outdir}/warnings/{taxon}/{marker}'
            if os.path.isdir(db_path):
                new = False
            else:
                new = True
                os.makedirs(db_path)
            
            os.makedirs(tmp_path, exist_ok=bool)
            os.makedirs(warn_path, exist_ok=bool)
            # director
    sys.exit()