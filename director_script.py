#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:30:49 2022

@author: hernan
This script is used to access graboid from the command line
"""

import argparse
import sys
#%% parsers
print(sys.argv)
first_parser = argparse.ArgumentParser(add_help=False)
first_parser.add_argument('mode',
                          nargs='?',
                          default='help')
first_parser.add_argument('mode args',
                          nargs='*',
                          help="Mode specific arguments")
first_parser.add_argument('-h','--help',
                          action='store_true')

#%% help parser
help_parser = argparse.ArgumentParser(prog='Graboid',
                                      usage='%(prog)s MODE MODE_ARGS [-h]',
                                      description='Graboid is a program for the taxonomic identification of DNA amplicons of a specified marker. Have fun!',
                                      epilog='For a more detailed description of the function of each mode use "graboid MODE --help"')
help_parser.add_argument('mode',
                         help='''Specify graboid mode. Accepted values are:
                             database
                             mapping
                             calibrate
                             design
                             classify''',
                         nargs='?',
                         default='help')
help_parser.add_argument('mode args',
                    nargs='*',
                    help="Mode specific arguments")

#%% database parser
db_parser = argparse.ArgumentParser(prog='Graboid DATABASE',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid DATABASE downloads records from the specified taxon/marker pair from the NCBI and BOLD databases')
db_parser.add_argument('mode')
db_parser.add_argument('-o', '--out_dir',
                       help='Output directory for the retrieved files',
                       type=str)
db_parser.add_argument('-T', '--taxon',
                       help='Taxon to search for',
                       type=str)
db_parser.add_argument('-M', '--marker',
                       help='Marker sequence to search for',
                       type=str)
db_parser.add_argument('-F', '--fasta',
                       help='Pre-constructed fasta file')
db_parser.add_argument('--bold',
                       help='Search for records in the BOLD database',
                       action='store_true')
db_parser.add_argument('-r', '--ranks',
                       help='Set taxonomic ranks to download. Default: Phylum Class Order Family Genus Species',
                       nargs='*')
db_parser.add_argument('-c', '--chunksize',
                       default=500,
                       help='Number of records to download per pass. Default: 500',
                       type=int)
db_parser.add_argument('-m', '--max_attempts',
                       default=3,
                        help='Max number of attempts to download a chunk of records. Default: 3',
                        type=int)
db_parser.add_argument('--mv',
                       help='If a fasta file was provided, move it to the output directory',
                       action='store_true')
db_parser.add_argument('--keep_tmp',
                       help='Keep temporal files',
                       action='store_true')

#%% mapping parser
mp_parser = argparse.ArgumentParser(prog='Graboid MAPPING',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid MAPPING aligns the downloaded sequences to a specified reference sequence. Alignment is stored as a numeric matrix with an accession list')
mp_parser.add_argument('mode')
mp_parser.add_argument('-o', '--out_dir',
                       help='Output directory for the generated files',
                       type=str)
mp_parser.add_argument('-f', '--fasta',
                       help='Sequence file to be aligned',
                       type=str)
mp_parser.add_argument('-r', '--ref',
                       help='Reference sequence to be used as base of the alignment',
                       type=str)
mp_parser.add_argument('-db', '--db_dir',
                       help='OPTIONAL. BLAST database, alternative do reference sequence',
                       type=str)
mp_parser.add_argument('-on', '--out_name',
                       help='OPTIONAL. Name for the generated BLAST report and alignment matrix',
                       type=str)
mp_parser.add_argument('-rn', '--ref_name',
                       help='OPTIONAL. Name for the generated BLAST database',
                       type=str)
mp_parser.add_argument('-e', '--evalue',
                       help='E-value threshold for the BLAST matches. Default: 0.005',
                       type=float,
                       default=0.005)
mp_parser.add_argument('-c', '--clear',
                       help='Replace previous BLAST database file (if present)',
                       action='store_true')
mp_parser.add_argument('-t', '--threads',
                       help='Number of threads to be used in the BLAST alignment. Default: 1',
                       type=int,
                       default=1)

#%% calibrate parser
cb_parser = argparse.ArgumentParser(prog='Graboid CALIBRATE',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid CALIBRATE performs a grid search of the given ranges of K and n along a sliding window over the alignment matrix')
cb_parser.add_argument('mode')
cb_parser.add_argument('-o', '--out_dir',
                       help='Output directory for the generated files',
                       type=str)
cb_parser.add_argument('-mat', '--mat_file',
                       help='File containing the alignment matrix',
                       type=str)
cb_parser.add_argument('-acc', '--acc_file',
                       help='File containing the accession list for the alignment matrix',
                       type=str)
cb_parser.add_argument('-tax', '--tax_file',
                       help='File containing the taxonomy data for the alignment matrix',
                       type=str)
cb_parser.add_argument('-rt', '--row_thresh',
                       help='Empty row threshold',
                       type=float,
                       default=0.2)
cb_parser.add_argument('-ct', '--col_thresh',
                       help='Empty column threshold',
                       type=float,
                       default=0.2)
cb_parser.add_argument('-ms', '--min_seqs',
                       help='Minimum number of sequences allowed per taxon',
                       type=int,
                       default=10)
cb_parser.add_argument('-rk', '--rank',
                       help='Rank to be used for feature selection',
                       type=str,
                       default='genus')
cb_parser.add_argument('-ts', '--transition',
                       help='Transition penalization in the distance matrix',
                       type=float,
                       default=1)
cb_parser.add_argument('-tv', '--transversion',
                       help='Transversion penalization in the distance matrix',
                       type=float,
                       default=1)
cb_parser.add_argument('--id',
                       help='Use the identity matrix for distance calculation',
                       action='store_true')
cb_parser.add_argument('-wz', '--w_size',
                       help='Sliding window size',
                       type=int,
                       default=200)
cb_parser.add_argument('-wt', '--w_step',
                       help='Sliding window displacement',
                       type=int,
                       default=15)
cb_parser.add_argument('-mk', '--max_k',
                       help='Max value of K',
                       type=int,
                       default=15)
cb_parser.add_argument('-sk', '--step_k',
                       help='Rate of increase of K',
                       type=int,
                       default=2)
cb_parser.add_argument('-mn', '--max_n',
                       help='Max value of n',
                       type=int,
                       default=30)
cb_parser.add_argument('-sn', '--step_n',
                       help='Rate of increase of n',
                       type=int,
                       default=5)
cb_parser.add_argument('-nk', '--min_k',
                       help='Min value of K',
                       type=int,
                       default=1)
cb_parser.add_argument('-nn', '--min_n',
                       help='Min value of n',
                       type=int,
                       default=5)
cb_parser.add_argument('-f', '--out_file',
                       help='File name for the generated report (Will save into the given out_file)',
                       type=str)

#%% design parser
ds_parser = argparse.ArgumentParser(prog='Graboid DESIGN',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid DESIGN takes a given set of taxons or coordinates and generates suggest an experiment')
ds_parser.add_argument('mode')
ds_parser.add_argument('-i', '--in_dir')
ds_parser.add_argument('-t', '--taxon')
ds_parser.add_argument('-c', '-coords')

#%% classify parser
cl_parser = argparse.ArgumentParser(prog='Graboid CLASSIFY',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid CLASSIFY takes a fasta file and generates a classification report for each entry')
cl_parser.add_argument('mode')
cl_parser.add_argument('-i', '--in_dir')
cl_parser.add_argument('-f', '--fasta_file')
cl_parser.add_argument('-o', '--out_file')
cl_parser.add_argument('-k')
cl_parser.add_argument('-s')
cl_parser.add_argument('-m', '--mode')
cl_parser.add_argument('-sf', '--support_func')

#%%
parser_dict = {'DATABASE':db_parser,
               'MAPPING':mp_parser,
               'CALIBRATE':cb_parser,
               'DESIGN':ds_parser,
               'CLASSIFY':ds_parser,
               'HELP':help_parser}

base_args, unknown = first_parser.parse_known_args()

modes = ['DATABASE',
         'MAPPING',
         'CALIBRATE',
         'DESIGN',
         'CLASSIFY']
mode = base_args.mode.upper()

if mode not in modes:
    mode = 'HELP'
    sys.argv.append('--help')
parser = parser_dict[mode]
args = parser.parse_args()
print(args)

#TODO: PLOTTER mode. make plots from different modules. For now:
    # matrix2.plot_coverage_data(blast_file, evalue, figsize) # to plot coverage from a BLAST report
#%% execute
# database
def main(mode, args):
    from data_fetch.database_construction import director as db
    from data_fetch.dataset_construction import director as mp
    from calibration import calibrator as cb
    if mode == 'DATABASE':
        if (args.taxon is None or args.marker is None) and args.fasta is None:
            sys.argv.append('--help')
            taxmark = [tm for tm, arg in zip(('taxon', 'marker'), (args.taxon, args.marker)) if not arg is None]
            taxmark_msg = ' and '.join(taxmark)
            print(f'Missing value for {taxmark_msg}. Alternative, provide a fasta file')
            return
        args = parser.parse_args()
        
        db_out, db_tmp, db_warn = db.make_dirs(args.out_dir)
        db_director = db.Director(db_out, db_tmp, db_warn)
        
        # user specified ranks to use
        if not args.ranks is None:
            db_director.set_ranks(args.rank)
        
        if not args.fasta is None:
            # build db using fasta file (overrides taxon, mark)
            db_director.direct_fasta(args.fasta, args.chunksize, args.max_attempts, args.mv)
        if not (args.taxon is None or args.marker is None):
            #build db using tax & mark
            databases = ['NCBI']
            if args.bold:
                databases.append('BOLD')
            db_director.direct(args.taxon, args.marker, databases, args.chunksize, args.max_attempts)
        
        # clear temporal files
        if not args.keep_tmp:
            db_director.clear_tmp()
            
    elif mode == 'MAPPING':
        mp_out, mp_warn = mp.make_dirs(args.out_dir)
        mp_director = mp.Director(mp_out, mp_warn)
        
        if not args.db_dir is None:
            # BLAST database already exists, set directory
            mp_director.set_blastdb(args.db_dir)
        elif not args.ref is None:
            # Create BLAST database using given ref
            mp_director.build_blastdb(args.ref, args.ref_name, args.clear)
        else:
            print('Can\'t perform BLAST, provide a reference sequence or a BLAST database')
            return
        
        # Perform BLAST and build matrix
        mp_director.direct(args.fasta, args.out_name, args.evalue, args.threads)
    
    elif mode == 'CALIBRATE':
        cb_out, cb_warn = cb.make_dirs(args.out_dir)
        calibrator = cb.Calibrator(cb_out, cb_warn)
        
        # set parameters
        calibrator.set_row_thresh(args.row_thresh)
        calibrator.set_col_thresh(args.col_thresh)
        calibrator.set_min_seqs(args.min_seqs)
        calibrator.set_rank(args.rank)
        calibrator.set_cost_mat(args.transition, args.transversion, args.id)
        
        # load data
        if args.mat_file is None or args.acc_file is None or args.tax_file is None:
            print('One of the necesary files for calibration (mat_file, acc_file or tax_file) is missing')
            return
        if args.out_file is None:
            print('Missing argument for the output file (out_file)')
            return
        calibrator.set_database(args.mat_file, args.acc_file, args.tax_file)
        # calibration
        calibrator.grid_search(args.w_size,
                               args.w_step,
                               args.max_k,
                               args.step_k,
                               args.max_n,
                               args.step_n,
                               args.min_k,
                               args.min_n)
        calibrator.save_report(args.out_file)
