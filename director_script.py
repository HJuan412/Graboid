#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:30:49 2022

@author: hernan
This script is used to access graboid from the command line
"""

import argparse
import logging
import os
import pandas as pd
import pickle
import sys

#%% set logger
logger = logging.getLogger('Graboid')
logger.setLevel(logging.INFO)
# set formatters
fmtr0 = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fmtr1 = logging.Formatter('%(name)-30s %(levelname)-8s %(message)s')
# set console handler
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(fmtr1) # use shorter formatter, easier to read
logger.addHandler(console)
# debug logger
debug = logging.StreamHandler()
debug.setLevel(logging.DEBUG)
logger.addHandler(debug)
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
                         help='''Specify graboid mode. Accepted values are:\n
                             DATABASE\n
                             MAP\n
                             CALIBRATE\n
                             PLOT\n
                             REPORT\n
                             CLASSIFY''',
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
db_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
db_parser.add_argument('-T', '--taxon',
                       help='Taxon to search for',
                       type=str)
db_parser.add_argument('-M', '--marker',
                       help='Marker sequence to search for',
                       type=str)
db_parser.add_argument('-F', '--fasta',
                       help='Pre-constructed fasta file (use this in place of --taxon and --marker)')
db_parser.add_argument('--bold',
                       help='Include the BOLD database in the search',
                       action='store_true')
db_parser.add_argument('-r', '--ranks',
                       help='Set taxonomic ranks to include in the taxonomy table. Default: Phylum Class Order Family Genus Species',
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
                       action='store_true'),
db_parser.add_argument('--email',
                       help='Provide an email adress and an API key in order to use the NCBI Entrez utilities',
                       type=str)
db_parser.add_argument('--api_key',
                       help='API key associated to the provided email adress',
                       type=str)

#%% mapping parser
mp_parser = argparse.ArgumentParser(prog='Graboid MAPPING',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid MAPPING aligns the downloaded sequences to a specified reference sequence. Alignment is stored as a numeric matrix with an accession list')
mp_parser.add_argument('mode')
mp_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
mp_parser.add_argument('-B', '--base_seq',
                       default=None,
                       help='Marker sequence to be used as base of the alignment',
                       type=str)
mp_parser.add_argument('-db', '--db_dir',
                       default=None,
                       help='OPTIONAL. BLAST database, alternative to reference sequence',
                       type=str)
mp_parser.add_argument('-o', '--out_name',
                       default=None,
                       help='OPTIONAL. Name for the generated BLAST report and alignment matrix',
                       type=str)
mp_parser.add_argument('-bn', '--blast_name',
                       default=None,
                       help='OPTIONAL. Name for the generated BLAST database',
                       type=str)
mp_parser.add_argument('-e', '--evalue',
                       default=0.005,
                       help='E-value threshold for the BLAST matches. Default: 0.005',
                       type=float)
mp_parser.add_argument('-t', '--threads',
                       default=1,
                       help='Number of threads to be used in the BLAST alignment. Default: 1',
                       type=int)

#%% calibrate parser
cb_parser = argparse.ArgumentParser(prog='Graboid CALIBRATE',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid CALIBRATE performs a grid search of the given ranges of K and n along a sliding window over the alignment matrix')
cb_parser.add_argument('mode')
cb_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
cb_parser.add_argument('-rt', '--row_thresh',
                       default=0.2,
                       help='Empty row threshold. Default: 0.2',
                       type=float)
cb_parser.add_argument('-ct', '--col_thresh',
                       default=1.0,
                       help='Empty column threshold. Default: 0.2',
                       type=float)
cb_parser.add_argument('-ms', '--min_seqs',
                       default=10,
                       help='Minimum number of sequences allowed per taxon. Default: 10',
                       type=int)
cb_parser.add_argument('-rk', '--rank',
                       default='genus',
                       help='Rank to be used for feature selection. Default: genus',
                       type=str)
cb_parser.add_argument('-dm', '--dist_mat',
                       help='Distance matrix to be used for distance calculation. Valid codes: "id" and "s<int>v<int>"',
                       type=str)
cb_parser.add_argument('-wz', '--w_size',
                       default=200,
                       help='Sliding window size. Default: 200',
                       type=int)
cb_parser.add_argument('-ws', '--w_step',
                       default=15,
                       help='Sliding window displacement. Default: 15',
                       type=int)
cb_parser.add_argument('-mk', '--max_k',
                       default=15,
                       help='Max value of K. Default: 15',
                       type=int)
cb_parser.add_argument('-sk', '--step_k',
                       default=2,
                       help='Rate of increase of K. Default: 2',
                       type=int)
cb_parser.add_argument('-mn', '--max_n',
                       default=30,
                       help='Max value of n. Default: 30',
                       type=int)
cb_parser.add_argument('-sn', '--step_n',
                       default=5,
                       help='Rate of increase of n. Default: 5',
                       type=int)
cb_parser.add_argument('-nk', '--min_k',
                       default=1,
                       help='Min value of K. Default: 1',
                       type=int)
cb_parser.add_argument('-nn', '--min_n',
                       default=5,
                       help='Min value of n. Default: 5',
                       type=int)
cb_parser.add_argument('-o', '--out_file',
                       help='Optional. File name for the generated report (Will save into the given out_file)',
                       type=str)
cb_parser.add_argument('-t', '--threads',
                       default=1,
                       help='Number of threads to be used in the calibration. Default: 1',
                       type=int)

#%% plot parser
pt_parser = argparse.ArgumentParser(prog='Graboid PLOT',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid PLOT generates a heatmap from a calibration report')
pt_parser.add_argument('mode')
pt_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
pt_parser.add_argument('-rf', '--report_file',
                       help='Report file generated by a CALIBRATION step',
                       type=str)
pt_parser.add_argument('-rk', '--rank',
                       help='Generate plot using taxa of the given rank',
                       type=str,
                       default='genus')
pt_parser.add_argument('-ws', '--w_start',
                       help='Window start coordinate',
                       type=int)
pt_parser.add_argument('-we', '--w_end',
                       help='Window end coordinate',
                       type=int)
pt_parser.add_argument('-T', '--taxa',
                       help='Restrict parameter search to the given taxa',
                       type=str,
                       nargs='*')
pt_parser.add_argument('-s', '--show',
                       help='Display report in console',
                       action='store_true')
pt_parser.add_argument('-o', '--out_file',
                       help='Save report to the given file',
                       type=str)
#%% report parser
rp_parser = argparse.ArgumentParser(prog='Graboid REPORT',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid REPORT queries a specific portion of a given calibration report')
rp_parser.add_argument('mode')
rp_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
rp_parser.add_argument('-rf', '--report_file',
                       help='Report file generated by a CALIBRATION step',
                       type=str)
rp_parser.add_argument('-ws', '--w_start',
                       help='Window start coordinate',
                       type=int)
rp_parser.add_argument('-we', '--w_end',
                       help='Window end coordinate',
                       type=int)
rp_parser.add_argument('-T', '--taxa',
                       help='Restrict parameter search to the given taxa',
                       type=str,
                       nargs='*')
rp_parser.add_argument('-m', '--metric',
                       help='Metric used to select parameters. Default: "F1_score"',
                       type=str,
                       default='F1_score')
rp_parser.add_argument('-n', '--n_rows',
                       help='Number of combinations to show. Default: 3',
                       type=int,
                       default=3)
rp_parser.add_argument('-s', '--show',
                       help='Display report in console',
                       action='store_true')
rp_parser.add_argument('-o', '--out_file',
                       help='Save report to the given file',
                       type=str)

#%% classify parser
cl_parser = argparse.ArgumentParser(prog='Graboid CLASSIFY',
                                    usage='%(prog)s MODE_ARGS [-h]',
                                    description='Graboid CLASSIFY takes a fasta file and generates a classification report for each entry')
cl_parser.add_argument('mode')
cl_parser.add_argument('--work_dir',
                       help='Working directory for the generated files',
                       type=str)
cl_parser.add_argument('-q', '--query_file',
                       help='Query sequence files',
                       type=str)
cl_parser.add_argument('-dm', '--dist_mat',
                       help='Distance matrix to be utilized for distance calculation',
                       type=str)
cl_parser.add_argument('-ws', '--w_start',
                       help='Starting coordinates for the window of the alignment to use in classification',
                       type=int)
cl_parser.add_argument('-we', '--w_end',
                       help='End coordinates for the window of the alignment to use in classification',
                       type=int)
cl_parser.add_argument('--k',
                       help='K values to use in classification. Multiple values can be provided',
                       type=int,
                       nargs='*')
cl_parser.add_argument('--n',
                       help='Number of informative sites to use in the classification',
                       type=int)
cl_parser.add_argument('-md', '--cl_mode',
                       help='Classification criterion to be used. "m" : majority, "w" : wKNN, "d" : dwKNN',
                       type=str)
cl_parser.add_argument('-rk', '--rank',
                       help='Rank to be used for feature selection',
                       type=str,
                       default='genus')
cl_parser.add_argument('-o', '--out_file',
                       help='File name for the results file',
                       type=str)
cl_parser.add_argument('--keep_tmp',
                       help='Keep temporal files',
                       action='store_true')

#%%
parser_dict = {'DATABASE':db_parser,
               'MAP':mp_parser,
               'CALIBRATE':cb_parser,
               'PLOT':pt_parser,
               'REPORT':rp_parser,
               'CLASSIFY':cl_parser,
               'HELP':help_parser}

base_args, unknown = first_parser.parse_known_args()
mode = base_args.mode.upper()
if mode not in parser_dict:
    mode = 'HELP'
    sys.argv.append('--help')
parser = parser_dict[mode]
args = parser.parse_args()
logger.debug(f'Arguments {args}')

#%% execute
from database import director as db
from mapping import director as mp
from preprocess import feature_selection as fsele
from calibration import calibrator as cb
from calibration import reporter as rp
from calibration import plotter as pt
from classification import director as cl

def main(mode, args):
    # get file catalog dict
    catalog_path = f'{args.work_dir}/catalog.pickle' 
    try:
        with open(catalog_path, 'rb') as catalog_handle:
            # file catalog dictionary contains generated files in the working directory
            file_catalog = pickle.load(catalog_handle)
    except FileNotFoundError:
        file_catalog = {}
    
    # build directories
    res_dir = f'{args.work_dir}/results'
    cal_dir = f'{args.work_dir}/calibration'
    data_dir = f'{args.work_dir}/data'
    tmp_dir = f'{args.work_dir}/tmp'
    wrn_dir = f'{args.work_dir}/warnings'
    
    for dir_path in [res_dir, cal_dir, data_dir, tmp_dir, wrn_dir]:
        os.makedirs(dir_path, exist_ok=True)
    # set log handler
    log_handler = logging.FileHandler(f'{args.work_dir}/graboid.log')
    log_handler.setLevel(logging.WARNING)
    log_handler.setFormatter(fmtr0)
    logger.addHandler(log_handler)
    
    if mode == 'DATABASE':
        db_director = db.Director(data_dir, tmp_dir, wrn_dir)
        try:
            db.set_entrez(args.email, args.api_key)
        except AttributeError:
            print('Missing email adress and/or API key')
        # user specified ranks to use
        if not args.ranks is None:
            db_director.set_ranks(args.rank)
        # set databases
        databases = ['NCBI']
        if args.bold:
            databases.append('BOLD')
        
        if not args.fasta is None:
            # build db using fasta file (overrides taxon, mark)
            db_director.direct_fasta(args.fasta, args.chunksize, args.max_attempts, args.mv)
        elif not (args.taxon is None or args.marker is None):
            #build db using tax & mark
            db_director.direct(args.taxon, args.marker, databases, args.chunksize, args.max_attempts)
        else:
            print('No search parameters provided. Either set a path to a fasta file in the --fasta argument or a taxon and a marker in the --taxon and --marker arguments')
            return
        
        # clear temporal files
        if not args.keep_tmp:
            db_director.clear_tmp()
        
        # update catalog
        file_catalog['fasta'] = db_director.seq_file
        file_catalog['tax'] = db_director.tax_file
        file_catalog['guide'] = db_director.guide_file
        file_catalog['acclist'] = db_director.acc_file
        file_catalog['ranks'] = db_director.rank_file
        file_catalog['valid'] = db_director.valid_file
        with open(catalog_path, 'wb') as catalog_handle:
            pickle.dump(file_catalog, catalog_handle)
            
    if mode == 'MAP':
        mp_director = mp.Director(data_dir, wrn_dir)
        selector = fsele.Selector(data_dir)
        
        if not args.db_dir is None:
            # BLAST database already exists, set directory
            mp_director.set_blastdb(args.db_dir)
        elif not args.base_seq is None:
            logger.debug('Building blast database')
            # Create BLAST database using given ref
            mp_director.build_blastdb(args.base_seq, args.blast_name)
        else:
            print('Can\'t perform BLAST. Either provide a reference sequence file as --base_seq or a BLAST database as --db_dir')
            return
        # locate fasta file
        try:
            fasta_file = file_catalog['fasta']
        except KeyError:
            print(f'No fasta file found in directory {args.work_dir}')
            return
        # locate taxonomy file
        try:
            tax_file = file_catalog['tax']
        except KeyError:
            print(f'No taxonomy table file found in directory {args.work_dir}')
            return
        # Perform BLAST and build matrix
        logger.debug('Starting mapping')
        mp_director.direct(fasta_file, args.out_name, args.evalue, args.threads, keep=True)
        # build order matrix
        tax_tab = pd.read_csv(tax_file, index_col=0).loc[mp_director.accs]
        print('Cuantifying per-site information')
        selector.set_matrix(mp_director.matrix, mp_director.bounds, tax_tab)
        selector.build_tabs()
        selector.save_order_mat()
        # update catalog
        file_catalog['db_dir'] = mp_director.db_dir
        file_catalog['blast'] = mp_director.blast_report
        file_catalog['matrix'] = mp_director.mat_file
        file_catalog['accs'] = mp_director.acc_file
        file_catalog['order'] = selector.order_file
        with open(catalog_path, 'wb') as catalog_handle:
            pickle.dump(file_catalog, catalog_handle)
    
    if mode in ['CALIBRATE', 'CLASSIFY']:
        # retrieve files common to CALIBRATE and CLASSIFY
        try:
            mat_file = file_catalog['matrix']
        except KeyError:
            print(f'No matrix file found in {args.work_dir}')
            return
        try:
            acc_file = file_catalog['accs']
        except KeyError:
            print(f'No accession file found in {args.work_dir}')
            return
        try:
            tax_file = file_catalog['tax']
        except KeyError:
            print(f'No taxonomy file found in {args.work_dir}')
            return
        try:
            order_file = file_catalog['order']
        except KeyError:
            print(f'No order file found in {args.work_dir}')
            return
        
    if mode == 'CALIBRATE':
        calibrator = cb.Calibrator(cal_dir, wrn_dir)
        # set parameters
        calibrator.set_row_thresh(args.row_thresh)
        calibrator.set_col_thresh(args.col_thresh)
        calibrator.set_min_seqs(args.min_seqs)
        calibrator.set_rank(args.rank)
        calibrator.set_dist_mat(args.dist_mat)
        calibrator.set_database(mat_file, acc_file, tax_file, order_file)
        
        ready, missing = calibrator.check_ready()
        if not ready:
            print(f"Calibrator couldn't begin grid search, missing attributes: {', '.join(missing)}")
            return
        # calibration
        calibrator.grid_search(args.w_size,
                               args.w_step,
                               args.max_k,
                               args.step_k,
                               args.max_n,
                               args.step_n,
                               args.min_k,
                               args.min_n,
                               args.out_file,
                               args.threads,
                               True) # TODO: this true is to keep the generated classification file, remember to set as false
        calibrator.save_report(args.out_file)
    
    if mode == 'PLOT':
        plotter = pt.Director(cal_dir)
        plotter.set_data(args.report_file)
        plotter.zoom_report(args.rank, args.w_start, args.w_end, args.metric, args.taxa)
        plotter.plot_report(args.show, args.out_file)
        
    if mode in ['REPORT', 'CLASSIFY']:
        # retrieve files common to REPORT and CLASSIFY
        try:
            guide_file = file_catalog['guide']
        except KeyError:
            print(f'No guide file found in {args.work_dir}')
            return
        
    if mode == 'REPORT':
        reporter = rp.Reporter(cal_dir)
        reporter.set_guide(guide_file)
        reporter.load_report(args.report_file)
        reporter.report(args.w_start, args.w_end, args.taxa, args.metric, args.n_rows, args.show)
        if not args.out_file is None:
            reporter.save_report(args.out_file)
    
    if mode == 'CLASSIFY':
        classifier = cl.Director(res_dir, tmp_dir, wrn_dir)
        # set necessary data
        try:
            db_dir = file_catalog['db_dir']
        except KeyError:
            print(f'No db directory found in {args.work_dir}')
            return

        classifier.set_ref_data(mat_file, acc_file, tax_file)
        classifier.set_taxguide(guide_file)
        classifier.set_order(order_file)
        classifier.set_db(db_dir)
        classifier.set_query(args.fasta_file, args.query_name, args.threads)
        classifier.set_dist_mat(args.dist_mat)
        classifier.classify(args.w_start, args.w_end, args.k, args.n, args.cl_mode, args.site_rank, args.out_file)

if __name__ == '__main__':
    main(mode, args)