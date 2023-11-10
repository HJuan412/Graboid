#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:32:46 2023

@author: hernan

Contains parsers for all modules of graboid
"""

import argparse
#%% parsers
# main parser
parser = argparse.ArgumentParser(prog='graboid',
                                 description='Graboid is a program for the taxonomic identification of DNA amplicons of a specified marker. Have fun!',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# subparsers
subparsers = parser.add_subparsers(title='task', dest='task')

#%% database parsers
gdb_parser = subparsers.add_parser('gdb',
                                   aliases=['database'],
                                   description='Graboid DATABASE: Downloads records from the specified taxon/marker pair from the NCBI and BOLD databases',
                                   help='Graboid DATABASE')
# set subparsers
gdb_subparsers = gdb_parser.add_subparsers(title='mode', dest='mode')
gdb_mke_parser = gdb_subparsers.add_parser('make',
                                           description='Make a new graboid database from online repositories or a local fasta file',
                                           help='Make a graboid database')
gdb_lst_parser = gdb_subparsers.add_parser('list',
                                           description='List existing graboid databases',
                                           help='List existing graboid databases')
gdb_del_parser = gdb_subparsers.add_parser('delete',
                                           description='Delete an existing graboid database (WARNING: This step is irreversible, in order to retrieve the database you will need to create it anew)',
                                           help='Delete a graboid database')
gdb_exp_parser = gdb_subparsers.add_parser('export',
                                           description='Copy the contents of an existing gaboid database into a specified directory',
                                           help='Export a graboid database')

# make parser
# make parser: positional arguments
gdb_mke_parser.add_argument('db',
                            help='Name for the generated database',
                            type=str)
gdb_mke_parser.add_argument('ref',
                            help='Reference sequence for the selected molecular marker. Must be a fasta file with one (1) sequence',
                            type=str)

# make parser: data source arguments
repo_group = gdb_mke_parser.add_argument_group('Repository arguments',
                                               'Retrieve data from online repositories. Use these arguments to set search parameters')
repo_group.add_argument('--taxon',
                        help='Taxon to search for',
                        type=str)
repo_group.add_argument('--marker',
                        help='Marker sequence to search for',
                        type=str)
repo_group.add_argument('--bold',
                        help='Include the BOLD database in the search',
                        action='store_true')
fasta_group = gdb_mke_parser.add_argument_group('Local file arguments',
                                                'Build the database from a local fasta file. If used, overwrites repository arguments')
fasta_group.add_argument('--fasta',
                         help='Pre-constructed fasta file',
                         type=str)

# make parser: download arguments
dl_group = gdb_mke_parser.add_argument_group('Download arguments',
                                             'Set data retrieval options')
dl_group.add_argument('--chunk',
                      help='Number of records to download per pass (default: 500)',
                      type=int,
                      default=500)
dl_group.add_argument('--attempts',
                      help='Max number of attempts to download a chunk of records (default: 3)',
                      type=int,
                      default=3)
dl_group.add_argument('--email',
                      help='Use this in conjunction with --apikey to enable parallel downloads from NCBI (must provide a valid NCBI API key)',
                      type=str)
dl_group.add_argument('--apikey',
                      help='Use this in conjunction with --email to enable parallel downloads from NCBI (must provide a valid NCBI API key)',
                      type=str)
dl_group.add_argument('-r', '--ranks',
                      help='Set taxonomic ranks to include in the taxonomy table (case insensitive) (default: Phylum Class Order Family Genus Species)',
                      nargs='*')

# make parser: alignment arguments
mke_aln_group = gdb_mke_parser.add_argument_group('Alignment arguments',
                                              'Configure reference sequence alignment')
mke_aln_group.add_argument('--evalue',
                        help='E-value threshold for the BLAST matches (default: 0.005)',
                        type=float,
                        default=0.005)
mke_aln_group.add_argument('--dropoff',
                        help='Percentage of mesa height drop to determine a border (default: 0.05)',
                        type=float,
                        default=0.05)
mke_aln_group.add_argument('--min_height',
                        help='Minimum sequence coverage to consider for a mesa candidate (percentage of maximum coverage) (default: 0.1)',
                        type=float,
                        default=0.1)
mke_aln_group.add_argument('--min_width',
                        help='Minimum width needed for a mesa candidate to register (default: 2)',
                        type=int,
                        default=2)

# make parser: optional arguments
gdb_mke_parser.add_argument('--desc',
                            help='Database description text',
                            type=str,
                            default='')
gdb_mke_parser.add_argument('--threads',
                            help='Threads to use when building the alignment (default: 1)',
                            type=int,
                            default=1)
gdb_mke_parser.add_argument('--keep',
                            help='Keep the temporal files',
                            action='store_true')
gdb_mke_parser.add_argument('--clear',
                            help='Overwrite existing database of the same name (if it exists)',
                            action='store_true')
    

# deletion parser
gdb_del_parser.add_argument('database',
                            help='Name of the database to be removed. Use * to delete all existing databases',
                            type=str)

# export parser
gdb_exp_parser.add_argument('database',
                            help='Name of the database to exort',
                            type=str)
gdb_exp_parser.add_argument('out_dir',
                            help='Output directory for the exported files',
                            type=str)

#%% calibration parser
cal_parser = subparsers.add_parser('cal',
                                   aliases=['calibrate', 'calibration'],
                                   description='Graboid CALIBRATE: Performs a grid search of the given ranges of K and n along a sliding window over the alignment matrix',
                                   help='Graboid CALIBRATE',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# calibration parser: positional arguments
cal_parser.add_argument('database',
                        help='Database to calibrate',
                        type=str)
cal_parser.add_argument('out_dir',
                        help='Output directory',
                        type=str)

# calibration parser: filtering arguments
cal_filter_group = cal_parser.add_argument_group('Filtering arguments',
                                                 'Arguments used to filter the training set')
cal_filter_group.add_argument('-rt', '--row_thresh',
                              default=0.2,
                              help='Empty row threshold',
                              type=float)
cal_filter_group.add_argument('-ct', '--col_thresh',
                              default=0.2,
                              help='Empty column threshold',
                              type=float)
cal_filter_group.add_argument('-ms', '--min_seqs',
                              default=10,
                              help='Minimum number of sequences allowed per taxon',
                              type=int)
cal_filter_group.add_argument('-rk', '--rank',
                              default='genus',
                              help='Rank to be used for feature selection',
                              type=str)

# calibration parser: sliding window arguments
win_group = cal_parser.add_argument_group('Sliding window arguments',
                                          'Arguments needed for using a sliding calibration window. Specify the size of the window and its rate of displacement (step)')
win_group.add_argument('--w_size',
                       default=200,
                       help='Sliding window size',
                       type=int)
win_group.add_argument('--w_step',
                       default=30,
                       help='Sliding window displacement',
                       type=int)

# calibration parser: specific window arguments
spec_group = cal_parser.add_argument_group('Specific window(s) arguments',
                                           'Select one or more regions of the alignment to perform the calibration on. Use in place of --w_size and --w_step. Each region is presented as a pair of values given in --w_start and --w_end')
spec_group.add_argument('--w_start',
                        nargs='*',
                        help='Start coordinates for one or more regions of the alignment',
                        type=int)
spec_group.add_argument('--w_end', 
                        nargs='*',
                        help='End coordinates for one or more regions of the alignment',
                        type=int)

# calibration parser: grid arguments
cal_grid_group = cal_parser.add_argument_group('Parameter grid arguments',
                                               'Configure the parameter space to be explored in the grid search')
cal_grid_group.add_argument('-mk', '--max_k',
                            default=15,
                            help='Max value of K',
                            type=int)
cal_grid_group.add_argument('-sk', '--step_k',
                            default=2,
                            help='Rate of increase of K',
                            type=int)
cal_grid_group.add_argument('-mn', '--max_n',
                            default=30,
                            help='Max value of n',
                            type=int)
cal_grid_group.add_argument('-sn', '--step_n',
                            default=5,
                            help='Rate of increase of n',
                            type=int)
cal_grid_group.add_argument('-nk', '--min_k',
                            default=1,
                            help='Min value of K',
                            type=int)
cal_grid_group.add_argument('-nn', '--min_n',
                            default=5,
                            help='Min value of n',
                            type=int)

# # calibration parser: classification arguments
cal_class_group = cal_parser.add_argument_group('Classification arguments',
                                                'Configure the behaviour of the classifier')
cal_class_group.add_argument('-s', '--transition',
                             default=1,
                             help='Cost for transition-type substitutions (A <-> T, C <-> G)',
                             type=float)
cal_class_group.add_argument('-v', '--transversion',
                             default=1,
                             help='Cost for transversion-type substitutions (A/T <-> C/G)',
                             type=float)
cal_class_group.add_argument('-c', '--criterion',
                             default = 'orbit',
                             help='Neighbour selection criterion. Orbit: Select all neighbours within K orbitals of the query sequence. Neighbour: Select neighbours from orbits up to the one containing the K nearest neighbour',
                             choices=['orbit', 'neighbour'])

# calibration parser: optional arguments
cal_parser.add_argument('-t', '--threads',
                        default=1,
                        help='Number of threads to be used in the calibration',
                        type=int)
cal_parser.add_argument('--keep',
                        action='store_true',
                        help='Keep the generated classifiaction files. WARNING: this will use aditional disk space.')

#%% classification parsers
cls_parser = subparsers.add_parser('cls',
                                   aliases=['classify', 'classification'],
                                   description='Graboid CLASSIFY: Performs taxonomic identification for a set of amplicon samples',
                                   help='Graboid CLASSIFY',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# classification common arguments
cls_common = argparse.ArgumentParser(add_help=False)
cls_common.add_argument('out_dir', help='Working directory', type=str)

# set subparsers
cls_subparsers = cls_parser.add_subparsers(title='operation', dest='operation')
cls_pre_parser = cls_subparsers.add_parser('pre',
                                           aliases=['prep', 'preparation'],
                                           parents = [cls_common],
                                           description='Set up working directory, select graboid database, load and map query file',
                                           help='Preparation, load query data, set reference database and build output directory',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cls_cal_parser = cls_subparsers.add_parser('cal',
                                           aliases=['calibrate', 'calibration'],
                                           parents = [cls_common],
                                           description='Perform a parameter calibration over the marker region covered by the query sequences',
                                           help='Calibration over query region',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cls_par_parser = cls_subparsers.add_parser('par',
                                           aliases=['params', 'parameters'],
                                           parents = [cls_common],
                                           description='Retrieve best parameter set determined by query calibration',
                                           help='Select best parameters for query data',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cls_run_parser = cls_subparsers.add_parser('run',
                                           parents = [cls_common],
                                           description='Classify query sequences',
                                           help='Perform classification',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# preparation parser
cls_pre_parser.add_argument('database',
                            help='Database to be used in the classification',
                            type=str)
cls_pre_parser.add_argument('query',
                            help='Query file in FASTA format',
                            type=str)

# preparation parser: classification arguments
cls_class_group = cls_pre_parser.add_argument_group('Classification arguments',
                                                    'Configure the behaviour of the classifier')
cls_class_group.add_argument('-s', '--transition',
                             default=1,
                             help='Cost for transition-type substitutions (A <-> T, C <-> G)',
                             type=float)
cls_class_group.add_argument('-v', '--transversion',
                             default=1,
                             help='Cost for transversion-type substitutions (A/T <-> C/G)',
                             type=float)
cls_class_group.add_argument('-c', '--criterion',
                             default = 'orbit',
                             help='Neighbour selection criterion. Orbit: Select all neighbours within K orbitals of the query sequence. Neighbour: Select neighbours from orbits up to the one containing the K nearest neighbour',
                             choices=['orbit', 'neighbour'])

# preparation parser: alignment arguments
# make parser: alignment arguments
cls_aln_group = cls_pre_parser.add_argument_group('Alignment arguments',
                                                  'Configure query sequence alignment')
cls_aln_group.add_argument('--evalue',
                           help='E-value threshold for the BLAST matches (default: 0.005)',
                           type=float,
                           default=0.005)
cls_aln_group.add_argument('--dropoff',
                           help='Percentage of mesa height drop to determine a border (default: 0.05)',
                           type=float,
                           default=0.05)
cls_aln_group.add_argument('--min_height',
                           help='Minimum sequence coverage to consider for a mesa candidate (percentage of maximum coverage) (default: 0.1)',
                           type=float,
                           default=0.1)
cls_aln_group.add_argument('--min_width',
                           help='Minimum width needed for a mesa candidate to register (default: 2)',
                           type=int,
                           default=2)
cls_aln_group.add_argument('--min_overlap_width',
                           default=10,
                           help='Minimum overlap width betwen reference and query mesas',
                           type=int)

# preparation parser: optional arguments
cls_pre_parser.add_argument('--threads',
                            help='Threads to use when building the alignment',
                            type=int,
                            default=1)

# query calibration parser
# query calibration parser: filtering arguments
cls_filter_group = cls_cal_parser.add_argument_group('Filtering arguments',
                                                     'Arguments used to filter the training set')
cls_filter_group.add_argument('-rt', '--row_thresh',
                              default=0.2,
                              help='Empty row threshold',
                              type=float)
cls_filter_group.add_argument('-ct', '--col_thresh',
                              default=0.2,
                              help='Empty column threshold',
                              type=float)
cls_filter_group.add_argument('-ms', '--min_seqs',
                              default=10,
                              help='Minimum number of sequences allowed per taxon',
                              type=int)
cls_filter_group.add_argument('-rk', '--rank',
                              default='genus',
                              help='Rank to be used for feature selection',
                              type=str)

# query calibration parser: grid arguments
cls_grid_group = cls_cal_parser.add_argument_group('Parameter grid arguments',
                                                   'Configure the parameter space to be explored in the grid search')
cls_grid_group.add_argument('-mk', '--max_k',
                            default=15,
                            help='Max value of K',
                            type=int)
cls_grid_group.add_argument('-sk', '--step_k',
                            default=2,
                            help='Rate of increase of K',
                            type=int)
cls_grid_group.add_argument('-mn', '--max_n',
                            default=30,
                            help='Max value of n',
                            type=int)
cls_grid_group.add_argument('-sn', '--step_n',
                            default=5,
                            help='Rate of increase of n',
                            type=int)
cls_grid_group.add_argument('-nk', '--min_k',
                            default=1,
                            help='Min value of K',
                            type=int)
cls_grid_group.add_argument('-nn', '--min_n',
                            default=5,
                            help='Min value of n',
                            type=int)

# query calibration parser: optional arguments
cls_cal_parser.add_argument('--threads',
                            help='Threads to use when performing the calibration',
                            type=int,
                            default=1)

# parameters parser
cls_par_parser.add_argument('window',
                            help='Calibration window index',
                            type=int)
cls_par_parser.add_argument('--metric',
                            help='Calibration metric used for parameter selection',
                            choices=['acc', 'prc', 'rec', 'f1', 'ce'],
                            default='f1')
cls_par_parser.add_argument('--cal_dir',
                            help='Calibration directory to be used in parameter selection. If none is provided the latest run will be used',
                            type=str)
cls_par_parser.add_argument('--taxa',
                            help='Select parameters that optimize selection for the given taxa',
                            type=str,
                            nargs='*')

# classification parser
# classification parser: filtering parameters
run_filter_group = cls_run_parser.add_argument_group('Filtering arguments',
                                                     'Arguments used to filter the query set')
run_filter_group.add_argument('-rk', '--rank',
                              default='genus',
                              help='Taxonomic rank to be used for feature selection',
                              type=str)
run_filter_group.add_argument('-rt', '--row_thresh',
                              default=0.1,
                              help='Maximum empty row threshold',
                              type=float)
run_filter_group.add_argument('-ct', '--col_thresh',
                              default=0.1,
                              help='Maximum empty column threshold',
                              type=float)
run_filter_group.add_argument('-ms', '--min_seqs',
                              default=10,
                              help='Minimum number of sequences allowed per taxon',
                              type=int)

# classification parser: classification arguments
run_classif_group = cls_run_parser.add_argument_group('Classification parameters')
run_classif_group.add_argument('-ws', '--w_start',
                               help='Start coordinate for the classification window',
                               type=int)
run_classif_group.add_argument('-we', '--w_end',
                               help='End coordinate for the classification window',
                               type=int)
run_classif_group.add_argument('-n',
                               help='Number of informative sites to be used in the classification',
                               type=int)
run_classif_group.add_argument('-k',
                               help='Number of neighbours to include in the classification',
                               type=int)
run_classif_group.add_argument('--method',
                               choices=['unweighted', 'wknn', 'dwknn'],
                               help='Weighting method',
                               type=str,
                               default='unweighted')

if __name__ == '__main__':
    args, unk = parser.parse_known_args()
    print(args)
    print(unk)