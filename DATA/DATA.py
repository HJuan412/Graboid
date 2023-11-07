#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:07:41 2023

@author: hernan
Guide file used to locate and index the graboid databases
Usage:
    from DATA import DATA
generates:
    DATA.DATAPATH : contains the path to the DATA directory
    DATA.DBASES : contains a list of all training databases
    DATA.DBASE_LIST : contains a dictionary with the description of each database
    DATA.MAPS : contains previously generated maps, {db:{fasta_file:{map:map_file,
                                                                     acc:acc_file}}}
"""

import copy
import json
import os

DATAPATH = os.path.dirname(__file__)
DBASES = [db_dir for db_dir in os.listdir(DATAPATH) if (os.path.isdir(DATAPATH + '/' + db_dir) and db_dir != '__pycache__')]

def update_maps(maps):
    with open(DATAPATH + '/maps.json', 'w') as handle:
        json.dump(maps, handle)

def add_map(database, fasta, map_file, acc_file, evalue=None, dropoff=None, min_height=None, min_width=None):
    if not database in MAPS.keys():
        MAPS[database] = {}
    MAPS[database].update({fasta:{'map':map_file, 'acc':acc_file, 'evalue':evalue, 'dropoff':dropoff, 'min_height':min_height, 'min_width':min_width}})
    update_maps(MAPS)

def database_exists(database):
    return database in DBASES

def get_database(database):
    if database_exists(database):
        return f'{DATAPATH}/{database}'
    raise Exception(f'Database {database} not found among: {" ".join(DBASE_LIST.keys())}')

def add_calibration(database, fasta, report, meta):
    MAPS[database][fasta]['cal'] = report
    MAPS[database][fasta]['cal_meta'] = meta
    update_maps(MAPS)
    
# load dbase descriptions
DBASE_LIST = {}
for db in DBASES:
    try:
        with open(DATAPATH + '/' + db + '/desc.json', 'r') as handle:
            db_desc = json.load(handle)
    except FileNotFoundError:
        db_desc = ''
    DBASE_LIST[db] = db_desc
    
# load map list
try:
    with open(DATAPATH + '/maps.json', 'r') as handle:
        MAPS = json.load(handle)
except FileNotFoundError:
    MAPS = {}
# check that maps still exist
for db, fasta_dict in copy.deepcopy(MAPS).items():
    for fasta, map_files in fasta_dict.items():
        # if either map or acc file are missing, remove map from list
        if not os.path.isfile(map_files['map']) or not os.path.isfile(map_files['acc']):
            del MAPS[db][fasta]
    if len(MAPS[db]) == 0:
        del MAPS[db]
update_maps(MAPS)

# TODO: add function to remove database, should issue warning and ask confirmation