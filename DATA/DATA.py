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
    DATA.DBASE_INFO : dictionary containing metadata of all existing databases
"""

import json
import os

DATAPATH = os.path.dirname(__file__)
DBASES = [db_dir for db_dir in os.listdir(DATAPATH) if (os.path.isdir(DATAPATH + '/' + db_dir) and db_dir != '__pycache__')]
DBASE_INFO = {}
for dbase in DBASES:
    with open(f'{DATAPATH}/{dbase}/meta.json', 'r') as handle:
        DBASE_INFO[dbase] = json.load(handle)

def database_exists(database):
    return database in DBASES

def get_database(database):
    if database_exists(database):
        return f'{DATAPATH}/{database}'
    raise Exception(f'Database {database} not found among: {" ".join(DBASES)}')