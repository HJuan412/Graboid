#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:45:20 2024

@author: hernan

Build Graboid database from a provided fasta file.
A taxonomy table must be provided 
"""

import os
import shutil
import subprocess

shell_path = os.path.dirname(__file__) + '/get_taxdmp.sh'

# copy or move fasta file to the graboid DATA directory
# get NCBI taxonomy code for taxa presented in taxonomy table
    # generate warning file if any taxa is not present in genbank

# output files:
    # fasta file
    # Lineage table
    # Taxonomy table
    # names table

def move_fasta(fasta_file, destination, mv=False):
    if mv:
        shutil.move(fasta_file, destination)
    else:
        shutil.copy(fasta_file, destination)

def get_taxdmp(out_dir):
    """Retrieve and format the NCBI taxdmp files and store them in out_file"""
    subprocess.run([shell_path, out_dir])

    