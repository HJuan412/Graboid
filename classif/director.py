#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:59:03 2022

@author: hernan
Director for the classification of sequences of unknown taxonomic origin
"""
#%%
from classif import classification
from data_fetch.dataset_construction import blast
from data_fetch.dataset_construction import matrix2
import numpy as np
import pandas as pd
#%%
# steps
# optional: hint paramters
# Select window
# classify
# report

class Director:
    def __init__(self, out_dir, tmp_dir):
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.mat_file = None
        self.acc_file = None
        self.tax_file = None
        self.report = None
        self.taxa = []
        self.windows = None
        self.query_blast = None
        self.query_map = None
        self.result = None
    
    def set_reference(self, mat_file, acc_file, tax_file):
        self.mat_file = mat_file
        self.acc_file = acc_file
        self.tax_file = tax_file
        
        self.ref_mat = np.load(mat_file)
        with open(acc_file, 'r') as acc_handle:
            self.acc_file = acc_handle.read().splitlines()
        self.tax_tab = pd.read_csv(tax_file, index_col = 0)
    
    def set_report(self, report_file):
        report = pd.read_csf(report_file)
        self.w_len = report['w_end'].iloc[0] - report['w_start'].iloc[1]
        self.w_step = report['w_start'].iloc[1] - report['w_start'].iloc[0]
        self.report = report
    
    def set_taxa(self, taxa):
        for tax in taxa:
            self.taxa.append(taxa)
    
    def get_windows(self, metric='F1_score'):
        windows = {}
        for qid, sub_map in self.query_map.groupby('qseqid'):
            l_bounds = np.floor(sub_map['sstart']/self.w_step) * self.w_step
            u_bounds = np.ceil(sub_map['length']/self.w_len) + l_bounds * self.w_step + self.w_len
            
            best_win = (0, 0, 0, 0)
            best_met = 0
            for lower, upper in zip(l_bounds, u_bounds):
                sub_report = self.report.loc[(self.report['w_start'] >= lower) & (self.report['w_end'] <= upper)]
                best_row = sub_report.loc[sub_report[metric].idxmax()]
                if best_row[metric] > best_met:
                    best_met = best_row[metric]
                    best_win = (best_row['w_start'],
                                best_row['w_end'],
                                best_row['K'],
                                best_row['n_sites'])
            windows[qid] = best_win
        self.windows = windows
    
    def hint_params(self, w_start, w_end, metric='F1_score'):
        if w_end - w_start > self.w_len * 1.5:
            print('Warning: The provided window length {w_end - w_start} is more than 1.5 times the window length used in the calibration {self.w_len}\n\
                  Parameter hints may not be reliable. Recommend performing a calibration step for the desired window')
        sub_report = self.report.loc[(self.report['w_start'] >= w_start) & (self.report['w_end'] <= w_end)]
        best_row = sub_report.loc[sub_report[metric].idxmax()]
        self.params = (best_row['K'], best_row['n_sites'])
    
    def map_query(self, fasta_file, db_dir, threads=1):
        blast_file = fasta_file.split('/')[-1].split('.')[0]
        blast_file = f'{self.tmp_dir}/{blast_file}.BLAST'
        blast.blast(fasta_file, db_dir, blast_file, threads)
        self.fasta_file = fasta_file
        self.query_blast = blast_file
        self.query_map = pd.read_csv(blast_file,
                                     sep = '\t',
                                     header = None,
                                     names='qseqid pident length qstart qend sstart send evalue'.split())
    
    def set_dist_mat(self):
        return
    
    def classify(self, w_start=None, w_end=None, k=None, n=None, metric='F1_score'):
        # infer windows if none were given
        if w_start is None or w_end is None:
            self.get_windows(metric)
        # hint parameters if none were given
        elif k is None or n is None:
            self.hint_params(w_start, w_end)
        
        # classify
        query_mat = matrix2.build_query_window(self.query_blast, self.fasta_file, w_start, w_end, self.windows)
        
        # get sites
        q = None
        data = None
        dist_mat = None
        classification.classify(q, k, data, self.tax_tab, dist_mat)
        return