#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

@author: hernan
"""

#%% libraries
import gb_classify as cl
import gb_preprocess as pp
import numpy as np
import pandas as pd

#%% variables
mat_dir = 'Dataset/12_11_2021-23_15_53/Matrices/Nematoda/18S'
tax_tab = 'Databases/12_11_2021-23_15_53/Taxonomy_files/Nematoda_18S_tax.tsv'
acc2tax_tab = 'Databases/13_10_2021-20_15_58/Taxonomy_files/Nematoda_18S_acc2taxid.tsv'

#%% functions
def get_metrics(confusion):
    taxons = confusion.index.tolist()
    metrics = pd.DataFrame(index = confusion.index, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    
    for tax in taxons:
        tp = confusion.loc[tax, tax]
        tn = confusion.loc[confusion.index != tax, confusion.columns != tax].to_numpy().sum()
        fp = confusion.loc[confusion.index != tax, tax].to_numpy().sum()
        fn = confusion.loc[tax, confusion.columns != tax].to_numpy().sum()

        acc = (tp + tn) / (tp + tn + fp + fn)
        prc = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prc * rec)/(prc + rec)
        
        metrics.at[tax, 'Accuracy'] = acc
        metrics.at[tax, 'Precision'] = prc
        metrics.at[tax, 'Recall'] = rec
        metrics.at[tax, 'F1'] = f1
    
    metrics.fillna(0, inplace = True)
    return metrics

def get_report_filename(out_tab, mat_path):
    split_file = mat_path.split('/')[-1].split('.mat')[0].split('_')
    filename = f'{split_file[0]}_{split_file[2]}_{split_file[3]}.csv'
    return f'{out_tab}/{filename}'

#%% classes
class Calibrator():
    def __init__(self, mat_dir, tax_tab, out_dir):
        self.mat_dir = mat_dir
        self.tax_tab = tax_tab
        self.out_dir = out_dir
        self.mat_browser = pp.MatrixLoader(mat_dir)
        
    def calibrate(self, k_range, p_range, ds_mode = 'jk', folds = 10, dist_mode = 'id', classif_mode = 'vote'):
        # load matrixes one by one
        for mat_idx in self.mat_browser.mat_tab.index:
            mat_path = self.mat_browser.get_matrix_path(mat_idx)
            print(mat_path)
            preproc = pp.PreProcessor(mat_path, self.tax_tab)
            
            window_report = pd.DataFrame(columns = ['P', 'K', 'Accuracy', 'Precision', 'Recall', 'F1'])
            for p in p_range:
                print(f'Using {p} bases')
                # select p most informative bases and create classifier
                preproc.select_columns(p)
                for k in k_range:
                    print(f'\tUsing {k} neighbors')
                    
                    if ds_mode == 'jk':
                        ds_generator = preproc.get_jk_datasets()
                    elif ds_mode == 'kf':
                        ds_generator = preproc.get_kf_datasets(folds)

                    uniq_taxes = np.unique(preproc.tax_codes)
                    confusion = pd.DataFrame(data = 0, index = uniq_taxes, columns = uniq_taxes, dtype = int)

                    n_ds = len(preproc.matrix)
                    ten_percent = max(int(n_ds / 10), 1)
                    progress = 0
                    # dataset tuples contain (accessions, tax_codes, matrix)
                    # classifier constructor takes matrix and tax_codes
                    for idx, datasets in enumerate(ds_generator):
                        train_ds, test_ds = datasets
                        if idx % ten_percent == 0:
                            print(f'{progress * 10}% done')
                            progress += 1
                        classifier = cl.Classifier(train_ds[2], train_ds[1])
                        classifier.set_query(test_ds[2])
                        # calculate distances
                        if dist_mode == 'id':
                            classifier.dist_by_id()
                        elif dist_mode == 'cost':
                            classifier.dist_by_cost()
                        # classify instances
                        if classif_mode == 'vote':
                            classifier.vote_classify(k)
                            predictions = classifier.classif['Code'].values
                        elif classif_mode == 'weight':
                            classifier.weight_classify(k)
                            predictions = classifier.weighted_classif['Code'].values # this won't work
                        
                        # for true_val, pred_val in zip(test_ds[1], predictions):
                        #     confusion[true_val, pred_val] += 1
                        confusion.at[test_ds[1], predictions[0]] += 1
                    
                    metrics = get_metrics(confusion)

                    metrics['P'] = p
                    metrics['K'] = k
                    
                    window_report = pd.concat((window_report, metrics))
            
            rep_filename = get_report_filename(self.out_dir, mat_path)
            window_report.to_csv(rep_filename)
#%%
mat_browser = pp.MatrixLoader(mat_dir)
mat_path = mat_browser.get_matrix_path(17)
garrus = Calibrator(mat_dir, tax_tab, 'calib_test')
con = garrus.calibrate(np.arange(3, 21, 2), np.arange(5, 26, 5))
