#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:04:30 2021

@author: hernan
Results visualizer
"""
import sys
sys.path.append('./Taxonomy')
import matplotlib.pyplot as plt
import numpy as np
import tax_reconstructor as trc
confusion = ''
total_true = ''
taxons = ''
#%%
def make_confusion(predicted_vals, true_vals, taxons, normalized = False):
    n_tax = len(taxons)
    confusion = np.zeros((n_tax, n_tax)) # index 0 = true values, index 1 = predicted values
    
    for idx0, tax0 in enumerate(taxons):
        actual_idx = np.where(true_vals == tax0)[0] # get actual instances of tax0
        predicted = predicted_vals[actual_idx] # get classifications assigned to instances of tax0
        for idx1, tax1 in enumerate(taxons):
            n_predicted = len(np.where(predicted == tax1)[0]) # get number of instances classified as tax1
            confusion[idx0, idx1] = n_predicted
    
    if normalized:
        total_true = confusion.sum(1) # get sum of true values for each category
        total_true = total_true.reshape(1,-1).T # do this to modify the matrix
        
        confusion = confusion / total_true
    return confusion

def get_metrics(confusion):
    n_taxons = confusion.shape[0]
    total_inds = confusion.sum()

    accuracy = np.zeros(n_taxons)
    precision = np.zeros(n_taxons)
    recall = np.zeros(n_taxons)

    for idx in range(n_taxons):
        true_pos = confusion[idx, idx]
        true_neg = np.delete(np.delete(confusion, idx, 0), idx, 1).sum()
        false_pos = np.delete(confusion[:,idx], idx).sum()
        false_neg = np.delete(confusion[idx], idx).sum()
        
        accuracy[idx] = (true_pos + true_neg) / total_inds
        precision[idx] = true_pos / (true_pos + false_pos)
        recall[idx] = true_pos / (true_pos + false_neg)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return np.array([accuracy, precision, recall, f1])

def graph_confusion(classifications, labels, taxons, names_tab, figsize = (12, 12)):
    # classifications: list returned by jacknife classify
    # labels: selected_taxonomy[rank].to_numpy()
    # taxons: handler.selected_taxons
    # names_tab: loaded from genbank_taxonmy names.tsv

    # prepare data
    confusion = make_confusion(np.array(classifications), labels, taxons)
    confusion_normal = make_confusion(np.array(classifications), labels, taxons, normalized = True)
    sci_tab = trc.retrieve_names(names_tab, taxons)

    tax_names = [sci_tab.loc[tax] for tax in taxons]
    total_true = confusion.sum(1).astype(int)
    total_predicted = confusion.sum(0).astype(int)    

    # initialize heatmap
    fig, ax = plt.subplots(figsize = figsize)
    ax.imshow(confusion_normal)
    
    ax.set_xticks(np.arange(len(taxons)))
    ax.set_yticks(np.arange(len(taxons)))
    ax.set_xticklabels([f'{tax} ({n_pred})' for tax, n_pred in zip(tax_names, total_predicted)], rotation = 60, ha = 'right')
    ax.set_yticklabels([f'{tax} ({n_true})' for tax, n_true in zip(tax_names, total_true)])
    
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('True values')
    ax.set_title('Confusion matrix')
    for idx0, x in enumerate(confusion):
        for idx1, y in enumerate(x):
            ax.text(idx0, idx1, int(confusion[idx1, idx0]), ha ='center', va = 'center')