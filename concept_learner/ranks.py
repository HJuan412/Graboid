#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:46:52 2024

@author: hernan
"""

#%% modules
import concurrent.futures
import numpy as np
import pandas as pd

from concept_learner.concept import Concept
#%% functions
def build_concept(taxon, rank_name, tax_idxs, matrix):
    """
    Wrapper function used to parallelize concept instantiation and learning

    Parameters
    ----------
    taxon : int
        TaxId of the concept taxon to be learned.
    rank_name : str
        Name of the concept taxon's rank.
    tax_idxs : array
        Indexes of the sequences belonging to the concept taxon.
    matrix : numpy array
        Encoded alignment matrix.

    Returns
    -------
    concept_tax : Concept.Concept
        Concept object with learned rules for the concept taxon.

    """
    concept_tax = Concept(taxon, rank_name)
    concept_tax.learn(matrix, tax_idxs)
    return concept_tax

#%% classes
class Rank:
    def __init__(self, name):
        self.name = name
        self.taxa = {}
    
    def __getitem__(self, taxon):
        return self.taxa[taxon]
    
    def get_type_counts(self):
        # Generate a dataframe counting the number of type site for each concept taxon
        
        type_summ = pd.DataFrame(0, index=self.taxa.keys(), columns = [0,1,2,3,4])
        for tax, concept in self.taxa.items():
            type_summ.loc[tax] = concept.types
        self.type_summ = type_summ
    
    def bin_type_counts(self, n_bins):
        # Generate bins for the type counts, used to generate the type distribution plots
        # Always define a separate bin for 0 sites
        # Return dataframe with index : bins, and columns : site types, values are the normalized count for each bin for each type
        # define bins
        
        bin_bounds = np.insert(np.linspace(0, self.n_sites+0.001, n_bins+1), 1, 1)
        bin_counts = {}
        for stype, col in self.type_summ.T.iterrows():
            
            # bin each type count, count&normalize observations of each bin among the rank taxa
            as_bins, bins = pd.cut(col, bin_bounds, retbins=True, duplicates='drop', right=False)
            bin_counts[stype] = as_bins.value_counts(normalize=True)
        bin_counts = pd.DataFrame(bin_counts)
        return bin_counts
    
    def list_solved(self):
        # List concept sequences, solved sequences and fraction of solved sequences per taxon
        seq_counts = pd.Series({tax:len(tax_concept.sequences) for tax, tax_concept in self.taxa.items()})
        solved_counts = pd.Series({tax:len(tax_concept.confirmed_seqs) for tax, tax_concept in self.taxa.items()})
        solved_norm = solved_counts / seq_counts
        
        self.seq_counts = seq_counts # sequences per taxon
        self.solved_counts = solved_counts # solved sequences per taxon
        self.solved_norm = solved_norm # fraction of solved sequences per taxon
        
    def get_confusion(self, rank_lineage):
        # get confusion matrix & normalized confusion matrix
        # Builds data frames with index : concept taxa, columns : out taxa
        # get raw confusion, out sequences compatible with each concept taxon
        confusion_raw = pd.DataFrame(False, index=self.taxa.keys(), columns=np.arange(self.n_seqs))
        for tax, concept in self.taxa.items():
            confusion_raw.loc[tax, concept.out_compatible_seqs] = True
        confusion_raw.columns = rank_lineage
        
        # get confusion matrix, group compatible out sequences by taxon
        confusion = pd.DataFrame(0, index=confusion_raw.index, columns=np.sort(rank_lineage.unique()))
        for tax in confusion.columns:
            confusion[tax] = confusion_raw[[tax]].sum(axis=1)
        
        # normalize confusion matrix, calculate the fraction of compatible outsider sequences belonging to each outsider taxon
        out_seq_counts = pd.Series({tax:len(tax_concept.out_sequences) for tax, tax_concept in self.taxa.items()})
        confusion_norm = (confusion.T / out_seq_counts).T
        
        self.out_seq_counts = out_seq_counts
        self.confusion = confusion
        self.confusion_norm = confusion_norm
            
    def learn(self, matrix, lineage_tab, lineage_flat, threads=1):
        # get all non-null taxa present in the rank
        rank_taxa = np.unique(lineage_tab[self.name])
        rank_taxa = rank_taxa[rank_taxa > 0]
        
        # record alignment dimensions
        self.n_seqs = matrix.shape[0]
        self.n_sites = matrix.shape[1]
        
        # learn concepts
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as Executor:
            future_concepts = [Executor.submit(build_concept, tax, self.name, lineage_flat.loc[[tax], 'idx'], matrix) for tax in rank_taxa]
            for future in concurrent.futures.as_completed(future_concepts):
                concept_tax = future.result()
                self.taxa[concept_tax.name] = concept_tax
        
        # summarize learning
        self.get_type_counts()
        self.list_solved()
        self.get_confusion(lineage_tab[self.name])
    

    def classify(self, query):
        # calculate signals for every taxon/query
        rank_signals = {}
        for tax, concept in self.taxa.items():
            rank_signals[tax] = concept.get_signal(query) / concept.n_rules
        rank_signals = pd.DataFrame(rank_signals)
        
        # filter out taxa with no calls (signal value of 1)
        called_taxa = rank_signals.loc[:, (rank_signals == 1).any(axis=0)] == 1
        
        # find queries with multiple taxon calls for the current rank
        rank_multihits = called_taxa.loc[called_taxa.sum(axis=1) > 1]
        
        # check for fully solved taxon calls in multihit queries
        solved_tab = pd.DataFrame(False, index=rank_multihits.index, columns=rank_multihits.columns)
        for tax in solved_tab.columns:
            try:
                if self[tax].solved == 'Full':
                    solved_tab[tax] = True
            except KeyError:
                pass
        # clear partially solved taxa sharing multihit with fully solved taxa
        solved_tab = solved_tab & rank_multihits
        solved_tab = solved_tab.loc[solved_tab.any(axis=1)]
        # update call table
        called_taxa.loc[solved_tab.index] = solved_tab.values
        
        return called_taxa, rank_signals
