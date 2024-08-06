#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:43:13 2024

@author: hernan
"""

#%% modules
import numpy as np
import pandas as pd

from ranks import Rank, rules_classify

#%% functions
def one_hot_encode(matrix):
    encoded = np.stack([matrix == 1,
                        matrix == 2,
                        matrix == 3,
                        matrix == 4], axis=2)
    return encoded

def flatten_lineage(R_lineage):
    # flatten lineage table, generate dataframe with columns: [idx, TaxId], filter out instances with unknown taxon
    lineage_flat = pd.concat(R_lineage[rk] for rk in R_lineage.columns).to_frame(name='TaxId').reset_index(names='idx')
    lineage_flat.set_index('TaxId', inplace=True)
    lineage_flat.drop(0, axis=0, inplace=True)
    return lineage_flat

def replace(df, names):
    """
    Replace data frame values (taxIDs) with their real names

    Parameters
    ----------
    df : pandas.DataFrame
        Classification table
    names : pandas.DataFrame
        Data frame mapping taxonoimic IDs to scientific names

    Returns
    -------
    df : pandas.DataFrame
        Modified data frame with scientific names instead of TaxIDs

    """
    df = df.copy()
    cols = df.columns
    for col in cols:
        df[col] = names[df[col]].values
    return df

def expand_report(classif_report, Q_branches):
    """
    Expand the classification report to include individual query sequences, indicating the branch they belong to

    Parameters
    ----------
    classif_report : pandas.DataFrame
        Compressed classification report
    Q_branches : pandas.DataFrame
        Dataframe containing the branch assignation of each query sequence

    Returns
    -------
    classif_expanded : pandas.DataFrame
        Data frame containing the taxonomic classification and assigned branch of each individual query sequence

    """
    selected_queries = Q_branches.query('Branch >= 0')
    report_expanded = classif_report.loc[selected_queries.Branch]
    report_expanded.index = selected_queries.index
    classif_expanded = pd.concat([Q_branches, report_expanded], axis=1)
    return classif_expanded

#%% classes
class ConceptLearner:
    def __getitem__(self, rank):
        return self.ranks[rank]
    
    def load_data(self, matrix, lineage_tab, lineage_collapsed, names_tab):
        self.matrix = one_hot_encode(matrix)
        self.lineage_tab = lineage_tab
        self.lineage_collapsed = lineage_collapsed
        self.lineage_flat = flatten_lineage(lineage_collapsed)
        self.names_tab = names_tab
        self.ranks = {rk:Rank(rk) for rk in lineage_tab.columns}
    
    def learn(self, threads=1):
        for rank in self.ranks.values():
            rank.learn(self.matrix, self.lineage_collapsed, self.lineage_flat, self.names_tab, threads=threads)
    
    def classify(self, Q, threads=1, out_file='report', keep=False, build_expanded=True, Q_branches=None):
        # Generates three data frames:
            # raw_report and norm_report have column multiindexes with levels Rank, Taxon. Contain raw and normalized scores respectively
            # classif report contains the final classification for each query at each level
        raw_scores = {}
        norm_scores = {}
        classification = []
        for rk, rank in self.ranks.items():
            # flassify queries for each rank
            rank_report = rank.classify(Q, threads)
            
            # rename taxon columns
            rank_report.rename(columns=self.names_tab, level=1, inplace=True)
            
            # separate rank reports into Raw signal, normalized signal and classification
            raw_scores[rk] = rank_report.Raw
            norm_scores[rk] = rank_report.Normalized
            classification.append(rank_report.Classification)
            
        raw_report = pd.concat(raw_scores, axis=1, names=['Rank', 'Taxon'])
        norm_report = pd.concat(norm_scores, axis=1, names=['Rank', 'Taxon'])
        classif_report = replace(pd.concat(classification, axis=1), self.names_tab)
        
        raw_report.to_csv(f'{out_file}_raw_score.csv')
        norm_report.to_csv(f'{out_file}_normalized_score.csv')
        classif_report.to_csv(f'{out_file}_classification.csv')
        if build_expanded and not Q_branches is None:
            classif_expanded = expand_report(classif_report, Q_branches)
            classif_expanded.to_csv(f'{out_file}_classification_expanded.csv')
            
        if keep:
            return raw_report, norm_report, classif_report
        return None, None, None
    
    def rules_classify(self, Q, threads=1):
        results = {}
        for rk, rank in self.ranks.items():
            rank_compatibles = rules_classify(Q, rank.rules_matrix, rank.taxa, threads = 1)
            unambiguous = rank_compatibles.query('Total_compatible == 1').drop(columns='Total_compatible')
            
            # assign classifications
            rank_compatibles['Classification'] = None
            rank_compatibles.loc[rank_compatibles.Total_compatible > 1, 'Classification'] = 'Unclear'
            if len(unambiguous) > 0:
                unambiguous_classif = np.tile(rank_compatibles.columns[:-2], (len(unambiguous), 1))[unambiguous]
                unambiguous_classif = self.names_tab.loc[unambiguous_classif].values
                rank_compatibles.loc[unambiguous.index, 'Classification'] = unambiguous_classif
            
            results[rk] = rank_compatibles
        results = pd.concat(results, axis=1, names=['Rank', 'Taxon'])
        results.rename(columns=self.names_tab, level=1, inplace=True)
        return results
    
    def build_summary(self, out_file):
        # get a summary of the learned concepts at each rank
        # TODO: add leading paragraph to summary, wxplaining what is what
        for rk, rank in self.ranks.items():
            rank.summary.format_tables(self.lineage_tab, self.names_tab)
            rank.summary.save(out_file)