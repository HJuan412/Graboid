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

from concept import Concept
#%% functions
def build_concept(taxon, rank_name, tax_idxs, lineage_tab, matrix):
    concept_tax = Concept(taxon, rank_name)
    concept_tax.learn(matrix, tax_idxs, lineage_tab)
    return concept_tax

def get_shared_info(taxa):
    """
    Collect information about the sites shared among a given rank's concepts' composite signals
    This information will be used in the calculation for the signal scores
    
    Parameters
    ----------
    taxa : dict
        Dictionary with key:value -> taxID:Concept
        Contains a taxon Concept instances
    
    Returns
    -------
    shared_tab : pandas.DataFrame
        Table with columns Max_shared, Rem_shared, Total_shared, Non_shared
        Counts the amount of shared sites for the composite signal of every taxon as well as the non-shared sites and the remainder shared
        Remainder shared is the sum of sites not shared with a taxon that has the maximum amount of shared sites
    non_shared : dict
        Dictionary with key:value -> taxID:numpy.array
        Contains the indexes of non-shared sites for each taxon
    
    """
    
    # build table containing the used sites by each concept taxon
    sites = np.unique(np.concatenate([tx.signal.index for tx in taxa.values()]))
    used_sites = pd.DataFrame(False, index=sites, columns=taxa.keys())
    for tax, tax_concept in taxa.items():
        used_sites.loc[tax_concept.signal.index, tax] = True
    
    # get non_shared sites for each concept taxon
    non_shared = {}
    # build shared information tab
    shared_tab = pd.DataFrame(0, index=taxa.keys(), columns='Max_shared Rem_shared Total_shared Non_shared'.split())
    for tax in taxa.keys():
        # select all sites used by tax, remove tax column
        tax_sites = used_sites.loc[used_sites[tax]].drop(columns=tax)
        
        # count shared sites with other taxa, get max amount of shared
        total_shared = tax_sites.sum()
        max_shared = total_shared.max()
        shared_tab.loc[tax, 'Max_shared'] = max_shared
        
        # record total shared and non shared sites
        shared_sites = tax_sites.any(axis=1)
        shared_tab.loc[tax, 'Total_shared'] = shared_sites.sum()
        shared_tab.loc[tax, 'Non_shared'] = (~shared_sites).sum()
        non_shared[tax] = tax_sites[~shared_sites].index.values
        
        # get rem_shared
        # rem shared is the sum of sites not shared with a taxon that has max_shared shared sites
        max_shared_taxa = total_shared[total_shared == max_shared].index # get taxa with max_shared shared sites
        max_shared_sites = tax_sites[max_shared_taxa].any(axis=1) # get indexes of sites shared with a max_shared_taxa taxon
        shared_tab.loc[tax, 'Rem_shared'] = tax_sites.loc[~max_shared_sites].any(axis=1).sum()
    
    return shared_tab, non_shared

def get_signal_scores(shared_tab):
    """
    Calculate signal scores for each concept using shared signal data.
    For shared composite signal sites, signal value is 1
    For non-shared composite signal sites, signal value must be greater than the maximum number of shared sites for the taxon minus the number of remainder shared
    For distinctive sites, signal value must be greater than the maximum composite signal
    
    Parameters
    ----------
    shared_tab : pandas.DataFrame
        Resulting table from function get_shared_info
        Columns: Max_shared, Rem_shared, Total_shared, Non_shared
    
    Returns
    -------
    scores_tab : pandasDataFrame
        Table with columns NS_score, Info_score
        Stores the calculated signal score for non-shared composite signal sites and informative sites for every concept taxon
    
    """
    scores_tab = pd.DataFrame(0.0, index=shared_tab.index, columns = 'NS_score Info_score'.split(), dtype=np.float32)
    
    # ns_score (non_shared score)
    # non shared score given by equation max(((max_shared - rem_shared + 1) / non_shared), 1)
        # max_shared - min_shared: the score to beat, the maximum possible score for an outsider taxa is max_shared, all sites not shared with a max_shared taxon increase the signal for the correct taxon with respect to the wrong one
        # add 1 to the score to beat to ensure the accumulated score of non shared sites is greater than the score to beat
        # divide by number of non_shared to distribute the weight among the sites
        # ensure the resulting score if the resulting score is not lower than 1
        # round scores up to the first decimal
        # sites with no non_shared values have an infinite score, which is defaulted to 0
    ns_score = ((shared_tab.Max_shared - shared_tab.Rem_shared + 1) / shared_tab.Non_shared).apply(lambda x : max(1, x)).astype(np.float32).replace(np.inf, 0)
    ns_score = np.ceil(ns_score * 10) / 10
    scores_tab['NS_score'] = ns_score
    
    # Score of informative (distinctive site)
    # given by equation total_shared + non_shared * ns_score + 1
        # shared sites have a score of 1
        # to this add the accumulated non_shared score
        # add 1 to ensure the informative score is above shared score
    scores_tab['Info_score'] = shared_tab.Total_shared + shared_tab.Non_shared * scores_tab.NS_score + 1
    return scores_tab

def build_signal_mat(matrix, taxa, scores_tab):
    """
    Construct the signal matrix
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    taxa : dict
        Dictionary with key:value -> taxID:Concept
        Contains a taxon Concept instances
    scores_tab : pandas.DataFrame
        Resulting table from function get_signal_scores
        Columns: NS_score, Info_score
    
    Returns
    -------
    signal_matrix : numpy.array
        3D Numpy array of shape (matrix.shape[1], 5, 5) containing the signal score for each value for each site in the alignment (remember index 0 corresponds to missing values)
    
    """
    signal_matrix = np.zeros((matrix.shape[1], 5, 5), dtype=np.float16)
    
    for tax, tax_concept in taxa.items():
        
        info_score = scores_tab.loc[tax, 'Info_score']
        
        non_shared_sites = tax_concept.non_shared.astype(int) # remember to update non_shared attribute of concept taxa after getting shared info
        non_shared_vals = tax_concept.signal[non_shared_sites].values.astype(int)
        ns_score = scores_tab.loc[tax, 'NS_score']
        
        # record informative score
        for site, vals in zip(tax_concept.informative_sites, tax_concept.informative_values):
            signal_matrix[site][vals, vals] = info_score
        
        # record signal scores
        if len(tax_concept.signal) > 0:
            # set default score (1) for all signal sites/values
            signal_matrix[tax_concept.signal.index, tax_concept.signal.values, tax_concept.signal.values] = 1
            # update score for non shared sites/values
            signal_matrix[non_shared_sites, non_shared_vals, non_shared_vals] = ns_score
    
    return signal_matrix

def build_rules_matrix(matrix, taxa):
    """
    Construct the rules matrix

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    taxa : dict
        Dictionary with key:value -> taxID:Concept
        Contains a taxon Concept instances

    Returns
    -------
    rules_matrix : numpy.array
        3D Numpy boolean array of shape (matrix.shape[1], 5, 5) indicating when a value at a given site forms part of a rule (remember index 0 corresponds to missing values)

    """
    rules_matrix = np.full((matrix.shape[1], 5,5), False)
    rules_matrix[:,0,:] = True
    rules_matrix[:,:,0] = True

    for tax, tax_concept in taxa.items():
        for rule, val in zip(tax_concept.rules, tax_concept.rules_values):
            rules_matrix[rule, val, val] = True
    
    return rules_matrix

def get_concept_signal(Q, M, sn_sites, sn_values, ml_sites, ml_values):
    """
    Calculate the signal for each query sequence in Q for a given concept

    Parameters
    ----------
    Q : numpy.array
        2D array containing the sequences of the query dataset
    M : numpy.array
        3D array containing the signal values for each site of the alignment
    sn_sites : numpy.array
        Array containing indexes of single value sites of the concept taxon
    sn_values : numpy.array
        Array containing the values present in the single value sites of the concept taxon
    ml_sites : numpy.array
        Array containing indexes of multiple value sites of the concept taxon
    ml_values : numpy.array
        2D array containing all unique value combinations in the multiple value sites of the concept taxon

    Returns
    -------
    signal_total : numpy.array
        Array containing the calculated concept signal for each query sequence´

    """
    # extract signal matrix layer for each query sequences
    M_q = M[np.arange(len(Q[0])), Q] # shape is (# queries, # sites, 5)
    
    # get signal for single value sites (base signal) and the best signal for the variable sites
    
    # single value sites
    signal_single = np.zeros(Q.shape[0])
    if len(sn_sites) > 0:
        M_q_single = M_q[:, sn_sites] # extract sites with single value in the concept taxon
        signal_single = M_q_single[:, np.arange(sn_sites.shape[0]), sn_values].sum(axis=1)
    
    # multiple value sites
    signal_multi = np.zeros(Q.shape[0])
    if len(ml_sites) > 0:
        # variable concept sites, find the combination of variable sites in the concept that gets the best signal
        M_q_multi = M_q[:, ml_sites] # extract sites with multiple values in the concept taxon
        
        # get best signal for each combination of variable site values
        signal_multi = np.array([M_q_multi[:, np.arange(ml_sites.shape[0]), vm] for vm in ml_values]).sum(axis=2).max(axis=0)
    
    signal_total = (signal_single + signal_multi).astype(np.float16)
    return signal_total

def check_compatible(Q, taxa):
    """
    Check if the sequences contained in Q are compatible with the tax concepts contained in taxa

    Parameters
    ----------
    Q : numpy.array
        Sub array of the rules matrix containing the observed value for each query sequence
    taxa : dict
        Dictionary of key:value pairs -> tax name: tax concept

    Returns
    -------
    is_compatible : pandas.DataFrame
        Table indicating the compatible taxa for each query sequence

    """
    is_compatible = []
    for tax, tax_concept in taxa.items():
        q_vals_concept = Q[:, tax_concept.rules]
        compatible = (q_vals_concept & tax_concept.rules_encoded)[:,:,1:]
        is_compatible.append(pd.Series(np.all(np.any(compatible, axis=2), axis=1), name=tax))
    is_compatible = pd.concat(is_compatible, axis=1)
    return is_compatible

def rules_classify(Q, rules_matrix, taxa, threads=1):
    """
    Classify query instances using the rules matrix. If more than 1 thread is specified, use check_compatible function which allows for parallel classification
    NOTE: multiprocssing mode is less efficient at lower numbers of query sequences

    Parameters
    ----------
    Q : numpy.array
        Array of query sequences
    rules_matrix : numpy.array
        Boolean matrix containing the learned rules for all taxon concepts
    taxa : dict
        Dictionary of key:value pairs -> tax name: tax concept
    threads : int, optional
        Number of processors to use in the classification. The default is 1.

    Returns
    -------
    compatible_tab : pandas.DataFrame
        Table indicating the compatible taxa for each query sequence. Taxa with no compatible sequences are ommited.
        Total_compatible column records the number of compatible taxa for each query

    """
    # slicer function is used to subdivide the taxa dictionary
    def slicer(taxa, threads=1):
        tax_names = np.array(list(taxa.keys()))
        n = len(tax_names)
        
        indexes = np.arange(0, n, int(n/threads) + 1)
        if indexes.max() < n-1:
            indexes = np.append(indexes, -1)
        keys = [tax_names[idx0:idx1] for idx0, idx1 in zip(indexes[:-1], indexes[1:])]
        for k in keys:
            selected = {k_:taxa[k_] for k_ in k}
            yield selected
    
    # generate a 3d array indicating the rules values for each site in each query sequence
    q_vals = np.array([rules_matrix[np.arange(len(q)), q] for q in Q])
    
    # check compatibility using either single or multiple processors
    if threads == 1:
        compatible_tab = pd.DataFrame(False, index=np.arange(len(Q)), columns=taxa.keys())
        for tax, tax_concept in taxa.items():
            # select sites involved in the current concept
            q_vals_concept = q_vals[:, tax_concept.rules]
            # ensure compatibility with the current concept
            compatible = (q_vals_concept & tax_concept.rules_encoded)[:,:,1:]
            compatible_tab[tax] = np.all(np.any(compatible, axis=2), axis=1)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(check_compatible, q_vals, taxa_chunk) for taxa_chunk in slicer(taxa, threads)]
            compatible_tab = pd.concat([future.result() for future in concurrent.futures.as_completed(futures)], axis=1)
    
    # prune table, keep only concepts with compatible queries, record compatible taxa for each sequence
    found_concepts = compatible_tab.sum(axis=0)
    found_concepts = (found_concepts[found_concepts > 0]).index
    compatible_tab = compatible_tab[found_concepts]
    compatible_tab['Total_compatible'] = compatible_tab.sum(axis=1)
    return compatible_tab

#%% classes
class Rank:
    def __init__(self, name):
        self.name = name
        self.taxa = {}
        self.scores_tab = None
        self.shared_tab = None
        self.signal_matrix = None
        self.rules_matrix = None
        self.summary = None
    
    def __getitem__(self, taxon):
        return self.taxa[taxon]
    
    def learn(self, matrix, lineage_tab, lineage_flat, names_tab, threads=1):
        """
        Learn all concept taxa present in a given rank and generate corresponding signal matrix

        Parameters
        ----------
        matrix : numpy.array
            Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
        lineage_tab : pandas.DataFrame
            Lineage table
        lineage_flat : pandas.DataFrame
            Flattend lineage table (contains indexes for each taxon).
        names_tab : pandas.DataFrame
            Data frame mapping taxonoimic IDs to scientific names
        threads : int, optional
            Number of processors to use during concept learning. The default is 1.

        Returns
        -------
        None.

        """
        
        # get all non-null taxa present in the rank
        rank_taxa = np.unique(lineage_tab[self.name])
        rank_taxa = rank_taxa[rank_taxa > 0]
        
        # learn concepts
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as Executor:
            future_concepts = [Executor.submit(build_concept, tax, self.name, lineage_flat.loc[[tax], 'idx'], lineage_tab, matrix) for tax in rank_taxa]
            for future in concurrent.futures.as_completed(future_concepts):
                concept_tax = future.result()
                self.taxa[concept_tax.name] = concept_tax
        
        # calculate scores
        # get shared information
        self.shared_tab, non_shared = get_shared_info(self.taxa)
        # update non shared values for each concept taxon
        for tax, ns in non_shared.items():
            self.taxa[tax].non_shared = ns
        self.scores_tab = get_signal_scores(self.shared_tab)
        self.signal_matrix = build_signal_mat(matrix, self.taxa, self.scores_tab)
        
        # build rules matrix
        self.rules_matrix = build_rules_matrix(matrix, self.taxa)
        
        #build summary & confusion
        summ_columns = pd.MultiIndex.from_product([['Sequences', 'Subtaxa'],
                                                   ['Total', 'Solved', 'Solved_partial']])
        summary = pd.DataFrame(index=self.taxa.keys(), columns = summ_columns)
        confusion_index = pd.Index(list(self.taxa.keys()), name='Taxa')
        confusion_cols = pd.Index(list(self.taxa.keys()), name='Predicted')
        confusion = pd.DataFrame(0, index=confusion_index, columns=confusion_cols, dtype=int)
        confusion['% unsolved'] = .0
        
        for tax, tax_concept in self.taxa.items():
            summary.loc[tax, ('Sequences', 'Total')] = len(tax_concept.sequences)
            summary.loc[tax, ('Sequences', 'Solved')] = len(tax_concept.solved)
            summary.loc[tax, ('Sequences', 'Solved_partial')] = len(tax_concept.not_solved)
            summary.loc[tax, ('Subtaxa', 'Total')] = len(tax_concept.unsolved_subtaxa)
            summary.loc[tax, ('Subtaxa', 'Solved')] = len(tax_concept.unsolved_subtaxa.query('Unsolved_h == 0 & Unsolved_v == 0'))
            summary.loc[tax, ('Subtaxa', 'Solved_partial')] = len(tax_concept.unsolved_subtaxa.query('Unsolved_h > 0 | Unsolved_v > 0').index.values)
            
            confusion.loc[tax, tax_concept.confused_out_taxa.index] = tax_concept.confused_out_taxa.Confused_v
            confusion.loc[tax, '% unsolved'] = (len(tax_concept.not_solved) / len(tax_concept.sequences)) * 100
        # self.summary = Summary(self.name, summary, confusion, lineage_tab, names_tab)
        self.summary = Summary(self.name, summary, confusion)
    
    def classify(self, Q, threads=1):        
        # get signal scores for each query sequence for each concept        
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            future_signals = {executor.submit(get_concept_signal, Q, self.signal_matrix, concept.sites_single, concept.values_single, concept.sites_multi, concept.values_multi):tax for tax, concept in self.taxa.items()}
            signals = pd.DataFrame({future_signals[fs]:fs.result() for fs in concurrent.futures.as_completed(future_signals)})
        
        # normalize
        # divide the score for each concept by its calculated informative site score
        info_scores = self.scores_tab.loc[signals.columns, 'Info_score'].values
        signals_norm = signals / info_scores
        
        # merge signal tables
        columns = pd.MultiIndex.from_product([['Raw', 'Normalized'], signals.columns])
        signal_tab = pd.concat([signals, signals_norm], axis=1)
        signal_tab.columns = columns
        signal_tab.index.name = 'Query'
        
        # get classification
        signal_tab[('Classification', self.name)] = signals_norm.columns.values[np.argmax(signals_norm, axis=1)]
        return signal_tab
    
#%% Summary functions
def get_super_rank(ranks, rank):
    parent_ranks = {rk:pr for rk,pr in zip(ranks[1:], ranks[:-1])}
    try:
        super_rank = parent_ranks[rank]
    except KeyError:
        super_rank = None
    return super_rank

def format_rank_confusion(confusion, summary, lineage_tab, names_tab, rank):
    # get parent rank
    super_rank = get_super_rank(lineage_tab.columns, rank)
    
    if super_rank is None:
        confusion_flat = pd.DataFrame()
    else:
        # reformat confusion matrix, keep only columns belonging to predicted taxa
        confusion = confusion.loc[:, confusion.index]
        
        # linearize confusion matrix
        rows = []
        for tax, tax_row in confusion.iterrows():
            # filter out taxons with no predictions
            tax_row = tax_row[tax_row > 0]
            # define multiindex with levels: 0 = real taxon, 1 = predicted taxon
            tax_row.index = pd.MultiIndex.from_product([[tax], tax_row.index])
            rows.append(tax_row)
        confusion_flat = pd.concat(rows).to_frame(name='Counts')
        
        # add relative counts
        confusion_flat['%_confused'] = confusion_flat.Counts.values / summary.loc[confusion_flat.index.get_level_values(0), ('Sequences', 'Total')].values * 100
        
        # add parent taxa for real and predicted taxa
        confusion_flat['Real_parent'] = lineage_tab.loc[confusion_flat.index.get_level_values(0), super_rank].values
        confusion_flat['Pred_parent'] = lineage_tab.loc[confusion_flat.index.get_level_values(1), super_rank].values
        
        # build 4 level multi index
        confusion_flat.reset_index(names=['Real_tax', 'Pred_tax'], inplace=True)
        confusion_flat.set_index('Real_parent Real_tax Pred_parent Pred_tax'.split(), inplace=True)
        confusion_flat.rename(index=names_tab, inplace=True)
        
        # get summary of confused supertaxa
        for (rp, rt, pp), subtab in confusion_flat.groupby(level=[0,1,2]):
            total = subtab.sum(axis=0)
            confusion_flat.loc[(rp, rt, pp, f'.{pp} total')] = total
        confusion_flat.sort_index(inplace=True)
    return confusion_flat

#%% Summary class
class Summary:
    def __init__(self, rank, summary, confusion):
        self.rank = rank
        self.confusion = confusion
        self.summary = summary
    
    def format_tables(self, lineage_tab, names_tab):
        self.confusion_flat = format_rank_confusion(self.confusion,
                                                    self.summary,
                                                    lineage_tab,
                                                    names_tab,
                                                    self.rank)
        self.summary_format = self.summary.rename(index=names_tab)
    
    def save(self, out_file):
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'display.precision', 2):
            body = '\n\n'.join([f'Rank: {self.rank}',
                                'Summary',
                                str(self.summary_format),
                                '_' * 64,
                                'Confusion',
                                str(self.confusion_flat),
                                '=' * 64,
                                ''])
        with open(out_file, 'a') as handle:
            handle.write(body)