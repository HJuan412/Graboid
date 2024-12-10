#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:58:06 2024

@author: hernan

ConceptLearner class defines an agent that classifies query sequences using
Concept learning criterion.
"""

#%% modules
import concurrent.futures
import numpy as np
import pandas as pd

from data_holder import DataHolder
# ignore numpy warnings when division by 0 appears
np.seterr(divide='ignore')

#%% functions
def count_value_differences(R_encoded, R_lineage):
    # flatten lineage
    lineage_flat = flatten_lineage(R_lineage)

    # count values per site for each taxon
    taxa = []
    val_counts = []
    
    for tax, tax_subtab in lineage_flat.groupby(lineage_flat.index):
        taxa.append(tax)
        val_counts.append(R_encoded[tax_subtab.idx.values].sum(axis=0))
    
    # calculate value differences
    val_counts = np.array(val_counts)
    full_count = R_encoded.sum(axis=0)
    count_diff = full_count - val_counts
    
    return count_diff, full_count, taxa

def build_type_tab(R_encoded, R_lineage):
    # count values per taxon & in the entire alignment
    count_diff, full_count, taxa = count_value_differences(R_encoded, R_lineage)
    # get changed values
    changed_vals = (count_diff != full_count).sum(axis=2)
    # get new zeros
    new_zeros = ((full_count != 0) & (count_diff == 0)).sum(axis=2)
    
    # get types
    # type 1 
    type1 = ((new_zeros > 0) & (new_zeros == changed_vals)).astype(int)
    # type 2
    type2 = ((new_zeros > 0) & (changed_vals > new_zeros)).astype(int)
    # type 3
    type3 = ((new_zeros == 0) & (changed_vals == 1)).astype(int)
    # type 4
    type4 = ((new_zeros == 0) & (changed_vals > 1)).astype(int)
    
    # build type matrix and table
    type_matrix = type1 + type2*2 + type3*3 + type4*4
    type_tab = pd.DataFrame(type_matrix, index=pd.Series(taxa, name='Taxon'))
    
    # summarize type table
    type_tab_summ = np.array([type1.sum(axis=1),
                              type2.sum(axis=1),
                              type3.sum(axis=1),
                              type4.sum(axis=1)]).T
    type_tab_summ = pd.DataFrame(type_tab_summ, columns = pd.Series([1,2,3,4], name='Site_Type'), index=pd.Series(taxa, name='Taxon'))
    type_tab_summ[0] = type_tab.shape[1] - type_tab_summ.sum(axis=1)
    return type_tab, type_tab_summ

#%% Concept learner
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
    
    def build_summary(self, out_file):
        # get a summary of the learned concepts at each rank
        # TODO: add leading paragraph to summary, wxplaining what is what
        for rk, rank in self.ranks.items():
            rank.summary.format_tables(self.lineage_tab, self.names_tab)
            rank.summary.save(out_file)

#%% Rank
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
        
        non_shared_sites = tax_concept.non_shared # remember to update non_shared attribute of concept taxa after getting shared info
        non_shared_vals = tax_concept.signal[non_shared_sites].values
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

class Rank:
    def __init__(self, name):
        self.name = name
        self.taxa = {}
        self.scores_tab = None
        self.shared_tab = None
        self.signal_matrix = None
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

#%% Summary
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
            
#%% Concept
def get_distinct_vals(matrix, in_indexes):
    """
    Identify all sequences within the in_indexes-defined group that contain at least one distinctive site for the concept taxon
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : array-like
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    
    Returns
    -------
    distinct_sites : numpy.array
        Array of sites containing distinctive values for the concept taxon
    distinct_vals : list
        List of arrays containing the distinctive values found in each site
        Contains the same number of elements as distinct_sites
        Each site may contain multiple distinctive values
    solved : numpy.array
        Array of indexes of solved (fully distinguished from the outsider sequences by at least one distinctive value) concept taxon sequences
    unsolved : numpy.array
        Array of indexes of concept taxon sequences without distinctive values
    
    """
    
    # count values in full alignment and taxon sub_alignment
    full_count = matrix.sum(axis=0)
    
    # define in_matrix as a subset of the matrix, count values in in_matrix
    in_matrix = matrix[in_indexes]
    in_count = in_matrix.sum(axis=0)
    
    # get count differences
    diff_count = full_count - in_count
    
    # identify distinctive sites & values
    new_zeros = ((full_count != 0) & (diff_count == 0)) # sites where the removal of in_matrix generates one or more value count to drop to 0
    distinct_sites = np.arange(matrix.shape[1])[new_zeros.sum(axis=1) > 0]
    distinct_vals = new_zeros * np.array([1,2,3,4])
    distinct_vals = [dv[nz] for dv, nz in zip(distinct_vals[distinct_sites], new_zeros[distinct_sites])] # get the distinctive values in each of the selected sites
    
    # determine solved sequences
    solved = np.full(len(in_indexes), False)
    for site, vals in zip(distinct_sites, distinct_vals):
        solved_by_site = np.any(in_matrix[:, site, vals-1], axis=1) # get sequences that are solved by the current value's distinctive sites
        solved = solved | solved_by_site
    
    # record indexes of solved and unsolved sequences
    unsolved = in_indexes[~solved].values
    solved = in_indexes[solved].values
    
    return distinct_sites, distinct_vals, solved, unsolved

def filter_sites(matrix, max_unknown=0.05):
    """
    Select sites that contain a single KNOWN value across all rows of the matrix
    Filter out sites with more than max_unknown % unknown sites
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    max_unknown : float, optional
        Unknown values threshold. The default is 0.05.
    
    Returns
    -------
    single_val_sites : numpy.array
        Array of sites with a single known value
    filtered_sites : numpy.array
        Array of single value sites with less than max_unknown % missing values
    
    """
    # get sites with single value in the in_matrix
    # get a list of site indexes in the in_matrix that have less than max_unknown missing values
    
    # get sites with single known value
    site_indexes = np.arange(matrix.shape[1])
    single_val_sites = site_indexes[np.any(matrix, axis=0).sum(axis=1) == 1]
    
    # filter sites by known content
    sites_content = np.any(matrix[:, single_val_sites], axis=2).sum(axis=0) / matrix.shape[0]
    sites_content = sites_content >= (1 - max_unknown)
    filtered_sites = single_val_sites[sites_content]
    
    return single_val_sites, filtered_sites

def get_full_sequences(matrix):
    """
    Get sequences that contain no missing values in the single value sites

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)

    Returns
    -------
    seq_full : numpy.array
        Indexes of sequences of matrix that contain no missing values in the single value sites

    """
    # get full sequences with single known values from the concept matrix
    
    # get single value sites
    single_val_sites, filtered_sites = filter_sites(matrix, 0)
    
    # get full sequences & incomplete sequences
    seq_full = np.any(matrix[:, single_val_sites], axis=2).sum(axis=1) == single_val_sites.shape
    
    # returns boolean array indicating which sequences in matrix are full
    return seq_full

def solve(matrix, in_indexes, out_indexes, site_indexes):
    """
    Identify the signal that differentiates the sequences of in_indexes from the largest amount of outsider sequences
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : numpy.array
        Array of indexes of concept sequences to be solved
    out_indexes : numpy.array
        Array of indexes of outsider sequences
    site_indexes : numpy.array
        Array of indexes of pre-filtered sites (single value, all known)
    
    Returns
    -------
    signal : numpy.array
        Array of site indexes included in the signal
    signal_solved : bool
        Boolean that indicates if the entirety of the outsider sequences could be distinguished
    to_exclude_seqs : numpy.array
        Array of outsider sequences that the signal sites cannot differentiate from the concept sequences
    
    """
    
    # get value for each site in the inner matrix & generate list of unchecked sites
    in_matrix_vals = np.any(matrix[in_indexes], axis=0)
    unchecked_sites = site_indexes.copy()
    
    # initialize signal array
    signal = []
    signal_solved = False
    to_exclude_seqs = out_indexes.copy()
    
    # iterate while solving signal
    for _ in np.arange(len(site_indexes)):
        # count shared values for each remaining unchecked site
        shared_vals = matrix[to_exclude_seqs][:, unchecked_sites].sum(axis=0)[in_matrix_vals[unchecked_sites]]
        min_shared = shared_vals.min()
        
        if min_shared == len(to_exclude_seqs):
            # if minimum shared values equals the number of remaining outsider sequences, we can't solve the signal any further
            break
        
        # select site with least shared values, best site is taken from the unchecked sites list
        sorted_sites = np.argsort(shared_vals)
        best_site = unchecked_sites[sorted_sites[0]]
        # update signal array
        signal.append(best_site)
        
        if min_shared == 0:
            # if there are no shared values remaining, signal is resolved
            signal_solved = True
            break
        
        # remove solved sequences from out_matrix
        best_site_value = in_matrix_vals[best_site]
        rows_to_keep = matrix[to_exclude_seqs, best_site, best_site_value].flatten() # keep all sequences that match the submatrix value at best site
        to_exclude_seqs = to_exclude_seqs[rows_to_keep]
        
        # remove selected best site from unchecked sites
        unchecked_sites = np.delete(unchecked_sites, sorted_sites[0])
    
    signal = np.array(signal)
    # out_excluded = np.setdiff1d(out_indexes, to_exclude_seqs, assume_unique=True)
    # return signal, signal_solved, out_excluded
    
    # return not excluded out_sequences (to_exclude_seqs)
    return signal, signal_solved, to_exclude_seqs

def get_composite_signal(matrix, in_indexes, unsolved_indexes):
    """
    Identify taxon signals that allow to differentiate unsolved concept sequences from the greatest number of outsider sequences
    Vertical signal: determined using only full sites (columns) of concept taxon alignment
    Horizontal signal: calculated using only full sequences (rows) of the contept taxon alignment
    
    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    in_indexes : array-like
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    unsolved_indexes : array-like
        Array containing the indexes of the concept sequences with no distinctive values. These sequences are the ones to be solved
    
    Returns
    -------
    signal_v : numpy.array
        Array of site indexes included in the VERTICAL signal
    values_v : numpy.array
        Array of semi-distinctive values of each site in the VERTICAL signal
    signal_h : numpy.array
        Array of site indexes included in the HORIZONTAL signal
    values_h : numpy.array
        Array of semi-distinctive values of each site in the HORIZONTAL signal
    intersection_v : numpy.array
        Array of outsider sequences indexes not differentiated by the VERTICAL signal
    intersection_h : numpy.array
        Array of outsider sequences indexes not differentiated by the HORIZONTAL signal
    seqs_h : numpy.array
        Array of concept sequences solved by the HORIZONTAL signal
        This array may contain incomplete concept sequences that nonetheless have known values in every site of the HORIZONTAL signal
    
    """
    
    # get matrix of unsolved taxon sequences & matrix of outsider sequences
    in_matrix = matrix[unsolved_indexes]
    out_indexes = np.delete(np.arange(matrix.shape[0]), in_indexes)
    
    # get full sites and full sequences
    single_val_sites, full_value_sites = filter_sites(in_matrix, 0)
    seqs_full = get_full_sequences(in_matrix) # seqs full is a boolean array that specifies which rows of in_matrix are complete sequences (no unknowns)
    
    # predefine output values
    signal_v = np.array([], dtype=int)
    values_v = np.array([], dtype=int)
    signal_h = np.array([], dtype=int)
    values_h = np.array([], dtype=int)
    intersection_v = np.array([], dtype=int)
    intersection_h = np.array([], dtype=int)
    seqs_h = np.array([], dtype=int)
    
    # get vertical signal (use full single value sites)
    if len(single_val_sites) > 0:
        signal_v, signal_solved_v, intersection_v = solve(matrix, unsolved_indexes, out_indexes, full_value_sites)
    
    # get horizontal signal (use complete sequences)
    if seqs_full.sum() > 0:
        signal_h, signal_solved_h, intersection_h = solve(matrix, unsolved_indexes[seqs_full], out_indexes, single_val_sites)
    
        # get incomplete sequences that include all sites of the horizontal signal
        incomp_in_h = np.any(in_matrix[~seqs_full][:, signal_h], axis=2).sum(axis=1) == (~seqs_full).sum()
        seqs_h = np.concatenate((unsolved_indexes[seqs_full], unsolved_indexes[~seqs_full][incomp_in_h]))
    
    # get signal values
    values_v = np.any(in_matrix[:, signal_v], axis=0)
    values_v = (values_v * np.array([1,2,3,4]))[values_v]
    values_h = np.any(in_matrix[:, signal_h], axis=0)
    values_h = (values_h * np.array([1,2,3,4]))[values_h]
    
    return signal_v, values_v, signal_h, values_h, intersection_v, intersection_h, seqs_h

def process_lineage(rank, lineage_tab):
    """
    Extracts the columns corresponding to rank and the one immediately below from the lineage table
    Renames column names as Rank and Sub_rank
    Replaces missing values in rank column for the values at the leading rank (unless rank is the highest)
    If the selected rank is the lowest one, both columns correspond to rank
    
    Parameters
    ----------
    rank : str
        Rank column to be selected
    lineage_tab : pandas.DataFrame
        Lineage table to be processed
    
    Returns
    -------
    lineage_processed : pandas.DataFrame
        Two column dataframe containing the rank column with updated missing values (values missing in the leading rank are left as 0)
    rank_tail : str
        Rank immediately below the selected one
    
    """
    
    rank_lead = ({lineage_tab.columns[0]:None} | {rk0:rk1 for rk0, rk1 in zip(lineage_tab.columns[1:], lineage_tab.columns[:-1])})[rank]
    rank_tail = ({rk0:rk1 for rk0, rk1 in zip(lineage_tab.columns[:-1], lineage_tab.columns[1:])} | {lineage_tab.columns[-1]:None})[rank]
    rank_tail = rank if rank_tail is None else rank_tail
    
    # get current and following ranks, if current rank is the lowest, use current rank as the following
    lineage_processed = lineage_tab[[rank, rank_tail]].copy()
    lineage_processed.columns = ['Rank', 'Sub_rank']
    
    if rank_lead is None:
        return lineage_processed, rank_tail
    
    #fill missing values in rank (if not at the highest rank)
    missing_vals = lineage_processed.query('Rank == 0').index
    lineage_processed.loc[missing_vals, 'Rank'] = lineage_tab.loc[missing_vals, rank_lead]
    
    return lineage_processed, rank_tail

def get_unsolved_subtaxa(seqs_h, seqs_v, in_indexes, rank, lineage_tab):
    """
    Count the number of unsolved sequences for vertical and horizontal signals
    
    Parameters
    ----------
    seqs_h : numpy.array
        Array of concept sequences solved by the HORIZONTAL signal
    seqs_v : numpy.array
        Array of concept sequences solved by the VERTICAL signal
    in_indexes : numpy.array
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    rank : str
        Taxonomic rank of the concept
    lineage_tab : pandas.DataFrame
        Lineage table
    
    Returns
    -------
    solved_subtaxa : pandas.DataFrame
        Data frame counting the unsolved sequences for each subtaxon of the concept. Also includes the total count of sequences for each subtaxon
    
    """
    
    # all solved sequences have 0 intersections
    
    # process lineage table (columns = multiindex with level 0 values Rank, Sub_rank)
    lineage_tab, sub_rank = process_lineage(rank, lineage_tab)
    
    in_subtaxa = lineage_tab.loc[in_indexes, 'Sub_rank'].unique()
    
    solved_subtaxa = pd.DataFrame(0, index=in_subtaxa, columns='Unsolved_h Unsolved_v Total'.split())
    solved_subtaxa['Total'] = lineage_tab.loc[in_indexes, 'Sub_rank'].value_counts()
    
    # get number of intersections in partial_h
    h_subtaxa = lineage_tab.loc[seqs_h, 'Sub_rank'].value_counts()
    solved_subtaxa.loc[h_subtaxa.index, 'Unsolved_h'] = h_subtaxa
    
    # get number of intersections ONLY in partial_v
    v_subtaxa = lineage_tab.loc[seqs_v, 'Sub_rank'].value_counts()
    solved_subtaxa.loc[v_subtaxa.index, 'Unsolved_v'] = v_subtaxa
    
    return solved_subtaxa

def get_confused_out_taxa(intersect_h, intersect_v, in_indexes, rank, lineage_tab):
    """
    Count the number  number of non distinguishable outsider sequences for vertical and horizontal signals
    
    Parameters
    ----------
    intersect_h : numpy.array
        Array of outsider sequences indexes not differentiated by the HORIZONTAL signal
    intersect_v : numpy.array
        Array of outsider sequences indexes not differentiated by the VERTICAL signal
    in_indexes : numpy.array
        Array containing the indexes of the concept sequences. Values must be between 0 and n_sequences - 1
    rank : str
        Taxonomic rank of the concept
    lineage_tab : pandas.DataFrame
        Lineage table
    
    Returns
    -------
    confused_subtaxa : pandas.DataFrame
        Data frame counting the number of non distinguishable sequences for each outsider subtaxa. Also includes the total count of sequences for each subtaxon

    """
    
    # sequences in partial_h have !exclusion.Horizontal intersections
    # sequences ONLY in partial_v have !exclusion.Vertical intersections
    
    out_taxa = lineage_tab.drop(in_indexes)[rank].unique()
    
    confused_out_taxa = pd.DataFrame(0, index=out_taxa, columns='Confused_h Confused_v Total'.split())
    confused_out_taxa['Total'] = lineage_tab.drop(in_indexes)[rank].value_counts()
    
    # get number of intersections in partial_h
    h_intersections = lineage_tab.loc[intersect_h, rank].value_counts()
    confused_out_taxa.loc[h_intersections.index, 'Confused_h'] = h_intersections
    
    # get number of intersections ONLY in partial_v
    v_intersections = lineage_tab.loc[intersect_v, rank].value_counts()
    confused_out_taxa.loc[v_intersections.index, 'Confused_v'] = v_intersections
    
    return confused_out_taxa

def get_MES(matrix, branch, branch_sites):
    """
    Select the MES (most entropic site) for the given group of sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    branch : numpy.array
        Array of indexes of the sequences included in the branch to be split
    branch_sites : list
        List of sites already in use for the branch to be split

    Returns
    -------
    MES : int
        Index of the most entropic site among the branch sequences (accounting for already used sites)

    """
    # remove used sites from the matrix
    branch_matrix = np.delete(matrix[branch], branch_sites, axis=1)
    used_sites = np.delete(np.arange(matrix.shape[1]), branch_sites)
    
    # select MES (most entropic site)
    freqs = branch_matrix.sum(axis=0) / branch_matrix.shape[0]
    ent = -(np.where(freqs == 0, freqs, np.log2(freqs)) * freqs).sum(axis=1)
    MES = np.argsort(ent)[-1]
    
    # adjust MES
    MES = used_sites[MES]
    return MES

def split_branch(matrix, branch, branch_sites):
    """
    Find the best site to separate a given group of sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    branch : numpy.array
        Array of indexes of the sequences included in the branch to be split
    branch_sites : list
        List of sites already in use for the branch to be split

    Returns
    -------
    new_branches : list
        List of newly generated branches, each one is a sub array of the original branch array
    new_branch_sites : list
        List of used sites in the newly generated branches, contains one list for each new branch
        New sites lists are a copies of the branch_sites list, updated to include the site selected in the current iteration
    closed_branches : list
        List of newly generated branches that are closed (cannot be further divided)

    """
    # locate most entropic site
    branch_MES = get_MES(matrix, branch, branch_sites)
    branch_sites = branch_sites + [branch_MES]
    
    # split by MES (we don't care about the value)
    splitter = matrix[branch, branch_MES].T # get MES in the encoded matrix
    splitter = splitter[splitter.any(axis=1)] # filter empty columns (bases without representatives)
    
    sub_branches = [branch[base] for base in splitter]
    
    new_branches = []
    closed_branches = []
    # filter sub branches
    for sub_branch in sub_branches:
        # if a sub branch has no variable sites or a single sequence, it is complete
        has_variable_sites = matrix[sub_branch].any(axis=0).sum(axis=1).max() > 1 # count number of values per each site (any & sum), determine if at least one site has multiple values (max > 1)
        if len(sub_branch) > 1 and has_variable_sites:
            new_branches.append(sub_branch)
        else:
            closed_branches.append(sub_branch)
    new_branch_sites = [branch_sites for _ in new_branches]
    return new_branches, new_branch_sites, closed_branches

def get_convos(matrix, sequences, variable_sites):
    """
    Get unique value combinations of variable selected sites in the concept

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    sequences : numpy.array
        Array of indexes of the sequences included in the concept tax
    variable_sites : numpy.array
        Array of indexes of the variable selected sites in the concept

    Returns
    -------
    convos : numpy.array
        2D-Array containing all unique combinations of values on the variable selected sites

    """
    
    # select variable sites from matrix
    matrix = matrix[:, variable_sites]
    # Define lists containing branches and branch sites
    branches = [sequences]
    branch_sites = [[]]
    # define list containing completed branches
    branches_closed = []
    
    # iterate and split branches
    for _ in variable_sites:
        # define containers for newly generated branches
        new_branches = []
        new_branch_sites = []
        
        for branch, br_sites in zip(branches, branch_sites):
            new_subbranches, new_subbranch_sites, closed_subbranches = split_branch(matrix, branch, br_sites)
            new_branches += new_subbranches
            new_branch_sites += new_subbranch_sites
            branches_closed += closed_subbranches
        branches = new_branches
        branch_sites = new_branch_sites
        if len(branches) == 0:
            # no more unsolved branches to solve
            break
    
    # get convo sequences
    bases_mat = np.tile([1,2,3,4], (matrix.shape[1], 1))
    convos = []
    for br in branches_closed:
        consensus = matrix[br].any(0)
        missing = consensus.any(1)
        br_seq = np.zeros(consensus.shape[0], dtype=int)
        br_seq[missing] = bases_mat[consensus]
        convos.append(br_seq)
    convos = np.array(convos)
    return convos

def compress(matrix, sequences, *sites):
    """
    Separate variable and invraiable sites, extract representatives of each combination of values present in the concept sequences

    Parameters
    ----------
    matrix : numpy.array
        Encoded alignment array. 3D boolean array of shape (sequences, sites, 4)
    sequences : numpy.array
        Array of indexes of the sequences included in the concept tax
    *sites : TYPE
        Arrays of selected sites in the concept tax

    Returns
    -------
    sites_single : numpy.array
        Array of indexes of selected sites with a single known value among the concept sequences
    values_single : numpy.array
        Array of values present in the single value sites of the concept taxon
    sites_multi : numpy.array
        Array of indexes of selected sites with multiple known values among the concept sequences
    values_multi : numpy.array
        2D array containing every unique combination of values present among the multiple value sites in the concept taxon

    """
    # get a list of unique and variable sites among the sequence set
    sites = np.unique(np.concatenate(sites)).astype(int)
    
    # get invariable sites
    single_value = matrix[sequences][:, sites].any(axis=0).sum(axis=1) <= 1
    sites_single = sites[single_value]
    sites_multi = sites[~single_value]
    
    # get representative values
    values_single = matrix[sequences][:, sites_single].any(axis=0)
    values_single = np.tile([1,2,3,4], (values_single.shape[0], 1))[values_single]
    values_multi = get_convos(matrix, sequences, sites_multi)
    return sites_single, values_single, sites_multi, values_multi

class Concept:
    def __init__(self, name, rank=None):
        self.name = name
        self.rank = rank
        self.sequences = np.array([])
        
        self.informative_sites = np.array([])
        self.informative_values = np.array([])
        
        self.seqs_v = np.array([])
        self.seqs_h = np.array([])
        self.seqs_v_only = np.array([])
        self.signal_v = np.array([])
        self.signal_h = np.array([])
        self.values_v = np.array([])
        self.values_h = np.array([])
        self.intersection_v = np.array([])
        self.intersection_h = np.array([])
        self.intersection_hv = np.array([])
        self.intersection_vh = np.array([]) # alias to avoid confusion
        self.signal = pd.Series()
        self.non_shared = np.array([])
        
        self.unsolved_subtaxa = None
        self.confused_out_taxa = None
        
        self.solved = np.array([])
        self.not_solved = np.array([])
        self.fully_solved = False
    
    def learn(self, matrix, concept_sequences, lineage_tab):
        # register indexes of concept and outsider sequences 
        self.sequences = concept_sequences.values
        
        # select sequences with at least one distinctive value
        dist_sites, dist_vals, solved, unsolved = get_distinct_vals(matrix, concept_sequences)
        self.informative_sites = dist_sites
        self.informative_values = dist_vals
        
        # attempt to find composite signal for unsolved sequences
        if len(unsolved) > 0:
            signal_v, values_v, signal_h, values_h, intersection_v, intersection_h, seqs_h = get_composite_signal(matrix, concept_sequences, unsolved)
            
            self.seqs_v = unsolved
            self.seqs_v_only = np.setdiff1d(unsolved, seqs_h, assume_unique=True)
            self.seqs_h = seqs_h
            self.signal_v = signal_v
            self.signal_h = signal_h
            self.values_v = values_v
            self.values_h = values_h
            self.intersection_v = intersection_v
            self.intersection_h = intersection_h
            self.intersection_hv = np.intersect1d(intersection_v, intersection_h)
            self.intersection_vh = self.intersection_hv # alias to avoid confusion
            
            # signal attributes is a series that merges the sites and values of horizontal and vertical signal
            self.signal = pd.Series(0, index=np.unique(np.concatenate((self.signal_v, self.signal_h))))
            self.signal.loc[self.signal_v] = self.values_v
            self.signal.loc[self.signal_h] = self.values_h
            
            # count fully and partially solved sequences
            if len(intersection_v) == 0:
                # vertical signal includes all unsolved sequences, if it excludes all outsider sequences (no intersection), all the concept is solved
                solved = np.concatenate([solved, unsolved])
                unsolved = np.array([])
            elif len(intersection_h) == 0:
                # sequences solved by horizontal signal registered, concept sequences not included in the horizontal signal are unsolved
                solved = np.concatenate([solved, seqs_h])
                unsolved = self.seqs_v_only
        self.solved = solved
        self.not_solved = unsolved
        if len(unsolved) == 0:
            self.fully_solved = True
        
        self.unsolved_subtaxa = get_unsolved_subtaxa(self.seqs_h, self.seqs_v, self.sequences, self.rank, lineage_tab)
        self.confused_out_taxa = get_confused_out_taxa(self.intersection_h, self.intersection_vh, self.sequences, self.rank, lineage_tab)
        
        # compress concept sequences
        self.sites_single, self.values_single, self.sites_multi, self.values_multi = compress(matrix, self.sequences, self.informative_sites, self.signal_v, self.signal_h)

#%% Main
def load_data(ref_dir,
              qry_file,
              qry_dir='.',
              evalue=0.0005,
              dropoff=0.05,
              min_height=0.1,
              min_width=2,
              qry_name='QUERY',
              min_cov=.95,
              filter_rank='family',
              max_unk_thresh=.2,
              threads=1):
    # load data
    data = DataHolder()
    data.load_reference(ref_dir)
    # load query
    data.load_query(qry_file,
                    qry_dir=qry_dir,
                    evalue=evalue,
                    dropoff=dropoff,
                    min_height=min_height,
                    min_width=min_width,
                    threads=threads,
                    qry_name=qry_name)
    
    # filter and process data
    data.filter_data(min_cov=min_cov, rank=filter_rank)
    data.collapse(max_unk_thresh=.2)
    return data

def main(ref_dir, qry_file, query_dir='.', evalue=0.0005, dropoff=0.05, min_height=0.1, min_width=2, threads=1, qry_name='QUERY'):
    
    # load data
    data = load_data(qry_file=qry_file,
                     query_dir=query_dir,
                     evalue=evalue,
                     dropoff=dropoff,
                     min_height=min_height,
                     min_width=min_width,
                     threads=threads,
                     qry_name=qry_name)
    
    # learn concept
    learner = ConceptLearner()
    learner.load_data(data.R.collapsed, data.R.lineage_tab, data.R.lineage_collapsed, data.R.names_tab)
    learner.learn(threads=threads)
    
    # classify query
    learner.classify(data.Q.collapsed, threads=threads)
