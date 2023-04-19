#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

@author: hernan
"""

#%% libraries
import concurrent.futures
import json
import logging
import numba as nb
import numpy as np
import os
import pandas as pd
import sys
import time

sys.path.append("..") # use this to allow importing from a sibling package
from calibration import reporter as rp
from classification import classification
from classification import cost_matrix
from DATA import DATA
from preprocess import feature_selection as fsele
from preprocess import windows as wn
#%% set
logger = logging.getLogger('Graboid.calibrator')
logger.setLevel(logging.DEBUG)

#%% functions
def get_metrics(confusion, taxons):
    # get calibration metrics for a given confusion matrix
    # confusion: confusion matrix
    # taxons: taxon index (generated by build_confusion)

    results = []
    for idx, tax in enumerate(taxons):
        # get true/false positives/negatives for each taxon
        tp = confusion[idx, idx]
        tn = np.delete(np.delete(confusion, idx, axis=0), idx, axis=1).sum()
        fp = np.delete(confusion[idx], idx).sum()
        fn = np.delete(confusion[:, idx], idx).sum()
        
        # calculate metrics (accuracy, precision, recall and F1)
        acc = (tp + tn) / (tp + tn + fp + fn)
        prc = 0
        rec = 0
        f1 = 0
        if tp > 0:
            # if there are no positive values, prec, rec and f1 are 0 by default
            prc = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (2 * prc * rec)/(prc + rec)
        
        results.append([tax, acc, prc, rec, f1])
    return results

def loo_generator(nrecs):
    # generates the indexes for the training dataset and the testing instance in leave-one-out calibration
    record_idxs = np.arange(nrecs)
    for idx in record_idxs:
        train_idx = np.delete(record_idxs, idx)
        test_idx = idx
        yield train_idx, test_idx

@nb.njit
def build_confusion(pred, real):
    # build the confusion matrix (for a given RANK)
    # pred: array of PREDICTED taxon for each training sample
    # pred: array of REAL taxon for each training sample

    # list & count taxes
    uniq_taxes = np.unique(real)
    n_taxes = len(uniq_taxes)
    
    confusion = np.zeros((n_taxes, n_taxes), dtype=np.int32)
    # rows: REAL taxons
    # columns: PREDICTED taxons
    
    for idx_0, tax_0 in enumerate(uniq_taxes):
        tax_pred = pred[real == tax_0]
        for idx_1, tax_1 in enumerate(uniq_taxes):
            pred_as = len(tax_pred[tax_pred == tax_1])
            confusion[idx_0, idx_1] = pred_as
    return confusion, uniq_taxes

def get_mem_magnitude(size):
    order = np.log2(size)
    # change order of magnitude at 750 b, KiB, MiB
    if order < 9.584962500721156:
        return size, 'b'
    if order < 19.584962500721158:
        return size / 1024, 'KiB'
    if order < 29.584962500721158:
        return size / 1048576, 'MiB'
    return size / 1073741824, 'GiB'

def get_classif_mem(classif_dir):
    file_list = os.listdir(classif_dir)
    mem = 0
    for file in file_list:
        mem += os.path.getsize(classif_dir + '/' + file)
    return get_mem_magnitude(mem)

def build_report(classification, taxonomy_matrix, w_start, w_end, out_file=None, log=False):
    # setup report logger
    logname = ''
    if log:
        logname = 'Graboid.calibrator.report'
    report_logger = logging.getLogger(logname)
    # classification is a pandas dataframe containing the classification results, index must be a multiindex with levels ['_k', 'n', 'mode']
    # taxonomy matrix is a numpy array with rows equal to the real taxonomies for the 
    
    # build rows & param guides
    # param guides keys are the ranks visited during calibration
    # each value is a list of tuples containing the parameter combination and the corresponding rows
    # this is used to retrieve the correct rows from each confusion matrix
    # looks like -> param_guides[rk] = [(params0, rows0), (param1, rows1),...]
    t01 = time.time()
    param_guides = {}
    for rk, rk_subtab in classification.groupby('rk'):
        param_guides[rk] = []
        aux_tab = pd.Series(index=rk_subtab.index, data=np.arange(len(rk_subtab)))
        for params, aux_subtab in aux_tab.groupby(level=[0,1,2]):
            param_guides[rk].append((params, aux_subtab.values))
    t02 = time.time()
    report_logger.debug(f'Report prep time {t02 - t01:.3f}')
    # get rank, index, predicted and real data from the classification table
    # index contains the query sequence position in the taxonomy matrix
    rk_data = classification.rk.astype(int).to_numpy()
    idx_data = classification.idx.astype(int).to_numpy()
    pred_data = classification.tax.astype(int).to_numpy()
    real_data = taxonomy_matrix[idx_data, rk_data]
    t03 = time.time()
    report_logger.debug(f'Report data gathered {t03 - t02:.3f}')
    # begin construction
    # report columns: Taxon Rank K n_sites mode Accuracy Precision Recall F1_score
    pre_report = [] # stores completed subreports
    # generate report by taxonomic rank
    for rk in np.unique(rk_data):
        t04 = time.time()
        # access the guide for the current rank
        guide = param_guides[rk]
        # locate rows in classification belonging to the current rank
        rk_idx = np.argwhere(rk_data == rk).flatten()
        # get predicted and actual values
        pred_submat = pred_data[rk_idx]
        real_submat = real_data[rk_idx]
        # get a list of unique taxons in the current rank
        rk_tax = np.unique(taxonomy_matrix[:,rk])
        mem, units = get_mem_magnitude(len(rk_idx) * 4)
        report_logger.debug(f'Rank {rk} confusion matrix of shape ({len(rk_idx)}, 4) and size {mem:.3f} {units}')
        # build confusion matrix per taxon in rank, generate metrics and move on
        for idx, tax in enumerate(rk_tax):
            tax_true = real_submat == tax
            tax_pred = pred_submat == tax
            
            tp = tax_true & tax_pred # true positives
            tn = ~tax_true & ~tax_pred # true negatives
            fn = tax_true & ~tax_pred # false negatives
            fp = ~tax_true & tax_pred # false positives
            confusion = np.array([tp, tn, fn, fp]).T
        
            # get the metrics for each parameter combination
            for params, rows in guide:
                # count occurrences of each case (true/false positive/negative) and calculate metrics
                sum_confusion = confusion[rows].sum(axis=0)
                # clip function used in metrics calculation to handle cases in which divisor is 0
                acc = (sum_confusion[0] + sum_confusion[1]) / (sum_confusion.sum(axis=0))
                prc = sum_confusion[0] / np.clip((sum_confusion[0] + sum_confusion[3]), a_min=np.e**-7, a_max=None)
                rec = sum_confusion[0] / np.clip((sum_confusion[0] + sum_confusion[2]), a_min=np.e**-7, a_max=None)
                f1 = (2 * prc * rec)/np.clip((prc + rec), np.e**-7, a_max=None)
                # build subreport, add rank and parameter data
                pre_subreport = [tax, rk, w_start, w_end, params[0], params[1], params[2], acc, prc, rec, f1]
                pre_report.append(pre_subreport)
        t05 = time.time()
        report_logger.debug(f'Rank {rk} calibration {t05 - t04:.3f}')
    t07 = time.time()
    report_logger.debug(f'Calibration complete {t07 - t03:.3f}')
    report = pd.DataFrame(pre_report, columns = 'Taxon Rank w_start w_end K n_sites mode Accuracy Precision Recall F1_score'.split())
    if out_file is None:
        return report
    report.to_csv(out_file, header = not os.path.isfile(out_file), index = False, mode = 'a')
    return

def get_ndists(idx, win, win_sites, cost_mat):
    # get all distances between item at position idx and the FOLLOWING elements in win for each level in win_sites (contains n_sites indexes)
    # returns dist_array and index_array
    # dist_array contains the calculated distances. It has a row for each level in win_sites and the number of columns equals the number of elements AFTER idx in win
    # index_array has the indexes of the pair of elements in each column of dist_array. Has two rows, idx goes in the upper one
    
    # pre generate arrays
    dist_array = np.zeros((len(win_sites), win.shape[0] - 1 - idx), dtype=np.float32)
    # construct index_array
    index_array = np.full((2, win.shape[0] - 1 - idx), idx, dtype=np.int16)
    index_array[1] = np.arange(idx + 1, win.shape[0])
    # calculate distances for each level in win_sites
    for nidx, n in enumerate(win_sites):
        # for each level, the distance includes the distance of the previous levels
        prev_dists = dist_array[max(nidx-1, 0)]
        dist_array[nidx] = classification.get_dists(win[[idx]][:,n], win[idx+1:][:,n], cost_mat)[0] + prev_dists
    
    return dist_array, index_array

def get_all_distances(win, win_sites, cost_mat):
    # get all possible paired distances in a single array per level
    # returns the concatenated dist_arrays and idx_arrays generated by get_ndists
    total_dists = np.empty((len(win_sites), 0), dtype=np.float32)
    total_indexes = np.empty((2,0), dtype=np.int16)
    # run get_ndists for every element except the last one, as it has no FOLLOWING elements
    for idx in np.arange(win.shape[0]-1):
        dists, indexes = get_ndists(idx, win, win_sites, cost_mat)
        # concatenate results to the existing matrices
        total_dists = np.concatenate((total_dists, dists), axis=1)
        total_indexes = np.concatenate((total_indexes, indexes), axis=1)
    return total_dists, total_indexes

def get_sorted_neighs(sorted_idxs, sorted_dists):
    # separate the neighbours for each individual element in the paired distance matrix
    # sorted_idxs contains the indexes array generated by get_all_distances, ordered by the distances calculated for a given level
    # returns arrays neighs (containing neighbour indexes) and dists (containing neighbour distances to the element)
    neighs = []
    dists = []
    # isolate the neighbours of each element and their respective distances
    for seq in np.arange(sorted_idxs.max() + 1):
        # get all pairs that include the current element
        seq_idxs = sorted_idxs == seq
        combined = seq_idxs[0] | seq_idxs[1] 
        seq_neighs = sorted_idxs[:, combined]
        # extract the ordered distances for the current element's neighbours
        dists.append(sorted_dists[combined])
        # extract the current element's neighbours' indexes
        ordered_neighs = np.full(seq_neighs.shape[1], -1, dtype=np.int16)
        ordered_neighs[seq_neighs[0] != seq] = seq_neighs[0][seq_neighs[0] != seq]
        ordered_neighs[seq_neighs[1] != seq] = seq_neighs[1][seq_neighs[1] != seq]
        neighs.append(ordered_neighs)
    return np.array(neighs), np.array(dists)

# classification
def compress_dists(distance_array):
    # for every (sorted) distance array, return unique values with counts (counts are used to group neighbours)
    return [np.stack(np.unique(dists, return_counts=True)) for dists in distance_array]

def wknn(distance_array):
    d1 = distance_array[:, [0]]
    dk = distance_array[:, [-1]]
    return (dk - distance_array) / (dk - d1)

def dwknn(distance_array, weighted):
    d1 = distance_array[:, [0]]
    dk = distance_array[:, [-1]]
    penal = (dk + d1) / (dk + distance_array)
    return weighted * penal

class Neighbours:
    def __init__(self, sorted_neighs, max_k):
        self.max_k = max_k
        # group neighbours by distance to query
        compressed = compress_dists(sorted_neighs[1])
        dist_pos = []
        indexes = []
        for neighs, comp in zip(sorted_neighs[0], compressed):
            neigh_dist_pos = np.full((2,max_k), np.nan)
            _k = min(len(comp[0]), max_k)
            neigh_dist_pos[:, :_k] = comp[:, :_k]
            dist_pos.append(neigh_dist_pos)
            indexes.append(neighs[:np.nansum(neigh_dist_pos[1]).astype(int)])
        dist_pos = np.array(dist_pos)
        self.distances = dist_pos[:,0] # contains collapsed distances up to the max_k position
        self.positions = dist_pos[:,1].astype(np.int16) # contains count of unique distance values (used to select indexes at a given orbit)
        self.indexes = indexes # contains #sequences arrays (of varying lengths) with the indexes of the neighbours that populate the orbits up to the k_max position
    
    def get_weights(self, k_range):
        self.k_range = k_range
        # weight and double weight distances for multiple values of k
        weighted = []
        double_weighted = []
        
        for k in k_range:
            k_weighted = wknn(self.distances[:,:k])
            weighted.append(k_weighted)
            double_weighted.append(dwknn(self.distances[:,:k], k_weighted))
        # both weighted and double weighted are lists containing len(k_range) arrays of weights of increasing size (from min(k_range) to max(k_range) by step_k)
        
        # both weighted and double_weighted are lists of len(k_range) elements corresponding to the weighted distances for the different values of K
        self.weighted = weighted
        self.double_weighted = weighted

# both classify functions return a list of #sequences elements containing #ranks tuples with an unweighted, weighted and double_weighted classification
# tax_tab should be the window's extended taxonomy passed as a numpy array
def orbit_classify(positions, indexes, weights, double_weights, tax_tab):
    # get all neighbours in the first k orbits
    k = weights.shape[1] # infer k from the number of calculated weights
    k_pos = positions[:,:k] # get position locators
    # get the weights for each orbit
    orbit_arrays = [] # use this as base to get the corresponding weight/double_weight for each neighbour, depending on the orbit it occupies
    for poss in k_pos:
        orbit_arrays.append(np.concatenate([[idx]*pos for idx, pos in enumerate(poss)]))
    weighted_arrays = [wghts[orbt] for wghts, orbt in zip(weights, orbit_arrays)]
    double_weighted_arrays = [dwghts[orbt] for dwghts, orbt in zip(double_weights, orbit_arrays)]
    # retrieve indexes of neighbours located within the k-th orbit
    k_indexes = [idxs[:len(orbt)] for idxs, orbt in zip(indexes, orbit_arrays)]
    
    classifs = []
    for idxs, w_arr, dw_arr in zip(k_indexes, weighted_arrays, double_weighted_arrays):
        sub_tax = tax_tab[idxs]
        rk_classifs = []
        for rk in sub_tax.T:
            taxa, counts = np.unique(rk, return_counts=True)
            tax_locs = [rk == tax for tax in taxa]
            # unweighted classify, majority_vote
            u_classif = taxa[np.argmax(counts)]
            # weighted classify
            tax_supp_weighted = [w_arr[tax].sum() for tax in tax_locs]
            w_classif = taxa[np.argmax(tax_supp_weighted)]
            # double weighted classify
            tax_supp_double_weighted = [dw_arr[tax].sum() for tax in tax_locs]
            d_classif = taxa[np.argmax(tax_supp_double_weighted)]
            rk_classifs.append((u_classif, w_classif, d_classif))
        classifs.append(rk_classifs)
    return classifs

def neigh_classify(positions, indexes, weights, double_weights, tax_tab):
    # get orbits containing the k-th neighbour
    k = weights.shape[1] # infer k from the number of calculated weights
    selected_orbits = np.argmax(np.cumsum(positions, axis=1) >= k, axis=1) + 1
    selected_positions = [pos[:orbt] for pos, orbt in zip(positions, selected_orbits)]
    orbit_arrays = [] # use this as base to get the corresponding weight/double_weight for each neighbour, depending on the orbit it occupies
    for poss in selected_positions:
        orbit_arrays.append(np.concatenate([[idx]*pos for idx, pos in enumerate(poss)]))
    weighted_arrays = [wghts[orbt] for wghts, orbt in zip(weights, orbit_arrays)]
    double_weighted_arrays = [dwghts[orbt] for dwghts, orbt in zip(double_weights, orbit_arrays)]
    # retrieve indexes of neighbours located within the k-th orbit
    k_indexes = [idxs[:len(orbt)] for idxs, orbt in zip(indexes, orbit_arrays)]
    
    classifs = []
    for idxs, w_arr, dw_arr in zip(k_indexes, weighted_arrays, double_weighted_arrays):
        sub_tax = tax_tab[idxs]
        rk_classifs = []
        for rk in sub_tax.T:
            taxa, counts = np.unique(rk, return_counts=True)
            tax_locs = [rk == tax for tax in taxa]
            # unweighted classify, majority_vote
            u_classif = taxa[np.argmax(counts)]
            # weighted classify
            tax_supp_weighted = [w_arr[tax].sum() for tax in tax_locs]
            w_classif = taxa[np.argmax(tax_supp_weighted)]
            # double weighted classify
            tax_supp_double_weighted = [dw_arr[tax].sum() for tax in tax_locs]
            d_classif = taxa[np.argmax(tax_supp_double_weighted)]
            rk_classifs.append((u_classif, w_classif, d_classif))
        classifs.append(rk_classifs)
    return classifs

def generate_classifications(sorted_neighbours, k_max, k_step, k_min, tax_tab, threads=3, criterion='orbit'):
    # criterion: orbit or neighbours
    # if orbit: get first k orbits, select all sequences from said orbits
    # if neighbours: set cutoff at the orbit that includes the first k neighs
    
    k_range = np.arange(k_min, k_max, k_step)
    # get max neighbours per level
    lvl_neighbours = [Neighbours(sorted_neigh, k_max) for sorted_neigh in sorted_neighbours]
    for lvl in lvl_neighbours:
        lvl.get_weights(k_range)
    
    grid_indexes = [(n_idx, k_idx) for n_idx in np.arange(len(lvl_neighbours)) for k_idx in np.arange(len(k_range))]
    classifs = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers = threads) as executor:
        future_classifs = {executor.submit(orbit_classify,
                                           lvl_neighbours[n_idx].positions,
                                           lvl_neighbours[n_idx].indexes,
                                           lvl_neighbours[n_idx].weighted[k_idx],
                                           lvl_neighbours[n_idx].double_weighted[k_idx],
                                           tax_tab): (n_idx, k_idx) for n_idx, k_idx in grid_indexes}
        for future in concurrent.futures.as_completed(future_classifs):
            cell = future_classifs[future]
            classifs[cell] = future.result()
            print(f'Done with cell {cell}')
    
    # return dictionary of the form (n_index, k_index):classifications
    return classifs
#%% classes
class Calibrator:
    def __init__(self, out_dir, warn_dir, prefix='calibration'):
        self.out_dir = out_dir
        self.classif_dir = out_dir + '/classification'
        self.warn_dir = warn_dir
        
        # make a directory to store classification reports
        os.makedirs(self.classif_dir, exist_ok=True)
        # prepare out files
        self.report_file = self.out_dir + f'/{prefix}.report'
        self.classif_file = self.out_dir + f'/{prefix}.classif'
        self.meta_file = self.out_dir + f'/{prefix}.meta'
        
        self.selector = fsele.Selector(out_dir, 'phylum class order family genus species'.split())
        self.loader = None
        
    @property
    def dist_mat(self):
        return self.__dist_mat
    @dist_mat.setter
    def dist_mat(self, mat_code):
        try:
            self.__dist_mat = cost_matrix.get_matrix(mat_code)
        except:
            raise
    
    @property
    def window_len(self):
        if hasattr(self, 'windows'):
            return self.windows[:,1] - self.windows[:,0]
    
    @property
    def ranks(self):
        return self.loader.tax_tab.columns.tolist()
    
    def set_database0(self, database):
        self.db = database
        try:
            self.db_dir = DATA.get_database(database)
        except Exception as excp:
            logger.error(excp)
            raise
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        self.guide_file = db_meta['guide_file'] # this isn't used in the calibration process, used in result visualization
        self.tax_ext = pd.read_csv(db_meta['expguide_file'], index_col=0)
        
        # load matrix & accession codes
        map_npz = np.load(db_meta['mat_file'])
        self.matrix = map_npz['matrix']
        self.max_pos = self.matrix.shape[1]
        with open(db_meta['acc_file'], 'r') as handle:
            self.accs = handle.read().splitlines()
        # build extended taxonomy
        tax_tab = pd.read_csv(db_meta['tax_file'], index_col=0).loc[self.accs]
        # the tax_tab parameter is the extended taxonomy for each record
        self.tax_tab = self.tax_ext.loc[tax_tab.TaxID.values]
        self.tax_tab.index = tax_tab.index
        
    def set_database(self, database):
        if not database in DATA.DBASES:
            print(f'Database {database} not found.')
            print('Current databases include:')
            for db, desc in DATA.DBASE_LIST.items():
                print(f'\tDatabase: {db} \t:\t{desc}')
            raise Exception('Database not found')
        self.db = database
        self.db_dir = DATA.DATAPATH + '/' + database
        # use meta file from database to locate necessary files
        with open(self.db_dir + '/meta.json', 'r') as meta_handle:
            db_meta = json.load(meta_handle)
        mat_file = db_meta['mat_file']
        tax_file = db_meta['tax_file']
        acc_file = db_meta['acc_file']
        order_file = db_meta['order_file']
        diff_file = db_meta['diff_file']
        self.guide_file = db_meta['guide_file'] # this isn't used in the calibration process, used in result visualization
        expguide_file = db_meta['expguide_file']
        
        # set the loader with the learning data
        self.loader = wn.WindowLoader('Graboid.calibrator.windowloader')
        self.loader.set_files(mat_file, acc_file, tax_file, expguide_file)
        self.max_pos = self.loader.dims[1]
        # load information files
        self.selector.load_order_mat(order_file)
        self.selector.load_diff_tab(diff_file)
        
        logger.info(f'Set database: {database}')
    
    def set_sliding_windows(self, size, step):
        if size >= self.max_pos:
            raise Exception(f'Given window size: {size} is equal or greater than the total length of the alignment {self.max_pos}, please use a smaller window size.')
        
        # adjust window size to get uniform distribution (avoid having to use a "tail" window)
        last_position = self.max_pos - size
        n_windows = int(np.ceil(last_position / step))
        w_start = np.linspace(0, last_position, n_windows, dtype=int)
        self.windows = np.array([w_start, w_start + size]).T
        self.w_type = 'sliding'
        logger.info(f'Set {n_windows} windows of size {size} at intervals of {w_start[1] - w_start[0]}')
    
    def set_custom_windows(self, starts, ends):
        # check that given values are valid: same length, starts < ends, within alignment bounds
        try:
            raw_coords = np.array([starts, ends], dtype=np.int).T
            raw_coords = np.reshape(raw_coords, (-1, 2))
        except ValueError:
            raise Exception(f'Given starts and ends lengths do not match: {len(starts)} starts, {len(ends)} ends')
        invalid = raw_coords[:, 0] >= raw_coords[:, 1]
        if invalid.sum() > 0:
            raise Exception(f'At least one pair of coordinates is invalid: {[list(i) for i in raw_coords[invalid]]}')
        out_of_bounds = ((raw_coords < 0) | (raw_coords >= self.max_pos))
        out_of_bounds = out_of_bounds[:,0] | out_of_bounds[:,1]
        if out_of_bounds.sum() > 0:
            raise Exception(f'At least one pair of coordinates is out of bounds [0 {self.max_pos}]: {[list(i) for i in raw_coords[out_of_bounds]]}')
        self.windows = raw_coords
        self.w_type = 'custom'
        logger.info(f'Set {raw_coords.shape[0]} custom windows at positions {[list(i) for i in raw_coords]} with lengths {[ln for ln in raw_coords[:,1] - raw_coords[:,0]]}')
    
    def grid_search0(self,
                             max_n,
                             step_n,
                             cost_mat,
                             row_thresh=0.2,
                             col_thresh=0.2,
                             min_seqs=50,
                             rank='genus',
                             min_n=5,
                             threads=1):
        print('Beginning calibration...')
        t0 = time.time()
        
        # collapse windows
        print('Collapsing windows...')
        collapsed_windows = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            future_windows = {executor.submit(wn.Window, self.matrix, self.tax_tab, win[0], win[1], row_thresh, col_thresh, min_seqs):idx for idx, win in enumerate(self.windows)}
            for future in concurrent.futures.as_completed(future_windows):
                ft_idx = future_windows[future]
                try:
                    collapsed_windows[ft_idx] = future.result()
                    logger.info(f'Window {ft_idx} {self.windows[ft_idx]}: collapsed into matrix of shape {future.result().window.shape}')
                except Exception as excp:
                    logger.info(f'Window {ft_idx} {self.windows[ft_idx]}: ' + str(excp))
                    continue
        t1 = time.time()
        print(f'Collapsed windows in {t1 - t0:.3f} seconds')
        
        # translate the collapsed_windows dict into an ordered list
        win_indexes = np.sort(list(collapsed_windows.keys()))
        win_list = [collapsed_windows[idx] for idx in win_indexes]
        
        # select sites
        print('Selecting informative sites...')
        window_sites = []
        for win in win_list:
            win_tax = self.tax_ext.loc[win.taxonomy][[rank]] # trick: if the taxonomy table passed to get_sorted_sites has a single rank column, entropy difference is calculated for said column
            sorted_sites = fsele.get_sorted_sites(win.window, win_tax) # remember that you can use return_general, return_entropy and return_difference to get more information
            window_sites.append(fsele.get_nsites(sorted_sites[0], min_n, max_n, step_n))
        t2 = time.time()
        print(f'Done in {t2 - t1:.3f} seconds')
        
        # calculate distances
        print('Calculating paired distances...')
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            all_win_dists = [dst for dst in executor.map(get_all_distances, [win.window for win in win_list], window_sites, [cost_mat]*len(win_list))]
        t3 = time.time()
        print(f'Done in {t3 - t2:.3f} seconds')


        # sort neighbours
        print('Sorting neighbours...')
        # sort the generated distance arrays
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            # for each window, we are getting the ascending ORDER for the distance calculated at each level
            sorted_dists = [dists for dists in executor.map(np.argsort, [win_d[0] for win_d in all_win_dists], [1]*len(win_list))] # win_d[0] contains the distances array, win_d[1] contains the paired indexes (used later)

        sorted_win_neighbours = []
        # sorted_win_neighbours is structured as:
            # window 0:
                # level 0:
                    # sorted neighbour idxs
                    # sorted neighbour dists
                    # both these arrays have shape n_rows * n_rows - 1, as each row has n_rows - 1 neighbours                    
        for idx, (win_dists, sorted_win_dists) in enumerate(zip(all_win_dists, sorted_dists)):
            sorted_idxs = [win_dists[1][:, lvl] for lvl in sorted_win_dists]
            sorted_distances = [dsts[lvl] for dsts, lvl in zip(win_dists[0], sorted_win_dists)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                sorted_win_neighbours.append([ordered for ordered in executor.map(get_sorted_neighs, sorted_idxs, sorted_distances)])
        t4 = time.time()
        print(f'Done in {t4 - t3:.3f} seconds')
        return sorted_win_neighbours, win_list
        
        # classify
        print('Classifying...')
        
        # get metrics
        # report
            
    def set_windows(self, size=np.inf, step=np.inf, starts=0, ends=np.inf):
        # this function establishes the windows to be used in the grid search
        # size and step establish the length and displacement rate of the sliding window
            # default values use the entire sequence (defined by w_start & w_end) in a single run
        # start and end define the scope(s) to analize
        # multiple values of starts & ends allow for calibration on multiple separated windows
            # default values use the entire sequence
        
        ends = np.clip(ends, 0, self.max_pos) # ensure all coord ends are within sequence limits
        try:
            raw_coords = np.array([starts, ends], dtype=np.int).T
            raw_coords = np.reshape(raw_coords, (-1, 2)) # do this in case a single coord was given
        except ValueError:
            raise Exception(f'Given starts and ends lengths do not match: {len(starts)} starts, {len(ends)} ends')
        
        # clip coordinates outside of boundaries and detect the ones that are flipped
        clipped = np.clip(raw_coords, 0, self.max_pos).astype(int)
        flipped = clipped[:,0] >= clipped[:,1]
        for flp in raw_coords[flipped]:
            logger.warning(f'Window {flp} is not valid')
        windows = clipped[~flipped]
        # establish the scope
        w_tab = []
        w_info = {} # each window contains a list [params, coords], params includes: start end size step
        for w_idx, (w_start, w_end) in enumerate(windows):
            scope_len = w_end - w_start
            if scope_len < size:
                # do a single window
                w_coords = np.array([[w_start, w_end]])
                w_size = scope_len
                w_step = scope_len
            else:
                w_step = min(step, size)
                w_size = size
                start_range = np.arange(w_start, w_end, w_step)
                end_range = start_range + w_size
                w_coords = np.array([start_range, end_range]).T
                # clip windows
                w_coords = w_coords[end_range <= w_end]
                if w_coords[-1, 1].T > w_end:
                    # add a tail window, if needed, to cover the entire sequence
                    w_coords = np.append(w_coords, [[w_end - size, w_end]], axis=0)
            
            w_tab.append(pd.DataFrame(w_coords, columns = 'start end'.split(), index = [w_idx for i in w_coords]))
            w_info[f'w_{w_idx}'] = {'params':[int(param) for param in [w_start, w_end, w_size, w_step]],
                                    'coords':w_coords.tolist()}
        self.w_coords = pd.concat(w_tab)
        self.w_info = w_info
        logger.info(f'Set {len(w_coords)} windows of size {size} and step {step}')
    
    def grid_search(self,
                    max_k,
                    step_k,
                    max_n,
                    step_n,
                    min_seqs=10,
                    rank='genus',
                    row_thresh=0.2,
                    col_thresh=0.2,
                    min_k=1,
                    min_n=5,
                    threads=1,
                    keep_classif=False,
                    log_report=False):
        
        # k & n ranges
        k_range = np.arange(min_k, max_k, step_k)
        n_range = np.arange(min_n, max_n, step_n)
        
        # register report metadata
        meta = {'k':k_range.tolist(),
                'n':n_range.tolist(),
                'db':self.db,
                'guide': self.guide_file,
                'ranks':self.ranks,
                'windows':self.w_info,}
        with open(self.meta_file, 'w') as meta_handle:
            json.dump(meta, meta_handle)
            
        # begin calibration
        print('Beginning calibration...')
        t00 = time.time()
        for idx, (start, end) in enumerate(self.w_coords.to_numpy()):
            t0 = time.time()
            print(f'Window {start} - {end} ({idx + 1} of {len(self.w_coords)})')
            # extract window and select atributes
            window, window_cols, window_tax = self.loader.get_window(row_thresh, col_thresh, start=start, end=end)
            n_seqs = len(window)
            if n_seqs == 0:
                # no effective sequences in the window
                continue
            elif n_seqs < min_seqs:
                # not enough sequences passed the filter, skip iteration
                logger.info(f'Window {start} - {end}. Not enough sequences to perform calibration ({n_seqs}, min = {min_seqs}), skipping')
                continue
            
            n_sites = self.selector.get_sites(n_range, rank, window_cols)
            y = self.loader.tax_guide.loc[window_tax] # y holds the complete taxonomy of each sequence in the collapsed window, uses the expanded taxonomy guide
            # distance container, 3d array, paired distance matrix for every value of n
            dist_mat = np.zeros((n_seqs, n_seqs, len(n_range)), dtype=np.float32)
            # get paired distances
            t1 = time.time()
            logger.debug(f'prep time {t1 - t0:.3f}')
            for idx_0 in np.arange(n_seqs - 1):
                qry_seq = window[[idx_0]]
                idx_1 = idx_0 + 1
                ref_seqs = window[idx_1:]
                # persistent distance array, updates with each value of n
                dists = np.zeros((1, ref_seqs.shape[0]), dtype=np.float32)
                for n_idx, n in enumerate(n_range):
                    try:
                        sites = n_sites[n]
                        sub_qry = qry_seq[:, sites]
                        sub_ref = ref_seqs[:, sites]
                        dists += classification.get_dists(sub_qry, sub_ref, self.dist_mat).reshape(1, -1)
                    except KeyError:
                        # no new sites for n
                        pass
                    dist_mat[idx_0, idx_1:, n_idx] = dists
                    dist_mat[idx_1:, idx_0, n_idx] = dists # is this necessary? (yes), allows sortying of distances in a single step
            # fill the diagonal values with infinite value, this ensures they are never among the k neighs
            for i in range(len(n_range)):
                np.fill_diagonal(dist_mat[:,:,i], np.inf)
            t2 = time.time()
            logger.debug(f'dist calculation {t2 - t1:.3f}')
            # get ordered_neighbours and sorted distances
            neighbours = np.argsort(dist_mat, axis=1)
            ordered_dists = [dist_mat[np.tile(np.arange(n_seqs), (n_seqs, 1)).T, neighbours[...,n], n] for n in range(neighbours.shape[2])]
            
            guide = [(n, mode, classif) for mode, classif in classification.classif_funcs_nb.items() for n in range(len(n_range))]
            classif_report = []
            # use multiprocessing to speed up classification
            t3 = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                # since we're using numba functions, y must be cast as a numpy array
                future_classifs = {executor.submit(classifier, neighbours[...,n], ordered_dists[n], y.to_numpy(), k_range):(mode,n) for (n, mode, classifier) in guide}
                for future in concurrent.futures.as_completed(future_classifs):
                    pre_classif, columns = future.result()
                    mode, n = future_classifs[future]
                    classif = classification.get_classif(pre_classif, classification.classif_modes[mode])
                    mode_report = pd.DataFrame(classif, columns=columns)
                    mode_report['mode'] = mode
                    mode_report['n'] = n_range[n]
                    classif_report.append(mode_report)
            classif_report = pd.concat(classif_report)
            uniq_idxs = classif_report.idx.sort_values().unique()
            t4 = time.time()
            logger.debug(f'classification {t4 - t3:.3f}')
            # store intermediate classification results (if enabled)
            if keep_classif:
                classif_file = self.classif_dir + f'/{start}-{end}_{n_seqs}.classif'
                classif_report.to_csv(classif_file, index=False)
                # store table with real values as well
                y.loc[uniq_idxs].to_csv(self.classif_dir + f'/{start}-{end}_{n_seqs}.real')
            # get classification metrics
            t5 = time.time()
            classification_table = classif_report.set_index(['_k', 'n', 'mode'])
            tax_matrix = y.loc[uniq_idxs].to_numpy()
            build_report(classification_table, tax_matrix, start, end, self.report_file, log_report)
            #
            t6 = time.time()
            logger.debug(f'metric calculation {t6 - t5:.2f}')
            logger.info(f'Window {start} - {end} ({n_seqs} effective sequences) Calibrated in {t6 - t0:.2f} seconds')
        elapsed = time.time() - t00
        logger.info(f'Finished calibration in {elapsed:.2f} seconds')
        logger.info(f'Stored calibration report to {self.report_file}')
        if keep_classif:
            mem, unit = get_classif_mem(self.classif_dir)
            logger.info(f'Stored classification results to {self.classif_dir}, using {mem:.2f} {unit}')
        else:
            os.rmdir(self.classif_dir)
    
    def build_summaries(self):
        # run this function to build the summary files after running the grid search
        print('Building summary files')
        cal_report = pd.read_csv(self.report_file)
        out_files = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1_score']:
            print(f'Building {metric} summary')
            score_file = f'{self.out_dir}/{metric}_score.summ'
            param_file = f'{self.out_dir}/{metric}_param.summ'
            out_files[metric] = [score_file, param_file]
            rk_tab, param_tab = rp.make_summ_tab(cal_report, metric)
            rk_tab.to_csv(score_file)
            param_tab.to_csv(param_file)
            logger.info(f'Stored score summary to {score_file}')
            logger.info(f'Stored param summary to {param_file}')
        return out_files
