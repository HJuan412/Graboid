#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:40:50 2023

@author: hernan
This script is a tentative improvement for classification efficiency (reducing the number of operations)
"""
import numpy as np

dist_mat = np.array([[0,1,2,1],
                     [1,0,2,1],
                     [2,2,0,1],
                     [1,1,1,0]])

def sort_branches(branches, distances):
    sorted_idx = np.argsort(distances).flatten()
    sorted_dists = distances[sorted_idx]
    sorted_branches = [branches[idx] for idx in sorted_idx]
    n_seqs = np.cumsum([branch.nseqs for branch in sorted_branches])
    return sorted_branches, sorted_dists, n_seqs

class Tree:
    def __init__(self, matrix, cols):
        # set attrs
        self.matrix = matrix
        self.seqs = np.arange(len(matrix))
        self.cols = cols
        self.max_level = len(cols) - 1
        self.levels = [[] for lvl in self.cols]
        # build nodes
        Node(self, 0, None, self.seqs)
    
    def get_distances(self, seq, k=1, check_every=5):
        # calculate distance between sequence and the branches of the tree
        # seq : query sequence (whole, appropriate sites will be selected by cols)
        # k : neighbour threshold (ignore the rows further than the kth neighbour)
        # check_every : how often the tree is checked for prunning
        
        # initialize branches and distances (root branch, distance 0)
        branches = [self.levels[0]]
        distances = np.zeros(1)
        # explore the tree by level, not by branch
        for idx, lvl in enumerate(self.levels[1:]):
            col = self.cols[idx]
            branch_list = []
            dist_list = []
            # get the distance from each branch to the sequence
            for br, dist in zip(branches, distances):
                new_branches, new_dists = br.get_distance(seq[col], dist)
                branch_list += new_branches
                dist_list += new_dists
            dist_list = np.array(dist_list)
            # prune branches (delete those that have no chance of entering the k nearest neighbourhood)
            if (idx + 2) % check_every == 0:
                # sort branches by distance, get branch that contains the k nearest neighbour (branch_k)
                branch_list, dist_list, n_seqs = sort_branches(branch_list, dist_list)
                branch_k = np.argmax(n_seqs >= k)
                # get threshold, distance from k_branch to seq + max theoric distance - min theoric distance
                # max theoric distance : maximum possible distance increase of branch_k for the remaining sequence
                # min theoric distance : minimum possible distance increase of branch_k for the remaining sequence
                # branches whose current distance is beyond the trheshold won't pass branch_k for the remaining sequence and can be discarded (prunned)
                threshold = dist_list[branch_k] + (dist_mat.max() - dist_mat.min()) * (self.max_level - 1 - idx)
                break_point = np.argmax(dist_list > threshold)
                branch_list = branch_list[:break_point]
                dist_list = dist_list[:break_point]
            branches = branch_list
            distances = dist_list
        # sort final branch list by distance, establish cutoff at branch_k
        branches, distances, n_seqs = sort_branches(branches, distances)
        branch_k = np.argmax(n_seqs > k)
        # get results, list of ref sequences and their distances to seq
        results = []
        for br, ds in zip(branches[:branch_k], distances[:branch_k]):
            results += [[sq, ds] for sq in br.seqs]
        return results

class Node:
    def __init__(self, tree, level, parent, seqs):
        # set attributes
        self.tree = tree
        self.level = level
        self.parent = parent
        self.seqs = seqs
        self.nseqs = len(seqs)
        # add self to tree
        self.tree.levels[level].append(self)
        # set children
        self.branches = {}
        if level < tree.max_level:
            vals = np.unique(tree.matrix[seqs, level])
            for val in vals:
                val_seqs = np.argwhere(tree.matrix[seqs, level] == val).flatten()
                self.branches[val] = Node(tree, level + 1, self, val_seqs)
                
    def get_distance(self, val, dist):
        # return a list of every children + its distance to value
        branch_list = []
        dist_list = []
        for branch_val, branch in self.branches.values():
            # TODO: replace dist_mat for the real deal
            branch_dist = dist_mat[val, branch_val] + dist
            branch_list.append(branch)
            dist_list.append(branch_dist)
        return branch_list, dist_list
