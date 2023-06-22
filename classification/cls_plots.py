#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:16:23 2023

@author: hernan
Build plots for Graboid classify
"""

#%% libraries
import matplotlib.pyplot as plt
import numpy as np
#%% functions
def plot_ref_v_qry(ref_coverage, ref_mesas, qry_coverage, qry_mesas, overlapps, figsize=(12,7)):
    x = np.arange(len(ref_coverage)) # set x axis
    # plot ref
    fig, ax_ref = plt.subplots(figsize = figsize)
    ax_ref.plot(x, ref_coverage, label='Reference coverage')
    for r_mesa in ref_mesas.astype(int):
        mesa_array = np.full(len(ref_coverage), np.nan)
        # columns in the mesa arrays are [mesa start, mesa end, mesa width, mesa average height]
        mesa_array[r_mesa[0]:r_mesa[1]] = r_mesa[3]
        ax_ref.plot(x, mesa_array, c='r')
    
    # plot qry
    ax_qry = ax_ref.twinx()
    ax_qry.plot(x, qry_coverage, c='tab:orange')
    for q_mesa in qry_mesas.astype(int):
        mesa_array = np.full(len(ref_coverage), np.nan)
        # columns in the mesa arrays are [mesa start, mesa end, mesa width, mesa average height]
        mesa_array[q_mesa[0]:q_mesa[1]] = q_mesa[3]
        ax_qry.plot(x, mesa_array, c='g')
    
    # # plot overlaps
    # # vertical lines indicating overlapps between query and reference mesas
    for ol in overlapps:
        ol_height = max(ol[4], (ol[3] / ref_coverage.max()) * qry_coverage.max())
        
        ax_qry.plot(ol[[0,0]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
        ax_qry.plot(ol[[1,1]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
    # ol_x = overlapps[:, [0,0,1,1]].flatten() # each overlap takes two times the start coordinate and two times the end coordinate in the x axis (this is so they can be plotted as vertical lines)
    # ol_rheight = (overlapps[:, 3] / ref_coverage.max()) * qry_coverage.max() # transform the reference mesa height to the scale in the query axis
    # ol_y = np.array([overlapps[:,4], ol_rheight, overlapps[:,4], ol_rheight]).T.flatten() # get the ref and qry height of each overlape TWICE and interloped so we can plot th evertical lines at both ends of the overlap
    # ax_qry.plot(ol_x, ol_y, c='k')
    
    ax_ref.plot([0], [0], c='r', label='Reference mesas')
    ax_ref.plot([0], [0], c='tab:orange', label='Query coverage')
    ax_ref.plot([0], [0], c='g', label='Query mesas')
    ax_ref.plot([0], [0], linestyle=':', c='k', label='Overlapps')
    ax_ref.legend()
    # TODO: fix issues with mesa calculations
    # only filter out overlap coordinates when they appear too closely together (<= 20 sites) in the x axis
    ol_coords = np.unique(overlapps[:,:2])
    ol_coor_diffs = np.diff(ol_coords)
    selected_ol_coords = ol_coords[np.insert(ol_coor_diffs > 20, 0, True)]
    ax_qry.set_xticks(selected_ol_coords)
    ax_ref.set_xticklabels(selected_ol_coords.astype(int), rotation=70)
    
    ax_ref.set_xlabel('Coordinates')
    ax_ref.set_ylabel('Reference coverage')
    ax_qry.set_ylabel('Query coverage')
    
    # TODO: save plot

def plot_result():
    # make a pie chart of sample characterization
        # each slice is a taxon
        # slices subdivided into branches (width proportional to sequences in branch)
        # each subslice has a darker color subsection representing support for the winning taxon (the closer to the edge of the pie, the more support, the dark section in subslices with 100% support has the same radius as the pie)
            # draw a black circle of half the pie radius to represent the support threshold (>= half support, there is another taxon that has the same support as the so called "winner")
    return
