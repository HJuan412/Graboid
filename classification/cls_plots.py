#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:16:23 2023

@author: hernan
Build plots for Graboid classify
"""

#%% libraries
from matplotlib import cm
import matplotlib.patches as ptch
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

def plot_sample_report(report, figsize=10):
    # build pie charts for sample characterization reports
    def preprocess(report):
        # split result by taxonomic rank, turn table into arrays with columns: taxon, n_seqs, supp
        ranks = report.colums.get_level_values(0)[:-1]
        rk_arrays = {}
        
        for rk in ranks:
            # sort matches by Support (decreasing order)
            # sort by lineage codes (Groups matches by taxon)
            rk_subtab = report[[rk, 'nseqs']]
            rk_arrays[rk] = rk_subtab.sort_values('support', ascending=False).sort_values('LinCode').drop('LinCode')
        return rk_arrays
    
    rk_arrays = preprocess(report)
    
    for rk, rk_report in rk_arrays.items():
        plot_result(rk_report, figsize)
    
def plot_result(report, figsize=10):
    # make a pie chart of sample characterization
        # each slice is a taxon
        # slices subdivided into branches (width proportional to sequences in branch)
        # each subslice has a darker color subsection representing support for the winning taxon (the closer to the edge of the pie, the more support, the dark section in subslices with 100% support has the same radius as the pie)
            # draw a black circle of half the pie radius to represent the support threshold (>= half support, there is another taxon that has the same support as the so called "winner")
    # report is adataframe with columns: Taxon(name, not id), support, n_seqs
    # report is grouped by lineage, each taxon is sorted by support (descending)
    # figsize determines the pie diameter
    
    cmap = cm.get_cmap('gist_rainbow')
    
    def rotate(vector, angle):
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        vector2 = np.dot(rot, vector)
        return vector2
    
    def deg2rad(angle):
        return angle/360 * 2 * np.pi    
    
    center = np.array([figsize, figsize]) / 2
    radius = figsize / 2
    start = 90 # pie chart initial position is at 12 o clock. Angle 0 is horizontal to the right, moves counterclockwise
    
    # add arcs column to report
    total_seqs = report.n_seqs.sum()
    report['arc'] = (report.n_seqs.to_numpy() / total_seqs) * 360
    
    # set colors
    uniq_taxes = report.Taxon.unique()
    tax_colors = {tax: tax_idx/len(uniq_taxes) for tax_idx, tax in enumerate(uniq_taxes)} # assign a color for each taxon
    
    # build wedges
    wedges = [] # outer circle, shows the number of sequences in the branch, same radii
    wedges_sec = [] # secondary circle, shows support for each branch, variable radii
    for idx, row in report.iterrows():
        theta = np.sort([start, start-row.arc]) # theta1 should always be smaller than theta 2
        tax_color = tax_colors[row.Taxon]
        wedges.append(ptch.Wedge(center, radius, theta[0], theta[1], color=cmap(tax_color), alpha=0.5)) # outer circle is clearer than the inner one, set alpha to 0.5
        wedges_sec.append(ptch.Wedge(center, radius*row.support, theta[0], theta[1], color=cmap(tax_color)))
        start -= row.arc # displace start position to the end of the current wedge
    
    # build separators
    separators = [] # radial lines that separate the wedges
    vector = np.array([0, radius]) # original vector points at 12 o clock
    for arc in report.arc.values():
        separators.append(np.array([center, center+vector]))
        vector = rotate(vector, deg2rad(-arc)) # rotate vector to the end position of the current wedge
    # get main separators, separators between taxa should be a little thiccer
    _tax, tax_positions = np.unique(report.Taxon, return_index=True)
    
    # build figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    
    # add wedges
    for w in wedges:
        ax.add_patch(w)
    for w2 in wedges_sec:
        ax.add_patch(w2)
    
    # draw separators
    for sep in separators:
        ax.plot(sep[:,0], sep[:,1], color='w', linewidth=1)
    for tx in tax_positions:
        tax_sep = separators[tx]
        ax.plot(tax_sep[:,0], tax_sep[:,1], color='w', linewidth=2)
    
    # TODO: add labels
    # hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # TODO: add legend
    # TODO: add title
    # TODO: save figures
    return
