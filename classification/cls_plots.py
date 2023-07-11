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
def plot_ref_v_qry(ref_coverage, ref_mesas, qry_coverage, qry_mesas, overlaps, figsize=(12,7), **kwargs):
    # kwargs:
        # ref_title: name of the reference database
        # qry_title: name of the query sequence file
        # out_file: name of the output file
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
    for ol in overlaps:
        ol_height = max(ol[4], (ol[3] / ref_coverage.max()) * qry_coverage.max())
        
        ax_qry.plot(ol[[0,0]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
        ax_qry.plot(ol[[1,1]], [0, ol_height], linestyle=':', linewidth=1.5, c='k')
    
    ax_ref.plot([0], [0], c='r', label='Reference mesas')
    ax_ref.plot([0], [0], c='tab:orange', label='Query coverage')
    ax_ref.plot([0], [0], c='g', label='Query mesas')
    ax_ref.plot([0], [0], linestyle=':', c='k', label='Overlapps')
    ax_ref.legend()
    # TODO: fix issues with mesa calculations
    # only filter out overlap coordinates when they appear too closely together (<= 20 sites) in the x axis
    ol_coords = np.unique(overlaps[:,:2])
    ol_coor_diffs = np.diff(ol_coords)
    selected_ol_coords = ol_coords[np.insert(ol_coor_diffs > 20, 0, True)]
    ax_qry.set_xticks(selected_ol_coords)
    ax_ref.set_xticklabels(selected_ol_coords.astype(int), rotation=70)
    
    ax_ref.set_xlabel('Overlaps coordinates')
    ax_ref.set_ylabel('Reference coverage (reads)')
    ax_qry.set_ylabel('Query coverage (reads)')
    
    # Add plot title
    if 'ref_title' in kwargs.keys() and 'qry_title' in kwargs.keys():
        # only add title if both ref_title and qry_title kwargs are present
        ax_ref.set_title(kwargs['ref_title'] + '(reference) coverage\nvs\n' + kwargs['qry_title'] + '(query) coverage')
    
    # Save plot
    if 'out_file' in kwargs.keys():
        fig.savefig(kwargs['out_file'])

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
    
def plot_result(report, confidence_threshold=0.5, figsize=10):
    # make a pie chart of sample characterization
        # each slice is a taxon
        # slices subdivided into branches (width proportional to sequences in branch)
        # each subslice has a darker color subsection representing support for the winning taxon (the closer to the edge of the pie, the more support, the dark section in subslices with 100% support has the same radius as the pie)
            # draw a black circle of half the pie radius to represent the support threshold (>= half support, there is another taxon that has the same support as the so called "winner")
    # report is adataframe with columns: Taxon(name, not id), support, n_seqs
    # report is grouped by lineage, each taxon is sorted by support (descending)
    # figsize determines the pie diameter
    
    def rotate(vector, angle):
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        vector2 = np.dot(rot, vector)
        return vector2
    
    def deg2rad(angle):
        return angle/360 * 2 * np.pi    
    
    def get_hatches(tax_idxs):
        template_idxs = np.repeat(np.arange(10), 10)
        n_repeats = np.ceil(len(tax_idxs) / 100).astype(int)
        
        hatch_idxs = np.concatenate(([0]*10, np.tile(template_idxs, n_repeats)))[tax_idxs]
        hatch_densities = (np.clip(tax_idxs - 10, 0, np.inf) / 100).astype(int)
        hatch_densities[10:] += 1
        
        hatch_styles = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
        hatches = [hatch_styles[hatch_idx] * hatch_density for hatch_idx, hatch_density in zip(hatch_idxs, hatch_densities)]
        return hatches
    
    # preprocess report, filter by confidence
    report = report.loc[report.norm_support > confidence_threshold].copy()
    report = report.sort_values(['norm_support', 'n_seqs'], ascending=False).reset_index(drop=True)
    # TODO: retrieve lineage codes
    # TODO: infer figsize from total seqs
    # TODO: merge clusters of a same taxon with the same support
    
    cmap = cm.get_cmap('tab10')
    center = np.array([figsize, figsize]) / 2
    radius = figsize / 2
    start = 90 # pie chart initial position is at 12 o clock. Angle 0 is horizontal to the right, moves counterclockwise
    
    # add arcs column to report
    total_seqs = report.n_seqs.sum()
    report['arc'] = (report.n_seqs.to_numpy() / total_seqs) * 360
    
    # get taxon indexes
    uniq_taxes = report.tax.unique()
    tax_idxs = np.arange(len(uniq_taxes)) # should use the order taken by sorting by lineage codes
    
    # set taxon colors and hatching styles
    color_idxs = tax_idxs % 10
    hatch_idxs = np.floor(tax_idxs / 10).astype(int)
    tax_hatches = get_hatches(tax_idxs)
    tax_colors = [cmap(color_idx) for color_idx in color_idxs]
    
    # build wedges
    wedges = [] # outer circle, shows the number of sequences in the branch, same radii
    wedges_sec = [] # secondary circle, shows support for each branch, variable radii
    tax_positions = [] # this list determines the start position of each taxon
    legend_reprs = [] # this list holds a representative (sec)wedge for each taxon, used to build legend
    for tax_color, tax_hatch, tax in zip(tax_colors, tax_hatches, uniq_taxes):
        tax_arcs = report.loc[report.tax == tax, ['arc', 'norm_support']].to_numpy()
        tax_positions.append(len(wedges))
        for arc, supp in tax_arcs:
            theta = np.sort([start, start-arc]) # theta1 should always be smaller than theta 2
            wedges.append(ptch.Wedge(center, radius, theta[0], theta[1], facecolor=tax_color, alpha=0.5, hatch=tax_hatch)) # outer circle is clearer than the inner one, set alpha to 0.5
            wedges_sec.append(ptch.Wedge(center, radius*supp, theta[0] - 0.05, theta[1], facecolor=tax_color, hatch=tax_hatch)) # inner circle, raduis equals support
            start -= arc # displace start position to the end of the current wedge
        legend_reprs.append(wedges_sec[-1]) # add current taxon representative
    
    # build thresh indicator
    thresh_circle = ptch.Circle(center, radius = radius * confidence_threshold, fill = False, linewidth = 1.5) # Draw a circle indicating the confidence threshold # TODO: make line width adjust dynamically to figsize
    
    # build figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    
    # add wedges
    for w in wedges:
        ax.add_patch(w)
    for w2 in wedges_sec:
        ax.add_patch(w2)
    
    # draw confidence threshold
    ax.add_patch(thresh_circle)
    ax.legend(legend_reprs, uniq_taxes) # TODO: move legend to a different subplot
    # TODO: add annotations
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # TODO: add title
    # TODO: save figures
    return
