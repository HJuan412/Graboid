#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:04:30 2021

@author: hernan
Results visualizer
"""
import matplotlib.pyplot as plt
import numpy as np
confusion = ''
total_true = ''
taxons = ''
#%%
fig, ax = plt.subplots(figsize = (12, 12))
ax.imshow(confusion / total_true.reshape(1,-1).T)

# TODO translate idcodes
ax.set_xticks(np.arange(len(taxons)))
ax.set_yticks(np.arange(len(taxons)))
ax.set_xticklabels(taxons, rotation = 45)
ax.set_yticklabels(taxons)
for idx0, x in enumerate(confusion):
    for idx1, y in enumerate(x):
        ax.text(idx0, idx1, int(confusion[idx1, idx0]), ha ='center', va = 'center')