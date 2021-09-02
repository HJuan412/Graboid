#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:53:55 2021

@author: hernan
"""

from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
from glob import glob
import pandas as pd
#%% sequence manipulation
def make_seqdict(file):
    seqdict = {}
    with open(file, 'r') as handle:
        for header, seq in sfp(handle):
            acc = header.split(' ')[0]
            seqdict[acc] = seq
    
    return seqdict

#%% file processing

class filetab():
    # list files in directory, presents information taken from the filenames
    def __init__(self, dirname, pattern, sep, tail = None):
        """
        Detects files in the given directory, builds preliminar dataframe
        
        Parameters
        ----------
        dirname : str
            Directory containing files, don't include final '/'.
        pattern : str
            Search pattern to look within the directory.
        sep : str
            Field separator in the filenames.
        tail : str
            Format suffix of the files. The default is None.

        Returns
        -------
        None.

        """
        self.dir = f'{dirname}/'
        self.make_filelist(pattern)
        self.tail = tail
        self.make_main_df(sep)
        self.reset_columns()
    
    def make_filelist(self, pattern):
        """
        Looks for pattern in the given file, stores a list with results.

        Parameters
        ----------
        pattern : str
            Search pattern to look within the directory.

        Returns
        -------
        None.

        """
        filelist = glob(f'{self.dir}{pattern}')
        filelist = [file.split('/')[-1] for file in filelist]
        self.filelist = filelist
    
    def crop_tail(self):
        """
        Remove format suffix from the files

        Returns
        -------
        cropped : list
            List of filenames without the format suffix.

        """
        cropped = [file.split(self.tail)[0] for file in self.filelist]
        return cropped

    def make_main_df(self, sep):
        """
        Build preliminary dataframe using the given separtator

        Parameters
        ----------
        sep : str
            Field separator in the filenames.

        Returns
        -------
        None.

        """
        nrows = len(self.filelist)
        ncols = len(self.filelist[0].split(sep))
        
        main_df = pd.DataFrame(index = range(nrows), columns = range(ncols))
        
        if self.tail is None:
            filelist = self.filelist
        else:
            filelist = self.crop_tail()

        for idx, file in enumerate(filelist):
            main_df.at[idx,:] = file.split(sep)
        self.main_df = main_df
    
    def split_col(self, ssep, index, subcols = None):
        """
        Split a column using a given secondary separator. Store generated subcolumns in a dataframe.

        Parameters
        ----------
        ssep : str
            Secondary field separator.
        index : int
            Column to split.
        subcols : list, optional
            Names to be given to the generated subcolumns. The default is None.

        Returns
        -------
        None.

        """
        col = self.main_df.loc[:, index].to_list()
        col = [row.split(ssep) for row in col]
        
        if subcols is None:
            subcols = range(len(col[0]))
        
        for idx, scol in enumerate(subcols):
            scol_values = [row[idx] for row in col]
            self.columns[scol] = scol_values
    
    def rename_col(self, oldname, newname):
        self.columns.rename({oldname:newname}, inplace = True)
    
    def reset_columns(self):
        self.columns = pd.DataFrame(index = range(len(self.filelist)))
        self.columns['File'] = self.filelist
    
    def add_column(self, index, colname):
        """
        Add column of main_df to the columns dataframe, specify new name of the column

        Parameters
        ----------
        index : int
            Column index.
        colname : str
            New column name.

        Returns
        -------
        None.

        """
        col = self.main_df.loc[:,index]
        self.columns[colname] = col
    
    def set_coltype(self, column, coltype):
        self.columns[column] = self.columns[column].astype(coltype)

    def build_filetab(self, cols):
        self.filetab = self.columns.loc[:, cols]