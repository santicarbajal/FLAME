__author__ = 'Santi'
# ###
#CLOSED. May, 15, 2017
# ###
# CsvHandler.py
# ###
"""CsvHandler.py
Provide a simple class for opening cvs  files, loading them into pandas dataframes"""
import myconstants
# ###
# import modules and read  data file
# ###
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import shuffle
# ###
#  END OF IMPORTS
# ###

class CsvHandler(object):
    'csv File Handler Class'
    className = "CsvHandler"
    classCounter = 0
    # -----------------------------------------
    def __init__(self):
        CsvHandler.classCounter += 1
        #print "I am the constructor inside class ", CsvHandler.__doc__
        #print "Total number of ", self.className, " objects  is ", CsvHandler.classCounter
        self.__data = pd.read_csv(myconstants.PROCESSEDDATA, index_col=myconstants.INDEX_COL)  # reading the data
        self.__fileName = myconstants.PROCESSEDDATA
        self.__indexCol = myconstants.INDEX_COL
        self.__columnNames = list(self.__data.columns.values)
        self.__listOfClasses =list(self.getData().columns.values)[- myconstants.TOBEPREDICTED:]
        self.reportDataFrame(myconstants.VERBOSITY,myconstants.DIRECTION)
        #self.plotDataFrame()
        #self.reportMissingValues()
        
    # -----------------------------------------


    # -----------------------------------------
    def getData(self):
        return self.__data
    # -----------------------------------------


    # -----------------------------------------
    def getFileName(self):
        return self.__fileName
    # -----------------------------------------


    # ----------------------------------------- 
    def getIndexCol(self):
        return self.__indexCol
    # -----------------------------------------


    # -----------------------------------------
    def getColumnNames(self):
        return self.__columnNames
    # ----------------------------------------- 


    # -----------------------------------------
    def getListOfClasses(self):
        return self.__listOfClasses
    # ----------------------------------------- 


    # -----------------------------------------
    def __str__(self):
        return "I am something " + " read from file " + self.getFileName() + "\n" 
    # ----------------------------------------- 


    # -----------------------------------------
    def plotDataFrame(self):
        """
            Shows Histogram 
        """
        self.getData().hist()
        plt.show()
    # -----------------------------------------


    # -----------------------------------------
    def reportMissingValues(self):
        """
            Counts the number of missing values inside dataframe
        """
        print "Missing values per column:"
        print self.getData().apply(self.num_missing, axis=myconstants.DIRECTION)
    # -----------------------------------------


    # -----------------------------------------
    def reportDataFrame(self,verbosity, direction):
        """
            Counts the number of fields inside dataframe
            0: applying on each column
            1: applying on each row
        """    
        if verbosity:
            df.head()
        print len (self.getColumnNames()), " Columns Found inside "
    # -----------------------------------------


    # -----------------------------------------
    def num_missing(self,x):
        """
            Any function to apply over the whole dataframe. 
            return: number of null values in column x
            params: x             
        """
        return sum(x.isnull()) 
    # -----------------------------------------

    





