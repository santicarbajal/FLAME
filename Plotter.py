__author__ = 'Santi'
# ###
# CLOSED. May, 15, 2017
# ###
# Plotter.py
# ###
"""Plotter.py
Provide a simple class for plotting csv  files"""
import myconstants
import os

# ###
# import modules and read  data fileb
# ###
from CsvHandler import *
import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn.svm import *
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from pandas.tools.plotting import scatter_matrix

# ###
# MLPClassifier is not yet available in scikit-learn 
# v0.17 (as of 1 Dec 2015). If you really want to use 
# it you could clone 0.18dev (however, I don't know how 
# stable this branch currently is).
# from sklearn.neural_network import MLPClassifier
# ###
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from random import shuffle

#import xgboost as xgb
from sklearn.grid_search import GridSearchCV

# ###
#  END OF IMPORTS
# ###

class Plotter(object):
    'csv PlotterClass Class'
    className = "Plotter"
    classCounter = 0
    # -----------------------------------------
    def __init__(self):
        Plotter.classCounter += 1
        #print "I am the  plotter constructor inside class ", Plotter.__doc__
        #print "Total number of ", self.className, " objects  is ", Plotter.classCounter
        self.__dataFrame = CsvHandler() # Reads data from processeddata.cvs, or any other place stated in PROCESSEDDATA      myconstants.PROCESSEDDATA     "/home/scarbajal/Desktop/FLAME/processeddata.csv"
        #self.__dataFrame.reportDataFrame(myconstants.VERBOSITY,myconstants.DIRECTION)
        #self.__dataFrame.plotDataFrame()
        #self.__dataFrame.reportMissingValues()                
    # -----------------------------------------


    # -----------------------------------------
    def getDataFrame(self):
        """
            Accessor.
        """
        return self.__dataFrame 
    # ----------------------------------------- 


    # -----------------------------------------
    def __str__(self):
        """
            .
        """
        return "I am something " + " read from file " + self.__dataFrame.getFileName() + " Plus  some plotting Functions" +"\n" 

    # ----------------------------------------- 


    # -----------------------------------------
    def showCorrelation(self, labels, value):
        """
        Correlation gives an indication of how related
        the changes are between two variables. If two
        variables change in the same direction they are
        positively correlated. If the change in opposite
        directions together (one goes up, one goes down),
        then they are negatively correlated.

        You can calculate the correlation between each 
        pair of attributes. This is called a correlation 
        matrix. You can then plot the correlation matrix
        and get an idea of which variables have a high 
        correlation with each other.

        This is useful to know, because some machine
        learning algorithms like linear and logistic 
        regression can have poor performance if 
        there are highly correlated input variables 
        in your data.
        """
        indexes  = self.getPositionsFromLabels(labels, value)

        cutPoint =  int  (   (len(self.__dataFrame.getData().columns.tolist()) * myconstants.RELEVANTFEATURES) )
        dfShort =  self.__dataFrame.getData().ix[:,3:cutPoint].iloc[indexes].copy() # To avoid the case where changing dfShort also changes df  
        correlations = dfShort.corr()
        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = plt.gca()
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,cutPoint,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        # You can specify a rotation for the tick labels in degrees or with keywords.
        ax.set_xticklabels(self.__dataFrame.getColumnNames()[3:cutPoint],rotation='vertical')
        ax.set_yticklabels(self.__dataFrame.getColumnNames()[3:cutPoint])
        
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)
        # Tweak spacing to prevent clipping of tick-labels
        #plt.subplots_adjust(bottom=0.15, top = 0.2)
        # Option 1
        # QT backend
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()

        # Option 2
        # TkAgg backend
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        # Option 3
        # WX backend
        #manager = plt.get_current_fig_manager()
        #manager.frame.Maximize(True)
  
        #plt.show()
        fileToSave = myconstants.PNGSPATH + '/'+ '0' + str(value) + '/'  + 'correlation' + str(value) +'.png'
        self.ensure_dir(fileToSave)
        fig.savefig(fileToSave)
        #plt.clf() 
        plt.close(fig)        
    # -----------------------------------------


    # -----------------------------------------
    def plotDataFrame(self):
        """
           Shows histogram for a Dataframe .
        """
               
        cutPoint =  int  (   (len(self.__dataFrame.getData().columns.tolist()) * myconstants.RELEVANTFEATURES) )
        dfShort =  self.__dataFrame.getData().ix[:,3:cutPoint].copy() # To avoid the case where changing dfShort also changes df 
        dfShort.hist()       
        fig = plt.gcf()
        # Option 1
        # QT backend
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()

        # Option 2
        # TkAgg backend
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        # Option 3
        # WX backend
        #manager = plt.get_current_fig_manager()
        #manager.frame.Maximize(True)
        #plt.show()
        fileToSave = myconstants.PNGSPATH + 'histograms.png'
        fig.savefig(fileToSave) 
        #plt.clf()        
        plt.close(fig)                 
    # -----------------------------------------


    # -----------------------------------------
    def getPositionsFromLabels(self, labels, value):
        """
            Returns the positions where value is found inside labels.
        """
        return[i for i, x in enumerate(labels) if x == value]        
    # -----------------------------------------


    # -----------------------------------------
    def ensure_dir(self, file_path):
        """
           Makes sure the path exists, creating it if it does not.
        """
        directory = os.path.dirname(file_path)
       
        if not os.path.exists(directory):
            print "Creating ", directory
            os.makedirs(directory)
    # -----------------------------------------
 

























