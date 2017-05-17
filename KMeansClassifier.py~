__author__ = 'Santi'
# ###
# CLOSED. May, 15, 2017
# ###
# KMeansClassifier.py
# ###
"""KmeansClassifier.py
Provide a simple class for reading our processeddata.csv file, and implement the elbow method"""
import myconstants

# ###
# import modules
# ###
import myconstants
# The kmeans algorithm is implemented in the scikits-learn library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


# ###
#  END OF IMPORTS
# ###



class KMeansClassifier(object):
    'KMeansClassifier Class'
    className = "KMeansClassifier"
    classCounter = 0
    # -----------------------------------------
    def __init__(self):
        KMeansClassifier.classCounter += 1
        #print "I am the  KMeansClassifier constructor inside class ", KMeansClassifier.__doc__
        #print "Total number of ", self.className, " objects  is ", KMeansClassifier.classCounter
        self.__fileName = myconstants.PROCESSEDDATA
        self.__processedData = pd.read_csv(self.__fileName, index_col=myconstants.INDEX_COL)  # reading the data
        #self.__processedData.min()   #ignore header
        self.__processedData.drop(self.__processedData.columns[[0,1,2 ,3,4,5,6,7,8,9,10]], axis=1, inplace=True)
        self.__labels = []   #list of labels for each example                        
    # -----------------------------------------
    

    # -----------------------------------------
    def getLabels(self):
        """
            Returns the labels obtained from automatic clustering process.
        """
        return self.__labels
    # -----------------------------------------


    # -----------------------------------------
    def showHistogram(self):
        """
            Shows histogram  from read csv file. Interactive.
            Do not use without X system 
        """
        self.__processedData.hist()
        plt.show() 
    # ----------------------------------------- 


    # -----------------------------------------
    def __str__(self):
        """
            .
        """
        return "I am something " + " read from file " + self.__fileName + " Plus  some plotting Functions" +"\n" 
    # ----------------------------------------- 


    # ----------------------------------------- 
    def elbowSearch(self):
        """
          Performs elbowSearch and leaves labels with the configuration of the maximum number of clusters.
          NOT  OPTIMAL. ONLY FOR VISUALIZING PURPOSES. CHOOSE OPTIMAL VALUE AND WRITE IT DOWN  TO myconstants.py
        """
        inertias =[]
        
        for k in range (1, myconstants.MAXCLUSTERS+1):
 
	    # Create a kmeans model on our data, using k clusters.  random_state ensures that the algorithm returns the same results each time.
	    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(self.__processedData.iloc[:, :])
	
	    # Fitted labels  (the first cluster has label 0).
	    self.__labels = kmeans_model.labels_
            #print labels
 
	    # Sum of distances of samples to their closest cluster center
	    inertia = kmeans_model.inertia_
	    print "k:",k, " cost:", inertia
            inertias.append(inertia)
        plt.grid()
        plt.plot(inertias)
        fileName = myconstants.PNGSPATH  +  '/'+ 'InertialElbowMethod' +  '.png'
        plt.savefig(fileName)
        #plt.show()
    # ----------------------------------------- 


