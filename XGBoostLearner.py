__author__ = 'Santi'
# XGBoostLearner.py

"""Plotter.py
XGBoost Classifier Class"""
import myconstants

# ###
# import modules and read  data fileb
# ###
from Plotter import *

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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from random import shuffle

import xgboost as xgb
from sklearn.grid_search import GridSearchCV

# ###
#  END OF IMPORTS
# ###

class XGBoostLearner(object):
    'XGBoost Learner Class'
    className = "XGBoost"
    classCounter = 0
    
    



    # -----------------------------------------
    def __init__(self):
        Plotter.classCounter += 1
        print "I am the  XGBoostLearner constructor inside class ", XGBoostLearner.__doc__
        print "Total number of ", self.className, " objects  is ", XGBoostLearner.classCounter

        self.__plotter   = Plotter()
        self.__cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
        self.__ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
         'objective': 'reg:linear'}
        self.__classifier = xgb.XGBClassifier(**self.__ind_params)

        self.__trainPerc = myconstants.TRAINPERC
        self.__testPerc  = myconstants.TESTPERC      
        self.__cvPerc    = 0.0

        # create random list of indices, to use same train/test with any classifier
        # ##########################
        lenData = len(self.getPlotter().getDataFrame().getData())
        self.__shuffledList = range(lenData)
        shuffle(self.__shuffledList)
        # ##########################
        
    # -----------------------------------------

    
    # -----------------------------------------
    def getPlotter(self):
        return self.__plotter

    # ----------------------------------------- 


    # -----------------------------------------
    def getClassifier(self):
        return self.__classifier

    # ----------------------------------------- 


    # -----------------------------------------
    def __str__(self):
        return "I am an XGBoostLearner " + "\n"

    # ----------------------------------------- 


    # -----------------------------------------
    def anything(self):
        return 0
    # -----------------------------------------


    # -----------------------------------------
    def pickleDumpModel(self, name):
        """  Save classifier: The sklearn API models are picklable
            must open in binary format to pickle
        """

        pickle.dump(self.__classifier, open( myconstants.MODELSPATH + str(name)+".pkl", "wb"))
        print myconstants.MODELSPATH + str(name)+".pkl"
    
    # -----------------------------------------


    # -----------------------------------------
    def pickleLoadModel(self, name):
        """ Load Classifier: The sklearn API models are picklable
            must open in binary format to pickle
        """
        self.__classifier = pickle.load(open(myconstants.MODELSPATH +str(name) +".pkl", "rb"))
        print myconstants.MODELSPATH + str(name)+".pkl"
        
      
    # -----------------------------------------


    # -----------------------------------------
    def getScore(self, classTitle):
        """
        return: float | accuracy score for classification model (e[0,1])
        params:
                   df: pandas dataframe
           classifier: sklearn classifier
           classTitle: string | title of class column in df
           trainPerc: percentage of data to train on (default=0.80)
           testPerc: percentage of data to test on (default=0.20)
                   (trainPerc + testPerc = 1.0)
        """
        assert self.__trainPerc + self.__testPerc == 1.0
        #print "Inside getScore"
        # split the dataset
        cvPerc =0.0
        training, cv, test = self.splitData()  # ***
    
        # get the features and classes
        featureNames = [col for col in self.getPlotter().getDataFrame().getData().columns if col != classTitle and col not in self.getPlotter().getDataFrame().getListOfClasses()]
        trainFeatures = training[ featureNames ].values
        trainClasses  = training[ classTitle   ].values
    
        # create class dict to track numeric classes
        classToString = {}
        classToNumber = {}
        for i, c in enumerate( sorted(set(trainClasses)) ):
            classToString[i] = c
            classToNumber[c] = i
            
        # change classes to numbers (if not already)
        trainClasses = [classToNumber[c] for c in trainClasses]
        #print trainClasses
    
        # fit the model
    
        self.__classifier.fit(trainFeatures, trainClasses) #Training
    
        self.pickleDumpModel(classTitle)  # Saving
    
        self.pickleLoadModel(classTitle) #cvPerc Loading
    
    
        relevantFeatures = self.studyImportances()
        #print "Trying to generate plots for Relevant variables"
        if myconstants.PLOTRELEVANT:
            pass 
            #plotRelevant(df,INPUTVARIABLES , TOBEPREDICTED, relevantFeatures,classTitle ) 
        #print "==================================="
        #print getColumNames(df,relevantFeatures)
        #print "==================================="
  


    
        # format for cross validation set
        testFeatures = test[ featureNames ].values
        testClasses  = [classToNumber[c] for c in test[classTitle].values]
    
        total =0
        right =0
        wrong = 0
        #print trainClasses
    
    
        prediction = self.__classifier.predict(testFeatures)
        #print prediction
        #print testClasses
    
        for i in range(len(prediction)):
            if (math.fabs(prediction[i] - testClasses[i])) <=myconstants.ERRORTOLERANCE:
                right = right +1
            else:
                wrong = wrong +1
            total = total +1
        
    
        print right," Correctly Classified  Out of ", total, " Tolerance: +- ", myconstants.ERRORTOLERANCE
        return float(right)/float(total) ,  self.__classifier.score(testFeatures, testClasses)   
    
        # compute the score on the test set
        #score = classifier.score(testFeatures, testClasses)
        #print "Leaving it"
        #return score
    # -----------------------------------------        


    # -----------------------------------------
    def testModel(self,  classTitle ):
        """
        return: list[float] | list of scores for model (e[0,1])
        params:
               df: pandas dataframe
        classifier: sklearn classifier
        classTitle: string | title of class column in df
            N: int | number of tests to run (default=1
        )
        trainPerc: percentage of data to train on (default=0.80)
        testPerc: percentage of data to test on (default=0.20)
               (trainPerc + testPerc = 1.0)
        """
        # compute N scores
        print "Starting test...", "trainFraction = ", self.__trainPerc, "testFraction = ", self.__testPerc  
        
        score,rawScore = self.getScore(classTitle)
        #print "==============="
        print classTitle,  "TEST SCORE: (1 is 100%% accuracy, second value is rawScore) ========> %2.2f , %2.2f" %(score,rawScore)
        #print "==============="
        
        print "Finish."
        
    # -----------------------------------------        


    # -----------------------------------------
    def splitData(self):
        """
        return: training, cv, test
                (as pandas dataframes)
                Breaking dataframe df  into train, test and crossvalidation
        params:
                  df: pandas dataframe
           trainPerc: float | percentage of data for trainin set 
           cvPerc: float | percentage of data for cross validation set 
           testPerc: float | percentage of data for test set 
                  (trainPerc + cvPerc + testPerc must be equal to 1.0)
    """
    
        # ###
        assert self.__trainPerc + self.__cvPerc + self.__testPerc == 1.0
        # ###
    
        l = self.__shuffledList
        # get splitting indicies
        dataLen = len(self.getPlotter().getDataFrame().getData())
        df = self.getPlotter().getDataFrame().getData()
        trainLen = int(dataLen*self.__trainPerc)
        cvLen    = int(dataLen*self.__cvPerc)
        testLen  = int(dataLen*self.__testPerc)
        # get training, cv, and test sets
        training = df.ix[l[:trainLen]]
        cv       = df.ix[l[trainLen:trainLen+cvLen]]
        test     = df.ix[l[trainLen+cvLen:]]

        #print len(cl), len(training), len(cv), len(test)
        return training, cv, test
    # -----------------------------------------        


    

    # -----------------------------------------        
    def getMoreRelevantIndexes(self, input):
        result= []
        while(len(result) < myconstants.RELEVANTFEATURES * len(input)):
            max    =  0.0
            posMax =  0

            for e in range(len(input)):
                if input[e] >= max  and e not in result:
                    max = input[e]
                    posMax = e
            result.append(posMax)
        return result
            
    
    # -----------------------------------------        

    def getFeatureImportances(self):
        """
        """
        return self.__classifier.feature_importances_
    
    # -----------------------------------------        


    # -----------------------------------------
    def studyImportances(self):
        """
        return: list of the most important variables
        params: a classifier
            If possible, studies the importance of the variables
            that classifier is using when making a decission
    
        """
        result =[]
        try:
            importances = self.getFeatureImportances()
            #print "RELEVANCE OF VARIABLES: "
            result = self.getMoreRelevantIndexes(importances)
            #print result
        except AttributeError:
            print "NO DATA ABOUT VARIABLE RELEVANCE"

        return result
    # -----------------------------------------        














