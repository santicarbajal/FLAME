__author__ = 'Santi'
import myconstants
import myutils
# The kmeans algorithm is implemented in the scikits-learn library
from sklearn.cluster import KMeans
from rawPlotter import *
from CsvHandler import *
from Plotter import *
from KMeansClassifier import *
#from XGBoostLearner import *
# ###


myutils.cleanFolder(myconstants.PNGSPATH)  # NOT CLEANING AT THE MOMENT. DANGEROUS

# ############################################################################
# First, we buid a rawPlotter object that will read data.cvs
# After creating the object, we will find processeddata.cvs in the same folder 
a = rawPlotter() # ReadingTrajectories
#  A general heatmap is already generated.
# ############################################################################

# ############################################################################
# Plotter object will read processeddata.csv
# and will generate histograms.png  
x = Plotter()    # Plotting processedData
x.plotDataFrame()
# ############################################################################


# ############################################################################
# The classifier  will apply k-means up to myconstants.MAXCLUSTERS
# and will generate the set of labels(types) for the trajectories
kMeansClassifier = KMeansClassifier()
#kMeansClassifier.showHistogram()
kMeansClassifier.elbowSearch()
# ############################################################################


# ############################################################################
# Now, we ask the plotter object to generate the correlation matrixes
# for each class (type) 
for i in range(myconstants.MAXCLUSTERS):
    x.showCorrelation(kMeansClassifier.getLabels(), i)
# ############################################################################


# ############################################################################
# and we get back to rawPlotter object to study the  representativeness and
# characteristics of each class, and to generate the different heatmaps
a.studyLabels(kMeansClassifier.getLabels())
a.plotAllHeatMaps(kMeansClassifier.getLabels())
# ############################################################################ 
# ###







 
