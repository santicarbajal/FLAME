__author__ = 'Santi'

# rawPlotter.py

""" rawPlotter.py
Provide a simple class for opening cvs  files, plotting raw trajectories and generating reconstructions with Lee Algorithm"""
import myconstants
import myutils
# ###
# import modules and read  data fileb
# ###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
#import seaborn as sns
import cv2
import os
from numpy  import array
import datetime as dt
import math

from Field import *

# ###
#  END OF IMPORTS
# ###

class rawPlotter(object):
    'raw Plotter Handler Class'
    className = "rawPlotter"
    classCounter = 0


    # -----------------------------------------
    def __init__(self):
        rawPlotter.classCounter += 1
        #print "I am the constructor inside class ", rawPlotter.__doc__
        #print "Total number of ", self.className, " objects  is ", rawPlotter.classCounter

        self.__entranceX        = int(myconstants.ENTRANCEX)
        self.__entranceY        = int (myconstants.ENTRANCEY)
        self.__xSize            = myconstants.XSIZE
        self.__ySize            = myconstants.YSIZE    
        self.__xRes             = myconstants.XRES
        self.__yRes             = myconstants.YRES
        self.__hilbertOrder     = int (math.log(self.__xRes,2))
        self.__cellSizeX        = myconstants.CELLSIZEX
        self.__cellSizeY        = myconstants.CELLSIZEY
        self.__dataPath         = myconstants.DATAPATH
        self.__indexCol         = myconstants.INDEX_COL
        self.__fileName         = myconstants.DATAPATH
        self.__outputFile       = myconstants.PROCESSEDDATA 
        self.__direction        = myconstants.DIRECTION

        self.__xHeats           = []
        self.__yHeats           = []

        self.__xHeatsVector     = [[] for x in xrange(0,myconstants.MAXCLUSTERS)]
        self.__yHeatsVector     = [[] for x in xrange(0,myconstants.MAXCLUSTERS)]

        self.__xStayingHeats    =  []
        self.__yStayingHeats    =  []

        self.__macs             = []
        self.__pathLengths      = []    
        self.__stayingTimes     = [] 
        self.__enteringTimes    = []
        self.__leavingTimes     = []
        self.__averageSpeeds    = []
        self.__redundancies     = []
        self.__detectionPoints  = []
        self.__gridCoverage     = np.zeros(self.__xRes * self.__yRes, dtype=int)
        self.__gridCoverages    = []
        self.__hilbertCoverage  = np.zeros(self.__xRes * self.__yRes, dtype=int)
        self.__hilbertCoverages = [] 
        
        self.__zOrderCoverage   = np.zeros(self.__xRes * self.__yRes, dtype=int)
        self.__gridSequence     = []
        self.__hilbertSequence  = []

        self.__logisticSequence = []
        self.__logisticSequences = []
 
        self.__logisticStaying = []
        self.__logisticStayings = []

        self.__gridSequences    = []
        self.__hilbertSequences = [] 
        self.__booleanGrid      = []
        self.__hilbertGrid      = []
        self.__booleanGrids     = []
        self.__hilbertGrids     = []
        self.__logisticPlaces   = [myconstants.PLUMBINGUPLEFT    ,   myconstants.PLUMBINGDOWNRIGHT,    \
                                   myconstants.ELECTRICITYUPLEFT ,   myconstants.ELECTRICITYDOWNRIGHT, \
                                   myconstants.UNNAMEDUPLEFT     ,   myconstants.UNNAMEDDOWNRIGHT,     \
                                   myconstants.WOODUPLEFT        ,   myconstants.WOODDOWNRIGHT,        \
                                   myconstants.PAINTINGUPLEFT    ,   myconstants.PAINTINGDOWNRIGHT,    \
                                   myconstants.CERAMICSUPLEFT    ,   myconstants.CERAMICSDOWNRIGHT,    \
                                   myconstants.BUILDINGUPLEFT    ,   myconstants.BUILDINGDOWNRIGHT     ]

        self.__logisticCoverage  = np.zeros(len(self.__logisticPlaces), dtype=int)
        self.__logisticCoverages = [] 

        self.__secLabels            = myconstants.SECTIONLABELS

        self.__trajectories= self.getTrajectories(self.__fileName)
        self.__stayingTimes = self.processTimes()

        self.__representativeness = np.zeros(myconstants.MAXCLUSTERS, dtype=float)
        self.__stayingAverages    = np.zeros((myconstants.MAXCLUSTERS, len(self.__secLabels)), dtype = float)

        self.plotLogicShop()
        
        for i in range(len(self.__trajectories)):
            print "Plotting trajectory ", i, "/",len(self.__trajectories)
            self.plotTrajectory(i)
 
        self.plotHeatMap()                              # generating heatmap at the very end
        self.generateProcessedData()

        #print len(self.__pathLengths), len(self.__stayingTimes)
       
    # -----------------------------------------

 
    # -----------------------------------------
    def mapIntoLogisticSection(self,x,y):
        """
        Converts x,y , meters  into pixels 
        """
        pX = (x * myconstants.PIXELSX) / myconstants.XSIZE
        pY = (y * myconstants.PIXELSY) / myconstants.YSIZE
        
        if pX <= 236:
                
            if  pY < myconstants.PLUMBINGDOWNRIGHT[1]:
                return 0     # PLUMBING

            if  pY >= myconstants.ELECTRICITYUPLEFT[1]  and pY < myconstants.ELECTRICITYDOWNRIGHT[1]:
                return 1     # ELECTRICITY

            if  pY >= myconstants.UNNAMEDUPLEFT[1]  and pY < myconstants.PIXELSY:
                return 2     # UNNAMED

        elif pX >= 265:


            if  pY < myconstants.WOODDOWNRIGHT[1]:
                return 3     # WOOD

            if  pY >=myconstants.PAINTINGUPLEFT[1]   and pY < myconstants.PAINTINGDOWNRIGHT[1]:
                return 4     # PAINTINGS
 
            if  pY >=myconstants.CERAMICSUPLEFT[1]   and pY < myconstants.CERAMICSDOWNRIGHT[1]:
                return 5     # CERAMICS

            if  pY >= myconstants.BUILDINGUPLEFT[1]  and pY < myconstants.PIXELSY:
                return 6     # BUILDING
   
            

        return myconstants.NUMBEROFSECTIONS-1         # _IF it does not belong to any defined section.... return UNDEFINED_AREA
    # -----------------------------------------





    # -----------------------------------------
    def getFirstTrajectory(self,fileName):
        """
            Reads the first trayectory inside data file. For fast tests.
        """

        result = [[self.__entranceX, self.__entranceY]]
        with open(fileName, 'r') as f:
            firstLine = f.readline()
        splitted = firstLine.split(",")
        firstMAC = splitted[0]        
        fIn = open(fileName, "r")
        for line in fIn:
            splittedLine = line.split(",")
            if splittedLine[0] != firstMAC:
                break
            result.append([float(splittedLine[2]), float(splittedLine[3])])
        result.append([self.__entranceX, self.__entranceY])        
        fIn.close()
        return result
    # -----------------------------------------


    # -----------------------------------------
    def getTrajectories(self,fileName):
        """
            Reads every trayectory from the data file
        """

        bigResult = []
        self.__macs      = []
        
        with open(fileName, 'r') as f:
            firstLine = f.readline()
        splitted = firstLine.split(",")
        currentMAC = splitted[0]
        enteringTime = splitted[1]
        bk = splitted[1]
        leavingTime  = splitted[1] 
        currentSession = splitted[4]

        self.__macs.append(currentMAC)
        self.__enteringTimes.append(enteringTime)  


        fIn = open(fileName, "r")

        result = [[self.__entranceX, self.__entranceY]]
        splittedLine = []


        for line in fIn:
            splittedLine = line.split(",")
            enteringTime = splittedLine[1]  

            if splittedLine[0] != currentMAC or currentSession != splittedLine[4]:
                result.append([self.__entranceX, self.__entranceY])
                bigResult.append(result)
                result = [[self.__entranceX, self.__entranceY]]
                currentMAC = splittedLine[0]
                currentSession = splittedLine[4]
                self.__macs.append(currentMAC)
                
                self.__leavingTimes.append(bk)
                enteringTime = splittedLine[1]
                self.__enteringTimes.append(enteringTime ) 
                  
            bk = splittedLine[1]     

            x = float(splittedLine[2]) if float(splittedLine[2]) <= self.__xSize else self.__xSize -1
            y = float(splittedLine[3]) if float(splittedLine[3]) <= self.__ySize else self.__ySize -1
            result.append([x, y])
        self.__leavingTimes.append(splittedLine[1])
        fIn.close()

        return bigResult

    # -----------------------------------------


    # -----------------------------------------
    def plotTrajectory(self,i):
        """
            Reconstructs, and plots a trajectory, generating png file, inside a directory whose name is the  MAC ADDRESS, under PNGS folder 
        """        

        font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
        s= (int(self.__xSize)+10, int(self.__ySize)+10)
        self.__gridCoverage     = np.zeros(self.__xRes * self.__yRes, dtype=int)
        self.__hilbertCoverage     = np.zeros(self.__xRes * self.__yRes, dtype=int)
        self.__logisticCoverage     = np.zeros(len(self.__logisticPlaces), dtype=int)
        self.__logisticStaying      =  np.zeros(len(self.__logisticPlaces), dtype=float)
        self.__hilbertSequence  = []
        self.__gridSequence     = []
        self.__logisticSequence = []  
             
        localRedundancy = np.zeros(s) 

        
        totalLength = 0.0
        totalTime = 0.0

        
        trajectory = self.__trajectories [i]        
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))
        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(-1, self.__xSize+1, 1.0))
        plt.yticks(np.arange(-1, self.__ySize+1, 1.0))
        plt.grid()
        plt.gca().invert_yaxis()
        plt.axes()
 
  
        for index in range(len(trajectory) -1):
            
            startPoint=  (int( trajectory[index][0]), int(trajectory[index][1]))
            finishPoint= (int(trajectory[index+1][0]), int(trajectory[index+1][1]))
            barriersPNG = self.buildBarriers()
   
            for eraser in range(len(trajectory)):
                barriersPNG = [ x for x in barriersPNG if x != (int(trajectory[eraser][0]), int(trajectory[eraser][1]) )]

            field = Field(len=myconstants.FIELDLENGTH, start=startPoint, finish=finishPoint,barriers=barriersPNG)
            field.emit()
            field()
            path = []
            try:
                path = field.get_path()
                field._clean() 
                
            except:
                pass
                
            
            for step in path:
                rectangle = plt.Rectangle((step[0], step[1]), 1.0, 1.0, fc = 'y',alpha = 0.9)
                self.__xHeats.append(step[0])
                self.__yHeats.append(step[1]) 
                localRedundancy[step[0]] [step[1]] +=1

                gridSection = myutils.mapIntoSection(step[0], step[1])
                xCorner,yCorner = myutils.bendIntoGridSection(step[0], step[1])
                self.__gridCoverage[gridSection] = 1
                self.__gridSequence.append(gridSection)
                hilbertSection = self.mapIntoHilbert(step[0], step[1],self.__hilbertOrder)
                self.__hilbertCoverage[hilbertSection] = 1
                self.__hilbertSequence.append(hilbertSection)
                logisticSection = self.mapIntoLogisticSection(step[0], step[1])
                self.__logisticCoverage[logisticSection]= 1
                self.__logisticStaying[logisticSection]+= 1
                self.__logisticSequence.append(logisticSection)

                plt.gca().add_patch(rectangle)
                totalLength +=1.0
                
        self.__pathLengths.append(totalLength)
        self.__gridSequence = myutils.unique( myutils.remove_adjacent(self.__gridSequence))
        self.__gridSequences.append(self.__gridSequence)    
        self.__hilbertSequence = myutils.unique(myutils.remove_adjacent(self.__hilbertSequence))
        self.__hilbertSequences.append(self.__hilbertSequence)
        self.__logisticSequence = myutils.unique(myutils.remove_adjacent(self.__logisticSequence))
        self.__logisticSequences.append(self.__logisticSequence)
        
        kSum = np.sum(self.__logisticStaying)
        for staying in range(len(self.__logisticStaying)):
            self.__logisticStaying[staying] = float(self.__logisticStaying[staying]* 100.0) / kSum

        self.__logisticStayings.append(self.__logisticStaying)
        self.__gridCoverages.append(float(len(filter(None, self.__gridCoverage))  ) / len(self.__gridCoverage))  
        self.__hilbertCoverages.append(float(len(filter(None, self.__hilbertCoverage))  ) / len(self.__hilbertCoverage))
        self.__logisticCoverages.append(float(len(filter(None, self.__logisticCoverage))  ) / len(self.__logisticCoverage))
        self.__detectionPoints.append(len(trajectory)-2)  
        
        for element in barriersPNG:
            rectangle = plt.Rectangle((element[0], element[1]), 1.0, 1.0,alpha = 0.9 ,fc ='darkgray', hatch = 'x')
            plt.gca().add_patch(rectangle)

        pos = 0
        for point in trajectory:
            rectangle = plt.Rectangle((point[0], point[1]), 1.0, 1.0, fc = 'red',alpha = 0.9)
            plt.gca().add_patch(rectangle)
                       
            if pos < len(trajectory)-1:
                plt.text(point[0]+0.25, point[1]+0.75, str(pos), fontdict=font)
             
            pos +=1
       
        redundancy = self.computeRedundancy( localRedundancy)
        self.__redundancies.append(redundancy)
        self.__averageSpeeds.append(float(self.__pathLengths[i])/self.__stayingTimes[i])
        

        if myconstants.SAVETRAJECTORIES:
            report = self.generateReport(i)

            datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
            im = image.imread(datafile)
            im[:, :, -1] = 0.33  # set the alpha channel
            plt.xlabel('X', fontdict=font)
            plt.ylabel('Y', fontdict=font)    
            plt.title(report ,loc ='left', fontdict=font)
            extent =[0,self.__xSize,self.__ySize,0]  
            plt.imshow(im, zorder= 1, extent=extent, alpha = 0.2)
            #self.plotGrid("red")
            self.plotHilbertGrid("red")
            fileName = myconstants.PNGSPATH  +  '/'  + self.__macs[i] + '/' + str(i) +'.png'
            myutils.ensure_dir(fileName)
            plt.savefig(fileName)
                
            #plt.show()

        ###
        plt.close()
        if myconstants.INCREMENTALHEAT:
            self.plotHeatMap()
        
        ###

     
    # -----------------------------------------

    def buildBarriers(self):
        """
            Builds logical barriers where  indicated by  barriers.png file
        """
       
        barriers = [] # This cant be class attribute because  for each trayectory, I will delete barriers at poits where user was detected
                      # Bad design, but safer
        img = cv2.imread(myconstants.BARRIERSFILE)
        height,width = img.shape[:2]

        for xDelta in range(int(self.__xSize) ):
            for yDelta in range(int(self.__ySize)):
                
                if img [int (yDelta * (height/self.__ySize))] [int (xDelta * (width/self.__xSize))] [0] < 10:   # Basically, if the point is more or less BLACK...
                 
                    barriers.append((xDelta,yDelta))        

        return barriers

    # -----------------------------------------
   

    # -----------------------------------------
    def getXHeats(self):
        """
            .
        """
        return self.__xHeats
    # -----------------------------------------


    # -----------------------------------------
    def getYHeats(self):
        """
            .
        """
        return self.__yHeats
    # -----------------------------------------


    # -----------------------------------------
    def __str__(self):
        """
            .
        """
        return "I am something " + " read from file " + self.getFileName() + "\n" 
    # ----------------------------------------- 


    # -----------------------------------------
    def computeRedundancy(self, localRedundancy):
        """
            Computes redundancy for a trayectory
        """
   
        if  float( len (localRedundancy[localRedundancy >0])) == 0.0:
            return 100.0 
        redundancy = float((float( len (localRedundancy[localRedundancy >0]))) - (float( len (localRedundancy[localRedundancy >1]))) ) / float( len (localRedundancy[localRedundancy >0])) 
        redundancy = (1.0 - redundancy) * 100.0
        return redundancy
    # -----------------------------------------


    # -----------------------------------------
    def plotGrid(self, theColor):
        """
            Plots Euclidean Grid
        """
   
        font = {'family': 'serif',
        'color':  theColor,
        'weight': 'normal',
        'size': 15,
        }         

        sec = 0
        for x in range (self.__xRes):
            for y in range(self.__yRes):
                rectangle = plt.Rectangle((x * self.__cellSizeX, y * self.__cellSizeY), self.__xSize, self.__ySize,fill=False, edgecolor=theColor,linewidth=2,linestyle='dashed')
                plt.gca().add_patch(rectangle)
                rectangle = plt.Rectangle((x * self.__cellSizeX, y * self.__cellSizeY), 2, 2,fill=False, edgecolor=theColor,linewidth=1,linestyle='dashed')
                plt.gca().add_patch(rectangle)
                plt.text(x * self.__cellSizeX+0.25, y * self.__cellSizeY+1.0, str(sec), fontdict=font)
                sec+=1
                       
    # -----------------------------------------


    # -----------------------------------------
    def plotHilbertGrid(self, theColor):
        """
            Plots Hilbert Grid, with indexes
        """

        font = {'family': 'serif',
        'color':  theColor,
        'weight': 'normal',
        'size': 15,
        }         
        
        sec = 0
        for x in range (self.__xRes):
            for y in range(self.__yRes):
                #print x,y
                rectangle = plt.Rectangle((x * self.__cellSizeX, y * self.__cellSizeY), self.__xSize, self.__ySize,fill=False, edgecolor=theColor,linewidth=2,linestyle='dashed')
                plt.gca().add_patch(rectangle)
                sec = self.mapIntoHilbert(x * self.__cellSizeX, y * self.__cellSizeY, self.__hilbertOrder)
                #print sec  
                #rectangle = plt.Rectangle((x * self.__cellSizeX, y * self.__cellSizeY), 2, 2,fill=False, edgecolor=theColor,linewidth=1,linestyle='dashed')
                #plt.gca().add_patch(rectangle)
                plt.text(x * self.__cellSizeX+0.25, y * self.__cellSizeY+1.0, str(sec), fontdict=font)
             

    # -----------------------------------------


    # -----------------------------------------
    def plotLogisticGrid(self, theColor, alpha ):
        """
            Plots Logistic Grid, with names
        """
        font = {'family': 'serif',
        'color':  theColor,
        'weight': 'normal',
        'size': 45,
        }      
        
        sec = 0
        #print  "LOGISTICS"
        #print len(self.__logisticPlaces)
        for i in range (0, len(self.__logisticPlaces), 2):
            x, y = myutils.pixelsToMeters(self.__logisticPlaces[i][0],self.__logisticPlaces[i][1])
            xLimit, yLimit  =  myutils.pixelsToMeters(self.__logisticPlaces[i+1][0],self.__logisticPlaces[i+1][1])
            width = xLimit - x
            height= yLimit - y     
            #print i, x,y   
            rectangle = plt.Rectangle((x,y), width, height,fill=False, edgecolor=theColor,linewidth=2,linestyle='dashed',hatch = 'x',alpha =alpha)
            plt.gca().add_patch(rectangle)
            #sec = self.mapIntoHilbert(x * self.__cellSizeX, y * self.__cellSizeY, self.__hilbertOrder)            
            plt.text(x+1 , y+1 , self.__secLabels[sec] + " (" + str(sec) + " )", fontdict=font,alpha = alpha)
            sec = sec + 1   
             

    # -----------------------------------------

        
    # ----------------------------------------- 

    def mapIntoHilbert(self,x, y, order ):
        """
        Hilbert Curves are part of a class of one-dimensional fractals known as
        space-filling curves, so named because they are one dimensional lines 
        that nevertheless fill all available space in a fixed area. They're 
        fairly well known, in part thanks to XKCD's use of them for a map of the 
        internet. As you can see, they're also of use for spatial indexing, since 
        they exhibit exactly the locality and continuity required. It's an ordering for 
        points on a plane.

        """

        hilbert_map = { 'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},'d': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},}

          
        #x,y = myutils.bend(x,y)
        #print x,y  
        x= int (int (x) / int (self.__cellSizeX))
        y=  int (int (y) / int (self.__cellSizeY))
        #print x,y
        current_square = 'a'
        position = 0
        for i in range(order - 1, -1, -1):
            position <<= 2
            quad_x = 1 if x & (1 << i) else 0
            quad_y = 1 if y & (1 << i) else 0
            quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
            position |= quad_position
        return position    


    # ----------------------------------------- 


    # -----------------------------------------
    def generateProcessedData(self):
        """
            .
        """
        fOut = open(myconstants.PROCESSEDDATA, "w")
        fOut.write("Id,")
        fOut.write("MAC,")
        fOut.write("Entering,")
        fOut.write("Leaving,")
        fOut.write("Staying,")
        fOut.write("PathLength,")
        fOut.write("Speed,")
        fOut.write("Points,")
        fOut.write("Redundancy,")
        fOut.write("GridCoverage,")
        fOut.write("HilbertCoverage,")
        fOut.write("LogisticCoverage,")
        """
        for i in range (self.__xRes * self.__yRes):
            fOut.write("g"+str(i) + ",") # GridSequence Booleanized 
        for i in range (self.__xRes * self.__yRes):
            fOut.write("h"+str(i) + ",") # HilbertSequence Booleanized 
        """
        for i in range (len(self.__secLabels)):
            fOut.write("l"+str(i)+self.__secLabels[i] + ",") # LogisticSequence Booleanized 


        fOut.write("FOOVARIABLETOBEPREDICTED")
        fOut.write("\n")


        #fOut.write(str(len(self.__gridCoverages)) + "\n")  

        for i in range (len (self.__gridCoverages)  ):

            fOut.write(str(i) + ",")  #Id
            fOut.write(str(self.__macs[i]) + ",") # MAC
            fOut.write(str(self.__enteringTimes[i]) + ",") # Entering
            fOut.write(str(self.__leavingTimes[i]) + ",") # Leaving 
            fOut.write(str(self.__stayingTimes[i]) + ",") # Staying
            fOut.write(str(self.__pathLengths[i]) + ",") # PathLength
            fOut.write(str(self.__averageSpeeds[i]) + ",") # Speeds
            
            fOut.write(str(self.__detectionPoints[i]) + ",") # Points
            fOut.write(str(self.__redundancies[i]) + ",") # Redundancy  
             
            fOut.write(str(self.__gridCoverages[i]) + ",") # Grid Coverage 
            fOut.write(str(self.__hilbertCoverages[i]) + ",") # Hilbert Coverage
            fOut.write(str(self.__logisticCoverages[i]) + ",") # Logistic Coverage 
            """
            for g in range (self.__xRes * self.__yRes):
                #print self.__gridSequences[i]
                if g in self.__gridSequences[i]:
                    fOut.write(str(1) + ",") # GridSequence Booleanized
                else:
                    fOut.write(str(0) + ",") # GridSequence Booleanized
            for h in range (self.__xRes * self.__yRes):
                if h in self.__hilbertSequences[i]:
                    fOut.write(str(1) + ",") # HilbertSequence Booleanized  
                else:
                    fOut.write(str(0) + ",") # HilbertSequence Booleanized 
            """
            for l in range ( len(self.__secLabels)):
                
                fOut.write(str(self.__logisticStayings[i][l]) + ",") # LogisticSequence Booleanized
                #if l in self.__logisticSequences[i]:
                    #fOut.write(str(1) + ",") # LogisticSequence Booleanized  
                #else:
                    #fOut.write(str(0) + ",") # LogisticSequence Booleanized 


            fOut.write(str(0) + "\n")

        fOut.close()
    # -----------------------------------------


    # -----------------------------------------
    def plotAllHeatMaps(self, labels):
        """
            .
        """

        for i in range(len(self.__trajectories)):
            print "Updating Heat ", i, "/",len(self.__trajectories)
            self.updateHeats(i,labels)
    
        for i in range(myconstants.MAXCLUSTERS):
            print "Generating Heat Map for class ", i, " ..." 
            self.plotWholeDiscriminantHeatMap(i) 
    # -----------------------------------------


    # -----------------------------------------
    def updateHeats(self,i,labels):
        """
            .
        """
        

        font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
        bigFont = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 1550,
        }

        s= (int(self.__xSize)+10, int(self.__ySize)+10)
       
        #datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
        #im = image.imread(datafile)
        #im[:, :, -1] = 0.33  # set the alpha channel
        totalLength = 0.0
        totalTime = 0.0

        
        trajectory = self.__trajectories [i]        
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))
        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(-1, self.__xSize+1, 1.0))
        plt.yticks(np.arange(-1, self.__ySize+1, 1.0))
        plt.grid()
        plt.gca().invert_yaxis()
        plt.axes()
 
  
        for index in range(len(trajectory) -1):
            
            startPoint=  (int( trajectory[index][0]), int(trajectory[index][1]))
            finishPoint= (int(trajectory[index+1][0]), int(trajectory[index+1][1]))
            barriersPNG = self.buildBarriers()
   
            for eraser in range(len(trajectory)):
                barriersPNG = [ x for x in barriersPNG if x != (int(trajectory[eraser][0]), int(trajectory[eraser][1]) )]

            field = Field(len=myconstants.FIELDLENGTH, start=startPoint, finish=finishPoint,barriers=barriersPNG)
            field.emit()
            field()
            path = []
            try:
                path = field.get_path()
                field._clean() 
                
            except:
                pass
                
            
            for step in path:
                rectangle = plt.Rectangle((step[0], step[1]), 1.0, 1.0, fc = 'y',alpha = 0.9)
                self.__xHeatsVector[labels[i]].append(step[0])
                self.__yHeatsVector[labels[i]].append(step[1])
                plt.gca().add_patch(rectangle)
                totalLength +=1.0
        
        
        for element in barriersPNG:
            rectangle = plt.Rectangle((element[0], element[1]), 1.0, 1.0,alpha = 0.9 ,fc ='darkgray', hatch = 'x')
            plt.gca().add_patch(rectangle)

        pos = 0
        for point in trajectory:
            rectangle = plt.Rectangle((point[0], point[1]), 1.0, 1.0, fc = 'red',alpha = 0.9)
            plt.gca().add_patch(rectangle)
                       
            if pos < len(trajectory)-1:
                plt.text(point[0]+0.25, point[1]+0.75, str(pos), fontdict=font)
             
            pos +=1
       
       

        if myconstants.SAVECLASSIFIEDTRAJECTORIES:
            
            datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
            im = image.imread(datafile)
            im[:, :, -1] = 0.33  # set the alpha channel
            # #############
            plt.text(20,20, str(labels[i]), fontdict=bigFont)
            plt.xlabel('X', fontdict=font)
            plt.ylabel('Y', fontdict=font)   
            report = self.generateReducedReport(i)        
            plt.title(report ,loc ='left', fontdict=font)        
            extent =[0,self.__xSize,self.__ySize,0]  
            plt.imshow(im, zorder= 1, extent=extent, alpha = 0.2)
            #self.plotGrid("red")
            self.plotHilbertGrid("red")
            #fileName = myconstants.PNGSPATH  +  '/'  + self.__macs[i] + '/' + str(i) + "CLASSIFIED"+'.png'
            fileName = myconstants.PNGSPATH  +  '/'+'0'  + str(labels[i]) + '/' + str(i) + "CLASSIFIED"+'.png'
            myutils.ensure_dir(fileName)        
            plt.savefig(fileName)
            #plt.close()
            #plt.show()
            # #############
        plt.close()
        ###
        if myconstants.INCREMENTALHEAT:
            self.plotDiscriminantHeatMap(i,labels)
        
        ###

      
    # -----------------------------------------


    # -----------------------------------------
    def plotHeatMap(self):
        """
            .
        """
        font = {'family': 'serif',
        'color':  'yellow',
        'weight': 'normal',
        'size': 25,
        }
        datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
        im = image.imread(datafile)
        im[:, :, -1] = 0.30  # set the alpha channel
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))  
        x = array(self.__xHeats)
        y = array(self.__yHeats)         
        xmin, xmax = 0,self.__xSize
        ymin, ymax = 0, self.__ySize
        nbins = 8
        xbins = np.linspace(xmin, xmax, nbins)
        ybins = np.linspace(ymin, ymax, nbins)
        extent=[xmin, xmax, ymin, ymax]

        #heatmap, xedges, yedges = np.histogram2d(x, y, bins=(self.__xSize, self.__ySize))
        heatmap, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(0, self.__xSize+1, 1.0))
        plt.yticks(np.arange(0, self.__ySize+1, 1.0))

        
        plt.imshow(heatmap.T, extent=[xmin, xmax, ymin, ymax], origin = "lower")
        
        plt.imshow(im, zorder= 1, extent=extent, origin = "lower")

        plt.gca().invert_yaxis() 
        self.plotHilbertGrid("lime")
        self.plotLogisticGrid("orange", 0.4) 
        
        yText = 2
        xText = float(self.__xSize) *0.75
        rectangle = plt.Rectangle((xText-1,1 ), self.__xSize/4.2, self.__ySize/6, fc = 'black', ec = 'white',alpha = 0.5)
        plt.gca().add_patch(rectangle)
        plt.text(xText, yText, "GENERAL "   , fontdict=font)
        yText += 0.6
        plt.text(xText, yText, "HEATMAP: " , fontdict=font)


        fileName = myconstants.PNGSPATH  +  '/' + 'heat' + '.png'
        plt.savefig(fileName)
        plt.clf() 
        plt.close()
    # -----------------------------------------


    # -----------------------------------------
    def plotDiscriminantHeatMap(self,i,labels):
        """
            Incremental Plot.
        """
        font = {'family': 'serif',
        'color':  'yellow',
        'weight': 'normal',
        'size': 25,
        }
        
        datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
        im = image.imread(datafile)
        im[:, :, -1] = 0.30  # set the alpha channel        
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))
        x = array(self.__xHeatsVector[labels[i]])        
        y = array(self.__yHeatsVector[labels[i]])
        xmin, xmax = 0,self.__xSize
        ymin, ymax = 0, self.__ySize
        nbins = 8
        xbins = np.linspace(xmin, xmax, nbins)
        ybins = np.linspace(ymin, ymax, nbins)
        extent=[xmin, xmax, ymin, ymax]
        heatmap, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
 
        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(0, self.__xSize+1, 1.0))
        plt.yticks(np.arange(0, self.__ySize+1, 1.0))

        
        plt.imshow(heatmap.T, extent=[xmin, xmax, ymin, ymax], origin = "lower")
        
        plt.imshow(im, zorder= 1, extent=extent, origin = "lower")

        plt.gca().invert_yaxis() 
        self.plotHilbertGrid("lime")
        self.plotLogisticGrid("orange", 0.4)  
        

        yText = 2
        xText = float(self.__xSize) *0.75
        rectangle = plt.Rectangle((xText-1,1 ), self.__xSize/4.2, self.__ySize/6, fc = 'black', ec = 'white',alpha = 0.5)
        plt.gca().add_patch(rectangle)
        plt.text(xText, yText, "Class " + str( labels[i])  , fontdict=font)
        yText += 0.6
        plt.text(xText, yText, "Representativeness: " + str( int (self.__representativeness[labels[i]])) + "%" , fontdict=font)

        for k in range(len(self.__secLabels)):
            yText += 0.6
            plt.text(xText, yText, self.__secLabels[k] +": " +  str( int(self.__stayingAverages[labels[i]][k])) +"%", fontdict=font)
            

           
        fileName = myconstants.PNGSPATH  +  '/'+'0'  + str(labels[i]) +  '/' + 'heat' + str(labels[i]) + '.png'
        myutils.ensure_dir(fileName)
        plt.savefig(fileName)
        plt.clf() 
        plt.close()


        
    # -----------------------------------------


    # -----------------------------------------
    def plotWholeDiscriminantHeatMap(self,i):
        """
            NON Incremental Plot. Plots heatmap for class i
        """
        font = {'family': 'serif',
        'color':  'yellow',
        'weight': 'normal',
        'size': 25,
        }
        
        datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
        im = image.imread(datafile)
        im[:, :, -1] = 0.30  # set the alpha channel        
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))
        x = array(self.__xHeatsVector[i])        
        y = array(self.__yHeatsVector[i])
        xmin, xmax = 0,self.__xSize
        ymin, ymax = 0, self.__ySize
        nbins = 8
        xbins = np.linspace(xmin, xmax, nbins)
        ybins = np.linspace(ymin, ymax, nbins)
        extent=[xmin, xmax, ymin, ymax]
        heatmap, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
 
        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(0, self.__xSize+1, 1.0))
        plt.yticks(np.arange(0, self.__ySize+1, 1.0))

        
        plt.imshow(heatmap.T, extent=[xmin, xmax, ymin, ymax], origin = "lower")
        
        plt.imshow(im, zorder= 1, extent=extent, origin = "lower")

        plt.gca().invert_yaxis() 
        self.plotHilbertGrid("lime")
        self.plotLogisticGrid("orange", 0.4)  
        

        yText = 2
        xText = float(self.__xSize) *0.75
        rectangle = plt.Rectangle((xText-1,1 ), self.__xSize/4.2, self.__ySize/6, fc = 'black', ec = 'white',alpha = 0.5)
        plt.gca().add_patch(rectangle)
        plt.text(xText, yText, "Class " + str(i)  , fontdict=font)
        yText += 0.6
        plt.text(xText, yText, "Representativeness: " + str( int (self.__representativeness[i])) + "%" , fontdict=font)

        for k in range(len(self.__secLabels)):
            yText += 0.6
            plt.text(xText, yText, self.__secLabels[k] +": " +  str( int(self.__stayingAverages[i][k])) +"%", fontdict=font)
            

           
        fileName = myconstants.PNGSPATH  +  '/'+'0'  + str(i) +  '/' + 'heat' + str(i) + '.png'
        myutils.ensure_dir(fileName)
        plt.savefig(fileName)
        plt.clf() 
        plt.close()


        
    # -----------------------------------------





    # -----------------------------------------
    def studyLabels(self,labels):
        """
            .
        """

        typeCounter = np.zeros(myconstants.MAXCLUSTERS, dtype=float) 

        for i in range(len(self.__trajectories)):
            print "Studying Trajectory ", i, "/",len(self.__trajectories)
            typeCounter[labels[i]] += 1.0

        length = len(self.__trajectories)
        self.__representativeness = typeCounter * (100.0/length)

        for i in range(len(self.__trajectories)):
            for j in range(len(self.__secLabels)):
                self.__stayingAverages[labels[i]][j] += self.__logisticStayings[i][j] 
        
        for i in range(len(self.__stayingAverages)):
            for j in range(len(self.__secLabels)):
                self.__stayingAverages[i][j] /= (typeCounter[i] )        
    # -----------------------------------------


    # -----------------------------------------
    def plotLogicShop(self):
        """
            .
        """
        font = {'family': 'serif',
        'color':  'yellow',
        'weight': 'normal',
        'size': 25,
        }
        datafile = cbook.get_sample_data(myconstants.SHOPFILE, asfileobj=False)
        im = image.imread(datafile)
        im[:, :, -1] = 0.20  # set the alpha channel
        plt.figure(figsize=(int (self.__xSize), int(self.__ySize)))  
             
        xmin, xmax = 0,self.__xSize
        ymin, ymax = 0, self.__ySize
        nbins = 8
        xbins = np.linspace(xmin, xmax, nbins)
        ybins = np.linspace(ymin, ymax, nbins)
        extent=[xmin, xmax, ymin, ymax]

        plt.xlim(-1.0, self.__xSize+1)
        plt.ylim(-1.0, self.__ySize+1)
        plt.xticks(np.arange(0, self.__xSize+1, 1.0))
        plt.yticks(np.arange(0, self.__ySize+1, 1.0))

        plt.imshow(im, zorder= 1, extent=extent, origin = "lower")

        plt.gca().invert_yaxis() 

        self.plotLogisticGrid("black",  0.9)            

        fileName = myconstants.PNGSPATH  +  '/' + 'logicshop' + '.png'
        plt.savefig(fileName)
        plt.clf() 
        plt.close()

    # -----------------------------------------
 

    # -----------------------------------------

    def generateReport(self,i):
        """
            Generates a String as a summary of a trayectory
        """
        report =           "Trajectory of user (MAC): " + str(self.__macs [i])              + '\n'
        report =  report + "Entering Time:            " + self.__enteringTimes[i]           + '  ' 
        report =  report + "Leaving Time:             " + self.__leavingTimes[i]            + '\n'
        report =  report + "Total Path Length:        " + str(self.__pathLengths[i])        + ' m'         + '  '    
        report =  report + "Staying Time:             " + str(self.__stayingTimes[i])       + " secs"      + '  ' 
        report  = report + "Average Speed:            " + str (self.__averageSpeeds[i])     +  " m/sec"    + '\n'
        report  = report + "Detection Points:         " + str(self.__detectionPoints[i])    + '\n'
        report  = report + "Redundancy: (%)           " + str (self.__redundancies[i])      + '\n'
        report = report  + "Grid Coverage:            " + str(self.__gridCoverages[i])      + "           Grid Sequence    : "     + str(self.__gridSequence)  + '\n'
        report = report  + "Hilbert Coverage:         " + str(self.__hilbertCoverages[i])   + "           Hilbert  Sequence: "     + str(self.__hilbertSequence)  + '\n'
        report = report  + "Logistic Coverage:        " + str(self.__logisticCoverages[i])  + "           Logistic  Sequence: "    + str(self.__logisticSequence)  + '\n'
        #report = report  + "z Order Coverage:         " + str(self.__zOrderCoverage)     + '\n'
        return report

    # -----------------------------------------


    # -----------------------------------------

    def generateReducedReport(self,i):
        """
            Generates a shorter  String as a summary of a trayectory
        """

        report =           "Trajectory of user (MAC): " + str(self.__macs [i])              +  '\n'
        report =  report + "Entering Time:            " + self.__enteringTimes[i]           +  '  ' 
        report =  report + "Leaving Time:             " + self.__leavingTimes[i]            +  '\n'
        report =  report + "Total Path Length:        " + str(self.__pathLengths[i])        +  ' m'      + '  '    
        report =  report + "Staying Time:             " + str(self.__stayingTimes[i])       +  " secs"   + '  ' 
        report  = report + "Average Speed:            " + str (self.__averageSpeeds[i])     +  " m/sec"  + '\n'
        report  = report + "Detection Points:         " + str(self.__detectionPoints[i])    +  '\n'
        report  = report + "Redundancy: (%)           " + str (self.__redundancies[i])      +  '\n'
        report = report  + "Grid Coverage:            " + str(self.__gridCoverages[i])      +  '\n'
        report = report  + "Hilbert Coverage:         " + str(self.__hilbertCoverages[i])   +  '\n'
        report = report  + "Logistic Coverage:        " + str(self.__logisticCoverages[i])  +  '\n'
        #report = report  + "z Order Coverage:         " + str(self.__zOrderCoverage)       +  '\n'
        return report
        

    # -----------------------------------------


    # ##############################################################################################################

    #    UTILITIES
    # ##############################################################################################################



    # -----------------------------------------
    def processTimes(self):
        """
            UTILITY: UTC time conversion routine. We need it to be inside class
        """
        result = []        
        for i in range (len(self.__enteringTimes)):
            result.append(myutils.calculateTimes(self.__enteringTimes[i], self.__leavingTimes[i]))

        return result

    # -----------------------------------------


    


   












