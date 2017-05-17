__author__ = 'Santi'
# ###
# CLOSED. May, 15, 2017
# ###
#  THIS WILL BE THE CONSTANTS FILE. 
# ###


# ###
#    How many  types of clients  do we want to identify?
#    Note: 
#          Set this value to a high number, say 50.
#          Run the code : python  -OO test.py
#          See the results  of Elbow Method in  the file Inertial ElbowMethod.png, under the PNGSPATH folder
#          Decide the maximum number of Types, set MAXCLUSTERS to that value, and
#          Run the program again: python  -OO test.py
#
#              
# ###

MAXCLUSTERS            =        20

# ###
#
# ###


# ###
#         SHOP DEFINITION
# ###
ENTRANCEX              =        11.4
ENTRANCEY              =        46.36
XSIZE                  =        40.0
YSIZE                  =        48.0
FIELDLENGTH            =        50                                                # For field  Class constructor. Must be max(XSIZE, YSIZE)
PIXELSX                =        490                                               ## Referred to the size of file tienda.png
PIXELSY                =        584                                               ## Referred to the size of file tienda.png
PLUMBINGUPLEFT         =        (0,0)
PLUMBINGDOWNRIGHT      =        (236,220)  
ELECTRICITYUPLEFT      =        (0,266)
ELECTRICITYDOWNRIGHT   =        (236,420)
UNNAMEDUPLEFT          =        (0,421)
UNNAMEDDOWNRIGHT       =        (236,584)  
WOODUPLEFT             =        (265,0)
WOODDOWNRIGHT          =        (490,170)
PAINTINGUPLEFT         =        (265,198)
PAINTINGDOWNRIGHT      =        (490,296)  
CERAMICSUPLEFT         =        (265,326)
CERAMICSDOWNRIGHT      =        (490,426)
BUILDINGUPLEFT         =        (265,426)
BUILDINGDOWNRIGHT      =        (490,584)
SECTIONLABELS          =        ["PLUMBING","ELECTRICITY","UNNAMED","WOOD","PAINTINGS","CERAMICS","BUILDING", "UNDEFINED_AREA"]
NUMBEROFSECTIONS       =        8                                                 # last one is -1, corridors, undefined places 

# ###
#         END OF SHOP DEFINITION
# ###


# ###
#         RESOLUTIONS of Grid, and Hilbert Grid 
# ###
XRES                   =        4                                                  # Always Powers of two ( Hilbert Order is log this,2)
YRES                   =        XRES
CELLSIZEX              =        float(XSIZE/XRES)
CELLSIZEY              =        float(YSIZE/YRES)

# ###
#         END OF RESOLUTIONS
# ### 


# ###
#        PATH TO IMPORTANT FILES
# ###
DATAPATH               =        "/home/scarbajal/Desktop/FLAME/data.csv"           # Input data file 
PROCESSEDDATA          =        "/home/scarbajal/Desktop/FLAME/processeddata.csv"  # Output processed file for classification
MODELSPATH             =        "/home/scarbajal/Desktop/FLAME/MODELS/"            # Where to save learned models (Unsused, at the moment)
PNGSPATH               =        "/home/scarbajal/Desktop/FLAME/PNGS/"              # Where to save local heat maps and processed trayectories
BARRIERSFILE           =        "/home/scarbajal/Desktop/FLAME/barriers.png"       # Barriers File, describing obstacles
SHOPFILE               =        "/home/scarbajal/Desktop/FLAME/tienda.png"         # Shows the store
# ###
#        END of PATHS SECTION  
# ###


# ###
#       MACHINE LEARNING RELATED
# ###

INDEX_COL                  =        "Id"
VERBOSITY                  =        False
PLOTRELEVANT               =        True 
SAVETRAJECTORIES           =        False                                           # Save trajectories MAC by MAC in folders
SAVECLASSIFIEDTRAJECTORIES  =        False                                           # DEBUG: Save classified trajectories in folders, by class  
INCREMENTALHEAT            =        False                                           # DEBUG: Save general heatmap incrementaally or at the end 
ERRORTOLERANCE             =        1.0
INPUTVARIABLES             =        11
TOBEPREDICTED              =        3     
TOTALCOLUMNS               =        INPUTVARIABLES + TOBEPREDICTED      
RELEVANTFEATURES           =        1.0                                                #( 0.1 is 10 %  of total features will be listed,  according to relevance) 
TRAINPERC                  =        0.8
TESTPERC                   =        0.2
DIRECTION                  =        0                                                  # Study columns: 0 . Rows: 1   
  
# ###
#        END
# ###





