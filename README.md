# GENERAL INFORMATION

Authors: <br>
- Monique Broeren
- Sanne van den Dungen
- Fenna van Hemmen 
- Liza Hoens
- Kaj de Jong 
- Nienke Sengers 
- Ema Topolnjak 
- Nika Vredenbregt 

Eindhoven University of Technology <br>
Course: Advanced programming and biomedical data analysis <br>
Course code: 8CC00 <br>
Assignment: 4 <br>
Group: 4 <br>

## DATA & FILE OVERVIEW

File List: 
- main.py:  This is the main code and was used to receive predictions for PKM2 inhibitor 
            and ERK2 inhibitor. This file is used to receive all results given in the report.
- exploring_possibilities.ipynb: This is the file that is used to explore all possibilities for 
                                 how to predict inhibition. 
- old_codes: This folder contains all old codes. For more details of how main.py and 
             exploring_possibilities.ipynb were made, intermediate steps can be found there.
- data: This folder contains the data that was used to make resuls. The file 
        tested_molecules.csv contains molecules with known inhibitions. The file 
        untested_molecules.csv contains molecules with unknown inhibitions and for these
        molecules, the inhibition is predicted. 
- results: This folder contains the results if main.py is run. It consists of confusion 
           matrices, cumulative explained variance graphs and a csv-file with the 
           predictions of the untested molecules. 
- .venv: This folder contains the virtual environment. The file requirements.txt in this
         folder can be used to make a virtual environment for running this code.


## TO RUN THIS GITHUB

1. Make a virtual environment with the requirements.txt file in the .venv folder.
2. Run main.py. 
3. View the results in the results folder.
4. For more information about the creation process of main.py, view the other files.


## METHODOLOGICAL INFORMATION

1. For the molecules given, features were calculated (so called molecular descriptors, binary fingerprints and 
   binarized molecular descriptors).
2. A preselection is made of features that are the most useful. This is done by removing columns with no variation,
   removing columns with high correlation, and selecting the top 5 features using SelectKBest from scikit-learn.
3. Principal component analysis with MinMax scaled features to create a space with maximal variation and tp 
   reduce dimensionality.
4. Multiple machine learning algorithms are tested to see which works best to predict inhibition of molecules. 
   To do this, the tested_molecules are split in a training and test set multple times. The training set is 
   resamples in multipe ways and the algorithm is trained based on the training set. Thereafter, the model is 
   validated with the test set.
5. The best machine learning algorithm is used to predict the inhibiton of untested_molecules. The whole set
   tested_molecules is used for training the algorithm.
