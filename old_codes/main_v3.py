#import needed libraries
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4' #needed to set because of SMOTE

import pandas as pd
import numpy as np

from scipy.stats import shapiro

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.utils import resample


def read_data(infile):
    """
    Reads the input (.csv) file.

    Parameters: infile (str)       - path where the file is located
    Returns:    df_raw (DataFrame) - consists of raw data that was in the file
    """
    df_raw = pd.read_csv(infile)
    return df_raw


def calculate_descriptors_and_fingerprints(smile, use_fingerprints=False):
    """
    Calculates the descriptors and fingerprints for the given SMILES.

    Parameters: smile (str)                - SMILES string of molecule
                use_fingerprints (Boolean) - adds fingerprints as feature of molecule if set True
    Returns:    descriptors_and_fp (list)  - list of all values that describe the molecule and its binary fingerprints.
    """
    # Extract the molecule
    mol = Chem.MolFromSmiles(smile)

    # If no molecule found, return None
    if mol is None:
        return [None] * len(descriptor_names) + ([None] * nBits if use_fingerprints else [])

    # Calculate descriptors
    descriptors = list(calculator.CalcDescriptors(mol))

    # Calculate binary fingerprints if required
    fingerprints = []
    if use_fingerprints:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
        fingerprints = list(fp)

    return descriptors + fingerprints


def create_dataframe(df_raw, use_fingerprints=False):
    """
    Creates a dataframe containing all the descriptors and fingerprints of all the SMILES.

    Parameters: df_raw (DataFrame)         - contains all the raw data
                use_fingerprints (Boolean) - adds fingerprints as feature of molecule if set True
    Returns:    df_all_info (DataFrame)    - contains all the raw data and the calculated
                                             molecule descriptor and fingerprint data
    """
    # Calculate descriptors and fingerprints for each row
    results = df_raw['SMILES'].apply(lambda smile: calculate_descriptors_and_fingerprints(smile, use_fingerprints))

    # Determine column names for the new DataFrame
    descriptor_columns = descriptor_names
    fingerprint_columns = [f'fp_{i}' for i in range(nBits)] if use_fingerprints else []
    new_columns = descriptor_columns + fingerprint_columns

    # Create new DataFrame from the results
    result_df = pd.DataFrame(results.tolist(), columns=new_columns)

    # Concatenate the original DataFrame with the new columns
    df_all_info = pd.concat([df_raw, result_df], axis=1)

    return df_all_info


def binarizing_columns(df_features):
    """
    Add columns to dataframe that have binarized information of columns that have discrete values.

    Parameters: df_features (DataFrame)           - contains all calculated features
    Returns:    df_features_binarized (DataFrame) - contains all calculated features and the extra binarized columns
    """
    
    df_features_binarized = df_features.copy()
    binarized_columns = {}
    
    for column in df_features.columns:
        # check if column is numeric
        if pd.api.types.is_numeric_dtype(df_features[column]):
            # check if columns only has integers
            if pd.api.types.is_integer_dtype(df_features[column].dropna()):
                # make binary column of that feature
                binarized_columns[f'{column}_binary'] = df_features_binarized[column].apply(lambda x: 1 if x != 0 else 0)
    
    # convert the dictionary to a DataFrame and concatenate it with the original DataFrame
    binarized_df = pd.DataFrame(binarized_columns)
    df_features_binarized = pd.concat([df_features_binarized, binarized_df], axis=1)
    return df_features_binarized


def remove_columns_no_info(df_features):
    """
    Remove columns that only have one value because they do not add extra information.

    Parameters: df_features (DataFrame) - contains all calculated features
    Returns:    df_features (DataFrame) - contains all calculated features without the columns with only one value in them
    """

    # check if there are columns with only the same value
    non_variating_columns = df_features.columns[df_features.nunique()==1].tolist()
    #print('The columns that only have the same value in them are:', non_variating_columns)

    # remove the columns with only the same value, because molecules cannot be differentiated on those columns
    if len(non_variating_columns) != 0:
        df_features.drop(columns = non_variating_columns, inplace = True)
    
    return df_features


def prefilter_irrelevant_columns(data_raw, df_features, label):

    # relevant_descriptors = [
    # 'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'FractionCSP3', 'LabuteASA', 'ExactMolWt',
    # 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 
    # 'NumAliphaticRings', 'NumSaturatedRings', 'NumAromaticCarbocycles', 'NumAliphaticCarbocycles',
    # 'NumAromaticHeterocycles', 'NumAliphaticHeterocycles', 'HallKierAlpha', 'PEOE_VSA1',
    # 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
    # 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi1',
    # 'Chi2n', 'Chi3n', 'Chi4n', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'Kappa1', 'Kappa2', 'Kappa3']

    # df_relevant = pd.DataFrame(df_features[relevant_descriptors])

    # only low correlated feataures
    # threshold = 0.8
    # corr_matrix = df_features.iloc[:,1:].corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # df_features = df_features.drop(columns=to_drop)

        
    # seperate features and label
    X_set = df_features.iloc[:, 1:] # features
    if label == 'PKM2_inhibition':
        y_set = data_raw.iloc[:, 1].values
    if label == 'ERK2_inhibition':
        y_set = data_raw.iloc[:, 2].values
 
    # Assuming X is your feature matrix and y is your target variable
    selector = SelectKBest(score_func=f_classif, k=10)
    X_new = selector.fit_transform(X_set, y_set)

    # Get the columns of the selected features
    selected_features = X_set.columns[selector.get_support()]

    # Create a new DataFrame with the selected features
    df_features = pd.DataFrame(X_new, columns=selected_features)


    # only high variance features
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
    df_features = selector.fit_transform(df_features)
    df_features = pd.DataFrame(df_features)

    # feature selection with PCA
    # scaler = MinMaxScaler()
    # scaled_array = scaler.fit_transform(df_features.iloc[:,1:])
    # df_features_scaled = pd.DataFrame(scaled_array, columns=df_features.iloc[:,1:].columns)

    # pca = PCA()
    # pca.fit(df_features_scaled)

    # # Get the explained variance ratios
    # explained_variance = pca.explained_variance_ratio_

    # # Determine the cumulative explained variance
    # cumulative_variance = np.cumsum(explained_variance)

    # # Set a threshold for explained variance (e.g., 95%)
    # threshold = 0.9

    # # Find the number of components to keep
    # num_components_to_keep = np.argmax(cumulative_variance >= threshold) + 1
    # print(f"Number of components to keep: {num_components_to_keep}")

    # # Get the principal components
    # components = pca.components_

    # # Select the top components
    # top_components = components[:num_components_to_keep]

    # # Calculate the contribution of each feature to the top components
    # feature_contributions = np.abs(top_components).sum(axis=0)

    # # Set a threshold to keep the most contributing features
    # contribution_threshold = np.percentile(feature_contributions, 95)  # For top 5% contributing features

    # # Identify the indices of the relevant features
    # relevant_feature_indices = np.where(feature_contributions >= contribution_threshold)[0]

    # # Get the names of the relevant features
    # relevant_features = df_features_scaled.columns[relevant_feature_indices]
    # print(f"Relevant features: {list(relevant_features)}")

    # # Filter the DataFrame to keep only the relevant features
    # df_relevant_features = df_features[relevant_features]

    # # Display the resulting DataFrame
    # df_relevant_features

    return df_features




def check_normality(df_features):
    """ Checks if values in columns are distributed normally.

    Parameters: df_features (DataFrame)     - dataframe for which you want to check if columns are
                                              distributed normally (all the features)
    Returns:    normal_distr_columns (list) - contains all column names that have p-value less than 
                                              0.05 in shapiro wilk test for normality
    """
    
    normal_distr_columns = []
    for column in df_features.columns:
        stat, p_value = shapiro(df_features[column].dropna())  # remove NaN values
        if p_value > 0.05:
            normal_distr_columns.append(column)
    return normal_distr_columns
    
    

def scale_data(df_features):
    """
    Scales all columns with MinMax scaling.

    Parameters: df_features (DataFrame)        - contains all calculated features
    Returns:    df_features_scaled (DataFrame) - contains all calculated features with MinMax scaled values
    """
    
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_features.iloc[:,1:])
    df_features_scaled = pd.DataFrame(scaled_array, columns=df_features.iloc[:,1:].columns)
    return df_features_scaled


def perform_PCA(df_features_scaled, cum_var_threshold=0.8):
    """
    Tests PCA to see how many PCs are needed to capture certain cumulative variance threshold and based on
    that, the PCs are calculated.

    Parameters: df_features_scaled (DataFrame) - contains all calculated features with MinMax scaled values
    Returns:    df_pc_scores (DataFrame)       - contains all PCs scores with information that contains
                                                 at least a threshold cumulative explained variance together
    """
    #test PCA to see how many PCs are wanted
    pca = PCA().fit(df_features_scaled)

    #calculate the cummulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    #choose amount of principal components based on how much cumulative explained variance is wanted
    num_components = np.argmax(cumulative_variance >= cum_var_threshold) + 1
    #print(f'The amount of principal components needed to capture at least {round(cum_var_threshold*100)}% variance is {num_components}.')

    # Apply PCA with the desired number of components
    pca = PCA(n_components=num_components)
    pca_scores = pca.fit_transform(df_features_scaled)
    df_pc_scores = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(num_components)])

    return df_pc_scores


def split_data(df_pc_scores, label, resample_method):
    """
    Splits data in training and test set for machine learning based on resampling method that is whished for
    (options: 'random_undersample', 'random_oversample', 'SMOTE', 'near_miss' and 'no_resampling'). Also, which
    inhibitor needs to be used as label is defined (options = 'PKM2_inhibition' and 'ERK2_inhibition').

    Parameters: df_pc_scores (DataFrame) - contains all PCs scores with information that contains
                                           at least a threshold cumulative explained variance together
                label (str)              - what needs to be predicted ('PKM2_inhibition' or 'ERK2_inhibition')
                resample_method (str)    - resampling method ('random_undersample', 'random_oversample', 'SMOTE', 
                                           'near_miss' or 'no_resampling')                  
    Returns:    X_train (ndarray)        - features for training set in ML 
                X_test (ndarray)         - features for test set in ML 
                y_train (array)          - labels for training set in ML 
                y_test (array)           - labels for test set in ML 
    """

    # seperate features and label
    X_set = df_pc_scores.iloc[:, :-2].values # features
    if label == 'PKM2_inhibition':
        y_set = df_pc_scores.iloc[:, -2].values
    if label == 'ERK2_inhibition':
        y_set = df_pc_scores.iloc[:, -1].values
    
    # data split in train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=42) 
    
    #random oversampling
    if resample_method == 'random_oversample':
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    #random undersampling
    if resample_method == 'random_undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    #SMOTE
    if resample_method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    #near miss
    if resample_method == 'near_miss':
        near_miss = NearMiss() 
        X_train, y_train = near_miss.fit_resample(X_train, y_train)

    #smote_and_near_miss
    if resample_method == 'smote_and_near_miss':
        # oversample the minority class using SMOTE
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        # undersample the majority class using NearMiss
        nearmiss = NearMiss()
        X_train, y_train = nearmiss.fit_resample(X_smote, y_smote) 
     
       
    return X_train, X_test, y_train, y_test


def classifier(X_train, X_test, y_train, classifier_type):
    """
    Performs machine learning and predicts values for test set based on chosen machine learning algorithm
    (options: decision tree (='DT'), random forest (='RF'), logistic regression ('LR') and support vector
    machine (= 'SVM')).

    Parameters: X_train (ndarray)    - features for training set in ML 
                X_test (ndarray)     - features for test set in ML 
                y_train (array)      - labels for training set in ML
                classfier_type (str) - type of ML algorithm that should be done ('DT', 'RF', 'LR' or 'SVM')
    Returns:    y_pred (array)       - labels what ML predicted for test set
    """
    
    #logistic regression
    if classifier_type == 'LR':
        log_reg = LogisticRegression(random_state=42) # initiate logistic regression
        log_reg.fit(X_train, y_train) # fit logistic regrssion

        y_prob = log_reg.predict_proba(X_test)[:, 1] # predict chance for outcome
        threshold = 0.4 # change this variable to different chance thresholds
        y_pred = (y_prob >= threshold).astype(int) # choose value based on threshold
    
    #decision tree
    if classifier_type == 'DT':
        dt = DecisionTreeClassifier(random_state=42) # initiate decision tree
        dt.fit(X_train, y_train) # fit decision tree
        y_pred = dt.predict(X_test) # make prediction
    
    #random forest
    if classifier_type == 'RF':
        num_estimators = 100 # change this variable to test different number of estimators
        rf = RandomForestClassifier(n_estimators=num_estimators, random_state=42) # initiate Random Forest
        rf.fit(X_train, y_train) # fit random forest
        y_pred = rf.predict(X_test) # make predictions

    #support vector machine
    if classifier_type == 'SVM':
        svm = SVC(kernel='poly')  # you can also try other kernels such as 'rbf', 'poly', etc 
        svm.fit(X_train, y_train) # fit support vector machine
        y_pred = svm.predict(X_test)

    return y_pred


def goodness_prediction(y_test, y_pred):
    """
    Evaluate ML model.

    Parameters: y_test (array)      - labels for test set in ML
                y_pred (array)      - labels what ML predicted for test set
    Returns:    conf_matrix (list)  - confusion matrix
                accuracy (float)    - accuracy score
                precision (float)   - precision score
                sensitivity (float) - sensitivity score
                specificity (float) - specificity score
    """
    conf_matrix = confusion_matrix(y_test, y_pred) #confusion matrix
    accuracy = accuracy_score(y_test, y_pred) #accuracy
    precision = precision_score(y_test, y_pred) #precision
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) #sensitivity (true positive rate)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) #specificity (true negative rate)

    return conf_matrix, accuracy, precision, sensitivity, specificity


def average_results_ML_model(df_pc_scores, label, classifier_type, resample_method, iterations=10):
    """
    Evaluate chosen ML model ('DT', 'RF', 'LR' or 'SVM') for chosen label ('PKM2_inhibition' or 
    'ERK2_inhibition') after chosen resample method is applied ('random_undersample', 
    'random_oversample', 'SMOTE', 'near_miss' or 'no_resampling'). The average evaluation metrics
    of the model are returned after number iterations given.

    Parameters: df_pc_scores_labels (DataFrame) - contains all PCs scores with information that contains at least a
                                                  threshold cumulative explained variance together and labels (last two 
                                                  columns)
                label (str)                     - what needs to be predicted ('PKM2_inhibition' or 'ERK2_inhibition')
                classifier_type (str)           - type of ML algorithm that should be done ('DT', 'RF', 'LR' or 'SVM')
                resample_method (str)           - resampling method ('random_undersample', 'random_oversample', 'SMOTE', 
                                                 'near_miss' or 'no_resampling')   
                iterations (int)                - number of iterations over which evaluation metrics need to be averaged
    Returns:    avg_accuracy (float)             - average accuracy score
                avg_precision (float)           - average precision score
                avg_sensitivity (float)         - average sensitivity score
                avg_specificity (float)         - average specificity score
                conf_matrix (list)              - confusion matrix of last iteration of model
    """
    
    accuracies, precisions, sensitivities, specificities = [], [], [], []

    for i in range(iterations):
        X_train, X_test, y_train, y_test = split_data(df_pc_scores, label, resample_method)
        y_pred = classifier(X_train, X_test, y_train, classifier_type)
        conf_matrix, accuracy, precision, sensitivity, specificity = goodness_prediction(y_test, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    avg_accuracy = sum(accuracies) / iterations
    avg_precision = sum(precisions) / iterations
    avg_sensitivity = sum(sensitivities) / iterations
    avg_specificity = sum(specificities) / iterations

    return avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, conf_matrix



if __name__ == '__main__':
    # Extract all descriptors
    descriptor_names = [desc[0] for desc in Descriptors._descList]

    # Number of bits for the binary fingerprints
    nBits = 1024  # Default number of bits
    fingerprint_names = [f'Bit_{i}' for i in range(nBits)]

    # Create a MolecularDescriptorCalculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # VARIABLES TO SET / CAN BE MODITIED
    use_fingerprints = False
    add_binarized_columns = False
    cum_var_threshold = 0.8
    label = 'PKM2_inhibition' # options: 'PKM2_inhibition', 'ERK2_inhibition' 
    resample_method = 'smote_and_near_miss' # options: 'random_undersample', 'random_oversample', 'SMOTE', 'near_miss', 'no_resampling', 'smote_and_near_miss'
    classifier_type = 'SVM' # options: 'DT', 'RF', 'LR', 'SVM'
    iterations = 10

    # Read file
    input_file = './data/tested_molecules.csv'
    df_all_info = create_dataframe(read_data(input_file), use_fingerprints)

    # delete PKM2_inhibition and ERK2_inhibition, as these are not features 
    df_features = df_all_info.drop(columns = ['PKM2_inhibition','ERK2_inhibition'])
    
    #make colums that can be made binary binary
    if add_binarized_columns == True:
        df_features = binarizing_columns(df_features)
    
    # check if there are values missing in columns
    nan_counts = df_features.columns[df_features.isnull().any()].tolist()
    #print('The amount of columns where values are missing values is:', len(nan_counts))

    # remove variables that do not provide extra information
    df_features = remove_columns_no_info(df_features)

    # only keep important columns
    df_features = prefilter_irrelevant_columns(df_all_info, df_features, label)

    # check which columns are normally distributed
    normal_distr_columns = check_normality(df_features.iloc[:, 1:]) #SMILES column cannot be checked for normality
    #print('The columns that are normally distributed are:', normal_distr_columns)

    # scale data
    df_features_scaled = scale_data(df_features)

    # perform PCA
    df_pc_scores = perform_PCA(df_features_scaled, cum_var_threshold)

    #add labels to df_pc_scores
    df_pc_scores_labels = df_pc_scores.copy()
    df_pc_scores_labels['PKM2_inhibition'] = df_all_info.iloc[:, 1]
    df_pc_scores_labels['ERK2_inhibition'] = df_all_info.iloc[:, 2]

    #train and test chosen ML algorithm multiple times and evaluate the algorithm
    avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, conf_matrix = average_results_ML_model(df_pc_scores_labels, label, classifier_type, resample_method, iterations)
    print(f'AVG RES: {classifier_type}, res_meth {resample_method}, fp {use_fingerprints}, bin_col {add_binarized_columns}, MODEL {label}:')
    print('The average accuracy is:', round(avg_accuracy, 4))
    print('The average precision is:', round(avg_precision, 4))
    print('The average sensitivity is:', round(avg_sensitivity, 4))
    print('The average specificity is:', round(avg_specificity, 4))
    print('The last predicition in the iteration had confusion matrix:\n', conf_matrix)

    # APPLY ML MODEL ON UNTESTED MOLECULES: I AM NOT SURE IF THIS IS CORRECT
    # Read file
    # input_file_new = '.data/untested_molecules.csv'
    # df_all_info_new = create_dataframe(read_data(input_file), use_fingerprints)
    
    # # Apply same preprocessing steps
    # if add_binarized_columns:
    #     df_new_data = binarizing_columns(df_all_info_new)
    # df_all_info_new = remove_columns_no_info(df_all_info_new)
    # df_features_scaled_new = scale_data(df_all_info_new)
    # df_pc_scores_new = perform_PCA(df_features_scaled_new, cum_var_threshold)

    # Predict using the trained model
    # I DID NOT KNOW HOW TO THIS YET, SO I STOPPED HERE

