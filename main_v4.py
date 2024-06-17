#import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


def read_data(infile):
    """ Reads the input (.csv) file.

    Parameters: infile (str)       - path where the file is located
    Returns:    df_raw (DataFrame) - consists of raw data that was in the file
    """
    df_raw = pd.read_csv(infile)
    return df_raw



def calculate_descriptors_and_fingerprints(smile, use_descriptors=True, use_fingerprints=False):
    """ Calculates the descriptors and fingerprints for the given SMILES.

    Parameters: smile (str)                - SMILES string of molecule
                use_desciptors (Boolean)   - adds descriptors as feature of molecule if set True
                use_fingerprints (Boolean) - adds fingerprints as feature of molecule if set True
    Returns:    descriptors_and_fp (list)  - list of all values that describe the molecule and 
                                             its binary fingerprints
    """
    # extract the molecule
    mol = Chem.MolFromSmiles(smile)

    # make list of features
    descriptors_and_fp = []

    # if no molecule found, return None
    if mol is None:
        if use_descriptors:
            descriptors_and_fp += [None] * len(descriptor_names)
        if use_fingerprints:
            descriptors_and_fp += [None] * nBits
        return descriptors_and_fp

    # calculate descriptors if required
    if use_descriptors:
        descriptors = list(calculator.CalcDescriptors(mol))
        descriptors_and_fp += descriptors

    # calculate binary fingerprints if required
    if use_fingerprints:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
        fingerprints = list(fp)
        descriptors_and_fp += fingerprints

    return descriptors_and_fp



def create_dataframe(df_raw, use_descriptors=True, use_fingerprints=False):
    """ Creates a dataframe containing all the descriptors and fingerprints of all the SMILES.

    Parameters: df_raw (DataFrame)         - contains all the raw data
                use_desciptors (Boolean)   - adds descriptors as feature of molecule if set True
                use_fingerprints (Boolean) - adds fingerprints as feature of molecule if set True
    Returns:    df_all_info (DataFrame)    - contains all the raw data and the calculated
                                             molecule descriptor and fingerprint data
    """
    # calculate descriptors and fingerprints for each row
    results = df_raw['SMILES'].apply(lambda smile: calculate_descriptors_and_fingerprints(smile, use_descriptors, use_fingerprints))

    # determine column names for the new DataFrame
    descriptor_columns = descriptor_names if use_descriptors else []
    fingerprint_columns = [f'fp_{i}' for i in range(nBits)] if use_fingerprints else []
    new_columns = descriptor_columns + fingerprint_columns

    # make dataframe that has original information and the calculated features
    result_df = pd.DataFrame(results.tolist(), columns=new_columns)
    df_all_info = pd.concat([df_raw, result_df], axis=1)

    return df_all_info



def binarizing_columns(df_features):
    """ Add columns to dataframe that have binarized information of columns that have discrete integers.

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
    """ Remove columns that only have one value, because they do not add extra information.

    Parameters: df_features (DataFrame) - contains all calculated features
    Returns:    df_features (DataFrame) - contains all calculated features without the columns 
                                          with only one value in them
    """
    # check if there are columns with only the same value
    non_variating_columns = df_features.columns[df_features.nunique()==1].tolist()
    # print('The columns that only have the same value in them are:', non_variating_columns)

    # remove the columns with only the same value, because molecules cannot be differentiated on those columns
    if len(non_variating_columns) != 0:
        df_features.drop(columns = non_variating_columns, inplace = True)
    
    return df_features



def make_correlation_matrix(df_features, figurename):
    """ Make a correlation matrix heatmap to see the correlation between the features 
    based on Spearman test.

    Parameters: df_features (DataFrame) - contains all calculated features
    Returns:    -
    """
    correlation_matrix = df_features.corr(method = 'spearman')

    # make a heatmap to visualize the correlation matrix
    plt.figure(figsize=(max(10,int(len(df_features.columns)*0.2)), max(10,int(len(df_features.columns)*0.2))))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title(figurename, fontdict={'fontsize' : max(14,int(len(df_features.columns)*0.4))})
    plt.savefig(f'results/{figurename}.svg')  # Save as SVG file



def prefilter_irrelevant_columns(df_raw, df_features, label):
    """ Remove columns that do add much extra information based on their correlation with other
    columns (columns with correlation higher than 0.8). Also, 5 best features are chosen 
    based on SelectKBest. 

    Parameters: df_raw (DataFramw)       - consists of raw data that was in input file
                df_features (DataFrame)  - contains all calculated features
                label (str)              - what needs to be predicted ('PKM2_inhibition' or 'ERK2_inhibition')
    Returns:    selected_features (list) - contains all features that are used for machine learning
                df_features (DataFrame)  - contains all data that will be used for machine learning
    """
    # only low correlated feataures are used
    threshold = 0.8
    corr_matrix = df_features.iloc[:,1:].corr().abs()
    # only check values in upper triangle of correlation matrix to avoid looking at values twice
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_features = df_features.drop(columns=to_drop)
    
    # only keep 5 best features based on SelectKBest method
    # seperate features and label
    X_set = df_features.iloc[:, 1:] 
    if label == 'PKM2_inhibition':
        y_set = df_raw.iloc[:, 1].values
    if label == 'ERK2_inhibition':
        y_set = df_raw.iloc[:, 2].values
 
    # make SelectKBest selector
    selector = SelectKBest(score_func=f_classif, k=5)
    X_new = selector.fit_transform(X_set, y_set)

    # get selected features and make dataframe based on that
    selected_features = X_set.columns[selector.get_support()]
    df_features = pd.DataFrame(X_new, columns=selected_features)

    return selected_features, df_features



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
    """ Scales all columns with MinMax scaling.

    Parameters: df_features (DataFrame)        - contains all used features
    Returns:    df_features_scaled (DataFrame) - contains all used features with MinMax scaled values
    """   
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_features)
    df_features_scaled = pd.DataFrame(scaled_array, columns=df_features.columns)
    return df_features_scaled



def perform_PCA(df_features_scaled, cum_var_threshold=0.8):
    """ Tests PCA to see how many PCs are needed to capture certain cumulative variance threshold and based on
    that, the PCs are calculated.

    Parameters: df_features_scaled (DataFrame) - contains all calculated features with MinMax scaled values
                cum_var_threshold (float)      - minimal cumulative explained variance of PCs together that 
                                                 need to be returned
    Returns:    pca (object)                   - pca object with wanted amount of cumulative explained variance
                num_components (int)           - number of PCs needed for wanted cumulative explained variance
                cumulative_variance (array)    - contains all cumulative explained variances
                df_pc_scores (DataFrame)       - contains all PCs scores with information that contains
                                                 at least a threshold cumulative explained variance together
    """
    # test PCA to see how many PCs are wanted
    pca = PCA().fit(df_features_scaled)

    # calculate the cummulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
  
    # choose amount of principal components based on how much cumulative explained variance is wanted
    num_components = np.argmax(cumulative_variance >= cum_var_threshold) + 1
    # print(f'The amount of principal components needed to capture at least {round(cum_var_threshold*100)}% variance is {num_components}.')

    # apply PCA with the desired number of components
    pca = PCA(n_components=num_components)
    pca_scores = pca.fit_transform(df_features_scaled)
    df_pc_scores = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(num_components)])

    return pca, num_components, cumulative_variance, df_pc_scores



def cumulative_explained_variance_graph(cumulative_variance, cum_var_threshold=0.8):
    """ Makes a cumulative explained variance graph that explains the PCs.

    Parameters: cumulative_variance (list) - contains all cumulative explained variances
                cum_var_threshold (float)  - minimal cumulative explained variance of PCs together
    Returns:    -
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.5, align='center')

    # make a threshold line based on the cumulative explained variance threshold that is chosen
    cum_var_threshold = 0.8
    plt.axhline(y=cum_var_threshold,color='r',linestyle='-')

    plt.xlabel('Principal component (PC)')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative explained variance for each principal component')
    plt.savefig('results/cumlative_explained_variance_graph.svg')



def split_data(df_pc_scores_labels, label, resample_method, molecules='tested'):
    """ Splits data in training and test set for machine learning and resamples training data based 
    on resampling method that is whished for (options: 'random_undersample', 'random_oversample', 
    'SMOTE', 'near_miss', 'smote_and_near_miss' and 'no_resampling'). Also, which inhibitor needs 
    to be used as label is defined (options = 'PKM2_inhibition' and 'ERK2_inhibition'). If molecules
    is set to 'tested', dataset is split. If molecules is set to 'untested', dataset is not split.

    Parameters: df_pc_scores_labels (DataFrame) - contains all PCs scores with information that contains
                                                  at least a threshold cumulative explained variance together and
                                                  labels.
                label (str)                     - what needs to be predicted ('PKM2_inhibition' or 'ERK2_inhibition')
                resample_method (str)           - resampling method ('random_undersample', 'random_oversample', 'SMOTE', 
                                                  'near_miss', 'smote_and_near_miss' or 'no_resampling')     
                moleculues (str)                - choose 'tested' or 'untested' based on if whole 'tested' molecules
                                                  should be training set    
    Returns:    X_train (ndarray)               - features for training set in ML 
                X_test (ndarray)                - features for test set in ML 
                y_train (array)                 - labels for training set in ML 
                y_test (array)                  - labels for test set in ML 
    """
    # seperate features and label
    X_set = df_pc_scores_labels.iloc[:, :-2].values # features
    if label == 'PKM2_inhibition':
        y_set = df_pc_scores_labels.iloc[:, -2].values
    if label == 'ERK2_inhibition':
        y_set = df_pc_scores_labels.iloc[:, -1].values
    
    # data split in train and test set
    if molecules == 'tested':
        X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=42) 
    if molecules == 'untested':
        X_train, y_train = X_set, y_set

    # random oversampling
    if resample_method == 'random_oversample':
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # random undersampling
    if resample_method == 'random_undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    # SMOTE
    if resample_method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # near miss
    if resample_method == 'near_miss':
        near_miss = NearMiss() 
        X_train, y_train = near_miss.fit_resample(X_train, y_train)

    # smote_and_near_miss
    if resample_method == 'smote_and_near_miss':
        # oversample the minority class using SMOTE
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

        # undersample the majority class using NearMiss
        nearmiss = NearMiss()
        X_train, y_train = nearmiss.fit_resample(X_smote, y_smote) 
     
    if molecules == 'tested':
        return X_train, X_test, y_train, y_test # returns part of tested molecules as training set and part as test set
    if molecules == 'untested':
        return X_train, y_train # returns whole tested molecules as training set



def classifier(X_train, X_test, y_train, classifier_type):
    """ Performs machine learning and predicts values for test set based on chosen machine learning algorithm
    (options: decision tree (='DT'), random forest (='RF'), logistic regression ('LR') and support vector
    machine (= 'SVM')).

    Parameters: X_train (ndarray)    - features for training set in ML 
                X_test (ndarray)     - features for test set in ML 
                y_train (array)      - labels for training set in ML
                classfier_type (str) - type of ML algorithm that should be done ('DT', 'RF', 'LR' or 'SVM')
    Returns:    y_pred (array)       - labels what ML predicted for test set
    """
    # logistic regression
    if classifier_type == 'LR':
        log_reg = LogisticRegression(random_state=42) 
        log_reg.fit(X_train, y_train) 
        y_prob = log_reg.predict_proba(X_test)[:, 1] 
        threshold = 0.25 # change this variable to different chance thresholds
        y_pred = (y_prob >= threshold).astype(int) # choose value based on threshold
    
    # decision tree
    if classifier_type == 'DT':
        dt = DecisionTreeClassifier(random_state=42) 
        dt.fit(X_train, y_train) 
        y_pred = dt.predict(X_test) 
    
    # random forest
    if classifier_type == 'RF':
        num_estimators = 100 # change this variable to test different number of estimators
        rf = RandomForestClassifier(n_estimators=num_estimators, random_state=42) 
        rf.fit(X_train, y_train) 
        y_pred = rf.predict(X_test) 

    # support vector machine
    if classifier_type == 'SVM':
        svm = SVC(kernel='poly')  # you can also try other kernels such as 'rbf', 'poly', etc 
        svm.fit(X_train, y_train) 
        y_pred = svm.predict(X_test)

    return y_pred



def goodness_prediction(y_test, y_pred):
    """ Evaluate ML model. 

    Parameters: y_test (array)                    - labels for test set in ML
                y_pred (array)                    - labels what ML predicted for test set
    Returns:    conf_matrix (list)                - confusion matrix
                accuracy (float)                  - accuracy score
                precision (float)                 - precision score
                sensitivity (float)               - sensitivity score
                specificity (float)               - specificity score
                negative_predictive_value (float) - negative predictive value score
    """
    conf_matrix = confusion_matrix(y_test, y_pred) 
    accuracy = accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred) 
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) #sensitivity (true positive rate)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) #specificity (true negative rate)
    negative_predictive_value = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

    return conf_matrix, accuracy, precision, sensitivity, specificity, negative_predictive_value



def average_results_ML_model(df_pc_scores, label, classifier_type, resample_method, iterations=50):
    """ Evaluate chosen ML model ('DT', 'RF', 'LR' or 'SVM') for chosen label ('PKM2_inhibition' or 
    'ERK2_inhibition') after chosen resample method is applied ('random_undersample', 
    'random_oversample', 'SMOTE', 'near_miss', 'smote_and_near_miss' or 'no_resampling'). The 
    average evaluation metrics of the model are returned after number iterations given.

    Parameters: df_pc_scores_labels (DataFrame) - contains all PCs scores with information that contains at least a
                                                  threshold cumulative explained variance together and labels (last two 
                                                  columns)
                label (str)                     - what needs to be predicted ('PKM2_inhibition' or 'ERK2_inhibition')
                classifier_type (str)           - type of ML algorithm that should be done ('DT', 'RF', 'LR' or 'SVM')
                resample_method (str)           - resampling method ('random_undersample', 'random_oversample', 'SMOTE', 
                                                  'near_miss', 'smote_and_near_miss' or 'no_resampling')   
                iterations (int)                - number of iterations over which evaluation metrics need to be averaged
    Returns:    avg_accuracy (float)            - average accuracy score
                avg_precision (float)           - average precision score
                avg_sensitivity (float)         - average sensitivity score
                avg_specificity (float)         - average specificity score
                avg_neg_pred_values (float)     - average negative predictive value score
                conf_matrix (list)              - confusion matrix of last iteration of model
    """  
    accuracies, precisions, sensitivities, specificities, neg_pred_values = [], [], [], [], []

    for i in range(iterations):
        # make and evaluate model once
        X_train, X_test, y_train, y_test = split_data(df_pc_scores, label, resample_method, molecules = 'tested')
        y_pred = classifier(X_train, X_test, y_train, classifier_type)
        conf_matrix, accuracy, precision, sensitivity, specificity, neg_pred_value = goodness_prediction(y_test, y_pred)

        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        neg_pred_values.append(neg_pred_value)

    # calculate avarages
    avg_accuracy = sum(accuracies) / iterations
    avg_precision = sum(precisions) / iterations
    avg_sensitivity = sum(sensitivities) / iterations
    avg_specificity = sum(specificities) / iterations
    avg_neg_pred_values = sum(neg_pred_values) / iterations

    return avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, avg_neg_pred_values, conf_matrix



if __name__ == '__main__':
    # create a MolecularDescriptorCalculator
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # number of bits for the binary fingerprints
    nBits = 1024  # default number of bits
    fingerprint_names = [f'Bit_{i}' for i in range(nBits)]
    
    # VARIABLES TO SET / CAN BE MODITIED
    use_descriptors = True
    use_fingerprints = False
    add_binarized_columns = False
    cum_var_threshold = 0.8
    label = 'PKM2_inhibition' # options: 'PKM2_inhibition', 'ERK2_inhibition' 
    resample_method = 'smote_and_near_miss' # options: 'random_undersample', 'random_oversample', 'SMOTE', 'near_miss', 'smote_and_near_miss', 'no_resampling'
    classifier_type = 'SVM' # options: 'DT', 'RF', 'LR', 'SVM'
    iterations = 50

    # read file
    input_file = 'data/tested_molecules.csv'
    df_all_info = create_dataframe(read_data(input_file), use_descriptors, use_fingerprints)

    # delete PKM2_inhibition and ERK2_inhibition, as these are not features 
    df_features = df_all_info.drop(columns = ['PKM2_inhibition','ERK2_inhibition'])
    
    # make colums that can be made binary binary
    if add_binarized_columns == True:
        df_features = binarizing_columns(df_features)
    
    # check if there are values missing in columns
    nan_counts = df_features.columns[df_features.isnull().any()].tolist()
    # print('The amount of columns where values are missing values is:', len(nan_counts))

    # remove variables that do not provide extra information
    df_features = remove_columns_no_info(df_features)
    make_correlation_matrix(df_features.iloc[:,1:], 'Correlation matrix of all features')

    # only keep important columns
    selected_features, df_features = prefilter_irrelevant_columns(df_all_info, df_features, label)
    make_correlation_matrix(df_features, 'Correlation matrix of filtered features')

    # check which columns are normally distributed
    normal_distr_columns = check_normality(df_features.iloc[:, 1:]) #SMILES column cannot be checked for normality
    #print('The columns that are normally distributed are:', normal_distr_columns)

    # scale data
    df_features_scaled = scale_data(df_features)

    # perform PCA
    pca, num_components, cumulative_variance, df_pc_scores = perform_PCA(df_features_scaled, cum_var_threshold)
    cumulative_explained_variance_graph(cumulative_variance, cum_var_threshold)

    # add labels to df_pc_scores
    df_pc_scores_labels = df_pc_scores.copy()
    df_pc_scores_labels['PKM2_inhibition'] = df_all_info.iloc[:, 1]
    df_pc_scores_labels['ERK2_inhibition'] = df_all_info.iloc[:, 2]

    # train and test chosen ML algorithm multiple times and evaluate the algorithm
    avg_accuracy, avg_precision, avg_sensitivity, avg_specificity, avg_neg_pred_values, conf_matrix = average_results_ML_model(df_pc_scores_labels, label, classifier_type, resample_method, iterations=iterations)
    print(f'RESULTS: classifier - {classifier_type}, resampling method - {resample_method}, use descriptors - {use_descriptors}, use fingerprints - {use_fingerprints}, use binarizing columns - {add_binarized_columns}, inhibitor prediction - {label}:')
    print('The average accuracy is:', round(avg_accuracy, 4))
    print('The average precision is:', round(avg_precision, 4))
    print('The average sensitivity is:', round(avg_sensitivity, 4))
    print('The average specificity is:', round(avg_specificity, 4))
    print('The average negative predictive value is:', round(avg_neg_pred_values, 4))
    print('The last predicition in the iteration had confusion matrix:\n', conf_matrix)

    # APPLY ML MODEL ON UNTESTED MOLECULES
    # read file
    input_file_new = 'data/untested_molecules.csv'
    df_all_info_new = create_dataframe(read_data(input_file), use_descriptors, use_fingerprints)
    
    # apply same preprocessing steps
    if add_binarized_columns:
        df_all_info_new = binarizing_columns(df_all_info_new)
    df_features_new = pd.DataFrame(df_all_info_new, columns=selected_features)
    df_features_scaled_new = scale_data(df_features_new)
    new_pca_scores = pca.transform(df_features_scaled_new)
    df_new_pc_scores = pd.DataFrame(new_pca_scores, columns=[f'PC{i+1}' for i in range(num_components)])

    results = {'SMILES': df_all_info_new.iloc[:,0]}
    for i in ['PKM2_inhibition', 'ERK2_inhibition']:
        # make new model based on all tested molecules and predict labels untested molecules
        # because both inhibitors work best with same settings, seperation of settings is not needed 
        X_train, y_train = split_data(df_pc_scores_labels, i, resample_method, molecules='untested')
        y_pred = classifier(X_train, np.array(df_new_pc_scores), y_train, classifier_type)
        results[i] = y_pred
    
    # write output 
    results_df = pd.DataFrame(results)
    output_file = 'results/predictions_untested_molecules.csv'
    results_df.to_csv(output_file, index=False, sep=',')
