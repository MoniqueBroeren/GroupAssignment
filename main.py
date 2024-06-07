# Imports
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from scipy.stats import shapiro


def read_data(infile):
    """
    Reads the input (.csv) file.

    Parameters: infile (str)       - path where the file is located
    Returns:    df_raw (DataFrame) - consists of raw data that was in the file
    """
    df_raw = pd.read_csv(infile)
    return df_raw


def calculate_descriptors(smile):
    """
    Calculates the descriptors for the given smiles.

    Parameters: smile (str) - SMILES string of molecule
    Returns:    list (list) - list of all values that describe the molecule. If there is
                              no molecule, None returns.
    """
    # extract the molecule
    mol = Chem.MolFromSmiles(smile)
    # if no molecule found return None
    if mol is None:
        return [None] * len(descriptor_names)
    return list(calculator.CalcDescriptors(mol))


def create_dataframe(df_raw):
    """
    Creates a dataframe containing all the descriptors of all the smiles.

    Parameters: df_raw (DataFrame)      - contains all the raw data
    Returns:    expanded_df (DataFrame) - contains all the raw data and the calculated
                                          molecule descriptor data
    """
    # Calculate descriptors for each molecule
    descriptor_data = df_raw['SMILES'].apply(calculate_descriptors)

    # Create a DataFrame with descriptor data
    descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptor_names)

    # Combine the original DataFrame with the descriptor DataFrame
    expanded_df = pd.concat([df_raw, descriptor_df], axis=1)

    return expanded_df


def check_normality(df):
    """ Checks if values in columns are distributed normally.
    
    Parameters: df (DataFrame)           - dataframe for which you want to check if columns are
                                           distributed normally
    Returns:    normality_results (dict) - for each column (as key), the corresponding p-value
                                           is given 
    """

    normality_results = {}
    for column in df.columns:
        stat, p_value = shapiro(df[column].dropna())  # remove NaN values
        normality_results[column] = p_value
    return normality_results


# Extract all descriptors
descriptor_names = [desc[0] for desc in Descriptors._descList]

# Create a MolecularDescriptorCalculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)  # Calculate to integers

# Read file
input_file = 'tested_molecules.csv'
create_dataframe(read_data(input_file))

expanded_df = create_dataframe(read_data(input_file))

print(expanded_df.head())