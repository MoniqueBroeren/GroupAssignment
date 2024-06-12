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


def calculate_descriptors_and_fingerprints(smile):
    """
    Calculates the descriptors and fingerprints for the given SMILES.

    Parameters: smile (str) - SMILES string of molecule
    Returns:    descriptors_and_fp (list) - list of all values that describe the molecule and its binary fingerprints.
    """
    # Extract the molecule
    mol = Chem.MolFromSmiles(smile)

    # If no molecule found, return None
    if mol is None:
        return [None] * len(descriptor_names) + [None] * nBits

    # Calculate descriptors
    descriptors = list(calculator.CalcDescriptors(mol))

    # Calculate binary fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    fingerprints = list(fp)

    return descriptors + fingerprints


def create_dataframe(df_raw):
    """
    Creates a dataframe containing all the descriptors and fingerprints of all the SMILES.

    Parameters: df_raw (DataFrame)      - contains all the raw data
    Returns:    expanded_df (DataFrame) - contains all the raw data and the calculated
                                          molecule descriptor and fingerprint data
    """
    # Calculate descriptors and fingerprints for each molecule
    descriptor_fp_data = df_raw['SMILES'].apply(calculate_descriptors_and_fingerprints)

    # Create a DataFrame with descriptor and fingerprint data
    descriptor_fp_df = pd.DataFrame(descriptor_fp_data.tolist(), columns=descriptor_names + fingerprint_names)

    # Combine the original DataFrame with the descriptor and fingerprint DataFrame
    expanded_df = pd.concat([df_raw, descriptor_fp_df], axis=1)

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

# Number of bits for the binary fingerprints
nBits = 1024  # Default number of bits
fingerprint_names = [f'Bit_{i}' for i in range(nBits)]

# Create a MolecularDescriptorCalculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# # Read file
# input_file = 'tested_molecules.csv'
# expanded_df = create_dataframe(read_data(input_file))

# print(expanded_df.head())

