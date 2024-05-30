# Imports
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors


def read_data(infile):
    """
    Reads the input (.csv) file.
    :param infile: (str) of the path where the file is located
    :return: (pandas dataframe) of the raw data
    """
    df_raw = pd.read_csv(infile)
    return df_raw


def calculate_descriptors(smile):
    """
    Calculates the descriptors for the given smiles.
    :param smile: (str) of the smile
    :return: (list) of all the descriptors
    """
    # Extract the molecule
    mol = Chem.MolFromSmiles(smile)
    # If no molecule found return None
    if mol is None:
        return [None] * len(descriptor_names)
    return list(calculator.CalcDescriptors(mol))


def create_dataframe(df_raw):
    """
    Creates a dataframe containing all the descriptors of all the smiles.
    :param df_raw: (pandas dataframe) of the raw data
    :return: (pandas dataframe) the expanded dataframe containing all the descriptors of all the smiles
    """
    # Calculate descriptors for each molecule
    descriptor_data = df_raw['SMILES'].apply(calculate_descriptors)

    # Create a DataFrame with descriptor data
    descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptor_names)

    # Combine the original DataFrame with the descriptor DataFrame
    expanded_df = pd.concat([df_raw, descriptor_df], axis=1)

    return expanded_df


# Extract all descriptors
descriptor_names = [desc[0] for desc in Descriptors._descList]

# Create a MolecularDescriptorCalculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)  # Calculate to integers

# Read file
input_file = 'tested_molecules.csv'
create_dataframe(read_data(input_file))

expanded_df = create_dataframe(read_data(input_file))

print(expanded_df.head())