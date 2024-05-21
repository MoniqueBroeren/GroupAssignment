# Imports
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# Test version
print(rdkit.__version__)

# Test code
smiles = 'COC(=O)c1c[nH]c2cc(OC(C)C)c(OC(C)C)cc2c1=O'
mol = Chem.MolFromSmiles(smiles)
print(mol)

smi = Chem.MolToSmiles(mol)
print(smi)


