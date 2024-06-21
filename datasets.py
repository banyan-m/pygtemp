import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

DATA_PATH = 'data/HIV.csv'
data = pd.read_csv(DATA_PATH)
print(data.head(10))