import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import torch_geometric
from torch_geometric.data import Dataset,Data
import numpy as np
import os
from rdkit.Chem import rdmolops
from tqdm import tqdm

DATA_PATH = 'data/raw/HIV.csv'
data = pd.read_csv(DATA_PATH)
print(data.head(10))

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return 'HIV.csv'
    
    @property
    def processed_file_names(self):
        return 'not_implemented.pt'
    
    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj= Chem.MolFromSmiles(mol["smiles"])
            
            node_feats = self._get_node_feats(mol_obj)

            edge_feats = self._get_edge_feats(mol_obj)

            edge_index = self._get_edge_index(mol_obj)

            label = self._get_label(mol["HIV_active"])

            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=label, smiles=mol["smiles"])

            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir, f'data_test_{index}.pt'))
                
            else:
                torch.save(data,
                           os.path.join(self.processed_dir, f'data__{index}.pt'))
                
    def _get_node_features(self, mol):

        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = []
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            node_feats.append(atom.GetTotalNumHs())
            node_feats.append(atom.GetNumRadicalElectrons())
            node_feats.append(atom.IsInRing())
            node_feats.append(atom.GetChiralTag())
            
            all_node_feats.append(node_feats)
        
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)
    
    
    def _get_edge_features(self, mol):
        
        all_edge_feats = []
        
        for bond in mol.GetBonds():
            edge_feats = []
            
            edge_feats.append(bond.GetBondTypeAsDouble())
            
            edge_feats.append(bond.IsInRing())
            
            all_edge_feats += [edge_feats, edge_feats]
        
        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)
    
    def _getadjacency_matrix(self, mol):
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
        return torch.tensor(adjacency_matrix, dtype=torch.float)




