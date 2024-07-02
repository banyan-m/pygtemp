from sklearn.metrics import confusion_matrix, f1score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")