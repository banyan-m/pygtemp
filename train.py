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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch,model,train_loader,optimizer,loss_fn):

    all_preds = []
    all_labels = []
    running_loss = 0.0
    steps = 0

    for _, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        steps += 1

        all_preds.append(np.rint(torch.sigmoid(pred).detach().cpu().numpy()))
        all_labels.append(batch.y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels,epoch, "train")
    return running_loss / steps


def test(epoch,test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        
        running_loss += loss.item()
        steps += 1

        all_preds.append(np.rint(torch.sigmoid(pred).detach().cpu().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).detach().cpu().numpy())
        all_labels.append(batch.y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    
    return running_loss / steps

def log_conf_matrix(y_pred, y_true, epoch):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["0", "1"]
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    cfm_plot.figure.savefig(f"data/images/cm_{epoch}.png")
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matric: \n {confusion_matrix(y_pred, y_true)}")
    print(f"\n F1 score: {f1_score(y_true, y_pred)}")
    print(f"\n Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n Precision: {prec}")
    print(f"\n Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)

    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f" ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)

    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print("ROC AUC score could not be calculated")


from mango import scheduler, Tuner
from cinfig import HYPERPARAMETERS, BEST_HYPERPARAMETERS, SIGNATURE
    