from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from model import GNN
from torch_geometric.data import DataLoader
import mlflow.pytorch
from dataset_featurizer import MoleculeDataset
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


from mango import Tuner, scheduler
from config import HYPERPARAMETERS, BEST_HYPERPARAMETERS, SIGNATURE


def run_one_training(params):
    params = params[0]

    with mlflow.start_run() as run:

        for key in params.keys():
            mlflow.log_param(key, params[key])

        
        print("loading dataset")
        train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
        train_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)

        params["module_edge_dim"] = train_dataset[0].edge_attr.shape[1]

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

        print("loading model")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"], 
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        best_loss = 1000
        early_stopping_counter = 0

        for epoch in range(300):
            if early_stopping_counter == 10:
            
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss: {loss}")
                mlflow.log_metric(key="Train loss", value= float(loss), step=epoch)
                
                model.eval()

                if epoch % 5 == 0:
                    loss = test(epoch, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss: {loss}")
                    mlflow.log_metric(key="Test loss", value= float(loss), step=epoch)

                    if float(loss) < best_loss:
                        best_loss = float(loss)
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0

                    else:
                        early_stopping_counter += 1

                scheduler.step()

            else:
                print("Early stopping due to imoprovement")
                return [best_loss]
    print(f"Finishing training with best loss: {best_loss}")
    return [best_loss]

print("Starting hyperparameter optimization")
config = dict()
config["optimizer"] = "Bayesian"
config["num_iteration"] = 100

tuner = Tuner(HYPERPARAMETERS, 
              objective = run_one_training, 
              conf_dict = config)

results = tuner.minimize()
    

                


    