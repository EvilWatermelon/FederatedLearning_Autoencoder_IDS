"""if-and-auto: A Flower / PyTorch app."""

import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
#dependencies for autoencoder
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score,roc_auc_score, average_precision_score, matthews_corrcoef
import matplotlib.pyplot as plt
import copy
from sklearn.tree import DecisionTreeClassifier
import psutil
from src.dataset_load import load_centralized_dataset,load_crossdataset
import os
class Autoencoder(nn.Module):
    def __init__(self, input_dim,dropout_rate=0.4): 
        super(Autoencoder, self).__init__()
        # Encoder: Compresses data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),      # Helps training stability
            nn.ReLU(),      
            
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(24, 6)         
        )
        
        # Decoder: Reconstructs data
        self.decoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(0.2),
            
            nn.Linear(48, input_dim) 

        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
rng = np.random.default_rng(seed=42)


def train(net, trainloader, validaton_loader,partition_id, epochs, lr,mu ,device):
    """Train the model on the training set."""
    print_memory_usage()
    net.to(device)
    global_params = copy.deepcopy(net).parameters()
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr, weight_decay=0.05) # weight_decay L2 regularization
    criterion = nn.MSELoss()
    print("training device", partition_id)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    for epoch in range(epochs):
        net.train()
        train_loss = 0
        for data in trainloader:
            proximal_term = 0
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #loss = criterion(outputs, inputs)
            
            #FedProx
            for local_weights, global_weights in zip(net.parameters(), global_params):
                 proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, inputs) + (mu / 2) * proximal_term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/len(trainloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for data in validaton_loader:
                inputs = data[0].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validaton_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss: 
                    best_val_loss = avg_val_loss
                    #best_model_wts = copy.deepcopy(net.state_dict()) 
                    patience_counter = 0
        else:
                    patience_counter += 1
            

        # Stop if overfitting detected
        if patience_counter >= patience:
            print(f'\n Early stopping triggered at epoch {epoch+1}')
            print(f'   Validation loss has not improved for {patience} epochs')
            print(f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            break
    print_memory_usage()    
    avg_trainloss = train_loss/len(trainloader)
    avg_valloss = val_loss / len(validaton_loader)
    print(f'Train Loss: {avg_trainloss:.6f} | Val Loss: {avg_valloss:.6f}')

    return avg_trainloss, avg_valloss    

    


def test(net, X_test_full, X_Validation,partition_id, device,X_train_dt,y_dt):
    """Validate the model on the test set."""
    print_memory_usage()
    net.eval()
    print("testing device", partition_id)
    net.to(device)
    with torch.no_grad():
         
        X_train_dt = torch.FloatTensor(X_train_dt).to(device)
        X_test_full = torch.FloatTensor(X_test_full).to(device)
        
        #encoding layer for classifier 
        X_train_encoded = net.encoder(X_train_dt).detach().numpy()
        X_test_encoded = net.encoder(X_test_full).numpy()
        
        # Validation reconstruction errors
        X_val_tensor = torch.FloatTensor(X_Validation).to(device)
        recon_val = net(X_val_tensor)
        errors_val = torch.mean(torch.abs(X_val_tensor - recon_val), dim=1).cpu().numpy()

        # reconstruction error for classifier
        recon_dt = net(X_train_dt)
        errors_dt = torch.mean(torch.abs(X_train_dt - recon_dt), dim=1).cpu().numpy()

        dt_stats = np.std(X_train_dt.cpu().numpy(), axis=1)
        
        # reconstruction error of full Test set
        X_test_full_tensor = X_test_full
        recon_full = net(X_test_full_tensor)
        errors_full = torch.mean(torch.abs(X_test_full_tensor - recon_full), dim=1).cpu().numpy()
        stats_test = np.std(X_test_full_tensor.cpu().numpy(), axis=1)
        
        mu_val = np.mean(errors_val)
        sigma_val = np.std(errors_val) + 1e-8 
    # For training (used in classifier). Z-score normalisation
    errors_dt_norm = (errors_dt - mu_val) / sigma_val

    # For testing (used in classifier). Z-score normalisation
    errors_full_norm = (errors_full - mu_val) / sigma_val 

    X_features = np.column_stack([
    X_train_encoded,  # AE latent
    errors_dt_norm,  # Recon error
    dt_stats  # Flow stats
    ])  # 8D total with 67 architecture, new architecture 8d since bottleneck is 6
    
    # DecisionTree (DT), alternative to unsupervised Thresholding,small for Tinyml deployment
    dt = DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=42,min_samples_leaf=20)  # 16 leaves max
    
    X_features_test = np.column_stack([X_test_encoded, errors_full_norm, stats_test])
    # DT fit and predict
    dt.fit(X_features, y_dt)  
    y_pred = dt.predict(X_features_test)
    y_proba = dt.predict_proba(X_features_test)[:, 1] 
    # Unsupervised Thresholding, Percentile
    threshold_percentile = np.percentile(errors_val,80)
    print("Threshold percentile 85")
    y_pred_percentile = (errors_full > threshold_percentile).astype(int)
    tr_star = np.mean(errors_val) + np.std(errors_val)
    y_tr_star= (errors_full > tr_star).astype(int)
    print_memory_usage()
    return threshold_percentile, y_pred_percentile ,tr_star, y_tr_star , errors_full, errors_val,y_pred,y_proba,dt,mu_val,sigma_val


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Autoencoder(67)

    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load data set
    # which_dataset 2 for BoTIoT; 3 for IoTID20 just like client

    _,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_centralized_dataset(which_dataset= 1)
    # which_dataset 0 for Training BoTIoT -> Testing IoTID20; everything else for Training IoTID20 -> Testing BoTIoT just like client
    #_,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_crossdataset(which_dataset = 1) 

    # Evaluate the global model on the test set
    threshold, y_pred_percentile, tr_star, y_tr_star, errors_full, errors_val,y_pred,y_proba,dt,mu,sigma = test(
        model,
        X_test_full, X_test_validation, 1000,
        device, X_train_dt,y_dt
    )

    # Metrics
    # Confusion matrix
    cmpercentile = confusion_matrix(y_true, y_pred_percentile)
    cm_dt = confusion_matrix(y_true,y_pred)
    #print(f"\nConfusion Matrix percentile Percentile:")
    #print(cmpercentile)
    #print(f"\nConfusion Matrix Proba:")
    #print(cm_proba)
  
    print(f"\nConfusion Matrix Classify:")
    print(cm_dt)
    tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y_true, y_pred).ravel()
    tn_percentile, fp_percentile, fn_percentile, tp_percentile = confusion_matrix(y_true, y_pred_percentile).ravel()
    # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT
    fprpercentile=cmpercentile[0][1]/ (cmpercentile[0][0]+cmpercentile[0][1])
    fnrpercentile=cmpercentile[1][0]/ (cmpercentile[1][1]+cmpercentile[1][0])
    
    fprdt=cm_dt[0][1] / (cm_dt[0][0]+cm_dt[0][1])
    fnrdt=cm_dt[1][0] / (cm_dt[1][1]+cm_dt[1][0])
    
    # calculate intrusion detection capability
    idc_dt = intrusion_detection_capability(y_true, y_pred)
    idc_percentile = intrusion_detection_capability(y_true, y_pred_percentile)

    print("Intrusion Detection Capability:", "DT",idc_dt,"percentile",idc_percentile)
    
    # PR AUC and ROC AUC for Percentile and DT
    a = average_precision_score(y_true, y_proba) #pr auc; do i need to invert it? finde nicht wirklich aussage kräftig. Sehr hohe werte für Imbalance
    b = roc_auc_score(y_true, y_proba) #ROC AUC is suboptimal; do i need to invert it? seems not dunno.
    c = roc_auc_score(y_true,errors_full)
    d = average_precision_score(y_true,errors_full)
    #print("Testing if it is right ",y_pred[0],y_true[0],y_proba)
    # Clasification Report
    #crpercentile = classification_report(y_true,y_pred_percentile)
    crc = classification_report(y_true,y_pred)
    
    print(crc)
    print(f"ROC AUC DT",b, "ROC AUC Error", c)
    print(f"Fprpercentile",float("{:.2f}".format(fprpercentile)),"Fnrpercentile",float("{:.2f}".format(fnrpercentile)),"FPR DT",float("{:.2f}".format(fprdt)),"FNR DT",
          float("{:.2f}".format(fnrdt)))
    
    # calculate Matthews Corrcoef
    mcc_dt = matthews_corrcoef(y_true,y_pred)
    mcc_percentile = matthews_corrcoef(y_true,y_pred_percentile)
    
    # calculate F1-Score
    f1_dt = f1_score(y_true, y_pred)
    f1_percentile = f1_score(y_true, y_pred_percentile)
    
    # calculate False Discovery Rate
    fdr_dt = fp_dt / (fp_dt + tp_dt) if (fp_dt + tp_dt) > 0 else 0
    fdr_percentile = fp_percentile / (fp_percentile + tp_percentile) if (fp_percentile + tp_percentile) > 0 else 0
    print(f"F1-Score DT: {f1_dt:.4f}",f"F1-Score percentile: {f1_percentile:.4f}")
    print(f"False Discovery Rate DT: {fdr_dt:.4f}", f"False Discovery Rate percentile: {fdr_percentile:.4f}")
    print("MCC DT", mcc_dt, "MCC percentile", mcc_percentile)
    
    # export DecisionTree for inference on ESP32
    def export_tree_to_c(tree, feature_names, filename="tree_model.h"):
          tree_ = tree.tree_

          with open(filename, "w") as f:
              f.write("int predict(float features[]) {\n")

              def recurse(node, depth):
                  indent = "    " * depth
                
                  if tree_.feature[node] != -2:
                      feature = tree_.feature[node]
                      threshold = tree_.threshold[node]
                    
                      f.write(f"{indent}if (features[{feature}] <= {threshold:.6f}f) {{\n")
                      recurse(tree_.children_left[node], depth + 1)
                      f.write(f"{indent}}} else {{\n")
                      recurse(tree_.children_right[node], depth + 1)
                      f.write(f"{indent}}}\n")
                  else:
                      class_id = tree_.value[node].argmax()
                      f.write(f"{indent}return {class_id};\n")

              recurse(0, 1)
              f.write("}\n")

          print(f"Tree exported to {filename}")
    export_tree_to_c(dt, feature_names=["score"])
    # for ESP32 inference, values used for Testing
    with open("values.h", "w") as f:            
              f.write(f"float mu_val = {mu}f\n")
              f.write(f"float sigma_val = {sigma}f\n")
              f.write(f"float threshold = {threshold}f\n")




    # Construct and return reply Message
    # Return the evaluation metrics
    return MetricRecord({
        "threshold": float(threshold),
      #  "threshold tr_star": float(tr_star),#"classification-report": classreport,
        "FPR AE": float("{:.2f}".format(fprpercentile)),
        "FNR AE": float("{:.2f}".format(fnrpercentile)),
        "MCC DT": float("{:.2f}".format(mcc_dt)),
        "MCC AE": float("{:.2f}".format(mcc_percentile)),
        "FPR DT": float("{:.2f}".format(fprdt)),
        "FNR DT": float("{:.2f}".format(fnrdt)),
        "ROC AUC DT": float("{:.2f}".format(b)),
        "ROC AUC AE": float("{:.2f}".format(c)),
        "PR AUC DT": float("{:.2f}".format(a)),
        "PR AUC AE": float("{:.2f}".format(d)),
        "IDC DT": float("{:.2f}".format(idc_dt)),
        "IDC AE": float("{:.2f}".format(idc_percentile)),
        "F1-Score DT": float("{:.4f}".format(f1_dt)),
        "F1-Score AE": float("{:.4f}".format(f1_percentile)),
        "FDR DT": float("{:.4f}".format(fdr_dt)),
        "FDR AE": float("{:.4f}".format(fdr_percentile)),                
        "Confusion Matrix DT": cm_dt.flatten().tolist(),
        "Confusion Matrix AE": cmpercentile.flatten().tolist(),
    })
    
def intrusion_detection_capability(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("TN",tn,"FP",fp,"FN",fn,"TP",tp)
    N = tp + tn + fp + fn

    # probabilities
    p_tp = tp / N
    p_tn = tn / N
    p_fp = fp / N
    p_fn = fn / N

    p_attack = (tp + fn) / N
    p_normal = (tn + fp) / N

    p_pred_attack = (tp + fp) / N
    p_pred_normal = (tn + fn) / N

    # Mutual Information
    terms = []

    if p_tp > 0:
        terms.append(p_tp * np.log2(p_tp / (p_attack * p_pred_attack)))

    if p_fn > 0:
        terms.append(p_fn * np.log2(p_fn / (p_attack * p_pred_normal)))

    if p_fp > 0:
        terms.append(p_fp * np.log2(p_fp / (p_normal * p_pred_attack)))

    if p_tn > 0:
        terms.append(p_tn * np.log2(p_tn / (p_normal * p_pred_normal)))

    mutual_information = sum(terms)

    # Entropy of actual classes
    Hx = 0
    if p_attack > 0:
        Hx -= p_attack * np.log2(p_attack)
    if p_normal > 0:
        Hx -= p_normal * np.log2(p_normal)

    # IDC
    IDC = mutual_information / Hx if Hx != 0 else 0

    return IDC


def print_memory_usage():
    process = psutil.Process(os.getpid())
    # Convert bytes to Megabytes
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Current RAM usage: {mem_mb:.2f} MB")