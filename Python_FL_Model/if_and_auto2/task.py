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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,roc_auc_score, average_precision_score, matthews_corrcoef
import matplotlib.pyplot as plt
import copy
from sklearn.tree import DecisionTreeClassifier

from if_and_auto2.dataset_load import load_centralized_dataset,load_crossdataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim,dropout_rate=0.4): #input dimension hardcoded
        super(Autoencoder, self).__init__()
# Encoder: Compresses data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),      # Helps training stability
            nn.LeakyReLU(0.2),       # LeakyReLU prevents "dead neurons"
            
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(24, 6)         
        )
        
        # Decoder: Reconstructs data
        self.decoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.2),
            
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.2),
            
            nn.Linear(48, input_dim) 

        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
rng = np.random.default_rng(seed=42)


def train(net, trainloader, validaton_loader,partition_id, epochs, lr,mu ,device):
    """Train the model on the training set."""
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
        #print(f'Epoch {epoch+1}, Loss: {train_loss/len(trainloader):.4f}')
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
        
    avg_trainloss = train_loss/len(trainloader)
    avg_valloss = val_loss / len(validaton_loader)
    print(f'Train Loss: {avg_trainloss:.6f} | Val Loss: {avg_valloss:.6f}')

    return avg_trainloss, avg_valloss    

    


def test(net, X_test_full, X_Validation,partition_id, device,X_train_classifier,y_class):
    """Validate the model on the test set."""
    
    net.eval()
    print("testing device", partition_id)
    net.to(device)
    with torch.no_grad():
         
        X_train_classifier= torch.FloatTensor(X_train_classifier).to(device)
        X_test_full = torch.FloatTensor(X_test_full).to(device)
        #encoding layer for classifier 
        X_train_encoded = net.encoder(X_train_classifier).detach().numpy()
        X_test_encoded = net.encoder(X_test_full).numpy()
        # Validation reconstruction errors
        X_val_tensor = torch.FloatTensor(X_Validation).to(device)
        recon_val = net(X_val_tensor)
        errors_val = torch.mean(torch.abs(X_val_tensor - recon_val), dim=1).cpu().numpy()

        # reconstruction error for classifier
        recon_class = net(X_train_classifier)
        errors_class = torch.mean(torch.abs(X_train_classifier - recon_class), dim=1).cpu().numpy()

        class_stats = np.std(X_train_classifier.cpu().numpy(), axis=1)
        
        # reconstruction error of full Test set
        X_test_full_tensor = X_test_full
        recon_full = net(X_test_full_tensor)
        errors_full = torch.mean(torch.abs(X_test_full_tensor - recon_full), dim=1).cpu().numpy()
        stats_test = np.std(X_test_full_tensor.cpu().numpy(), axis=1)
        
        mu_val = np.mean(errors_val)
        sigma_val = np.std(errors_val) + 1e-8 
    # For training (used in classifier)
    errors_class_norm = (errors_class - mu_val) / sigma_val

    # For testing (used in classifier)
    errors_full_norm = (errors_full - mu_val) / sigma_val 

    X_features = np.column_stack([
    X_train_encoded,  # AE latent
    errors_class_norm,  # Recon error
    class_stats  # Flow stats
    ])  # 10D total with 67 architecture, new architecture 4d since bottleneck is 4
    
    # DecisionTree (DT), alternative to unsupervised Thresholding,small for Tinyml deployment
    dt = DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=42,min_samples_leaf=20)  # 16 leaves max

    X_features_test = np.column_stack([X_test_encoded, errors_full_norm, stats_test])
    # DT fit and predict
    dt.fit(X_features, y_class)  
    y_pred = dt.predict(X_features_test)
    y_proba = dt.predict_proba(X_features_test)[:, 1] 
    # Unsupervised Thresholding, Percentile
    threshold_90 = np.percentile(errors_val,80)
    print("Threshold percentile 85")
    y_pred_90 = (errors_full > threshold_90).astype(int)
    tr_star = np.mean(errors_val) + np.std(errors_val)
    y_tr_star= (errors_full > tr_star).astype(int)

    return threshold_90, y_pred_90 ,tr_star, y_tr_star , errors_full, errors_val,y_pred,y_proba,dt,mu_val,sigma_val


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Autoencoder(67)

    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load data set
    # which_dataset 2 for BoTIoT; 3 for IoTID20 just like client

    #_,_,X_test_full, X_test_validation, y_true,X_train_classifier,y_class,scaler = load_centralized_dataset(which_dataset= 0)
    # which_dataset 0 for Training BoTIoT -> Testing IoTID20; everything else for Training IoTID20 -> Testing BoTIoT just like client
    _,_,X_test_full, X_test_validation, y_true,X_train_classifier,y_class = load_crossdataset(which_dataset = 1) 

    # Evaluate the global model on the test set
    threshold, y_pred_90, tr_star, y_tr_star, errors_full, errors_val,y_pred,y_proba,dt,mu,sigma = test(
        model,
        X_test_full, X_test_validation, 1000,
        device, X_train_classifier,y_class
    )

    # Metrics
    # Confusion matrix
    cm90 = confusion_matrix(y_true, y_pred_90)
    cm_class = confusion_matrix(y_true,y_pred)
    #print(f"\nConfusion Matrix 90 Percentile:")
    #print(cm90)
    #print(f"\nConfusion Matrix Proba:")
    #print(cm_proba)
  
    print(f"\nConfusion Matrix Classify:")
    print(cm_class)
    
    # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT
    fpr90=cm90[0][1]/ (cm90[0][0]+cm90[0][1])
    fnr90=cm90[1][0]/ (cm90[1][1]+cm90[1][0])
    
    fprclass=cm_class[0][1] / (cm_class[0][0]+cm_class[0][1])
    fnrclass=cm_class[1][0] / (cm_class[1][1]+cm_class[1][0])
    # Exam

    idc_dt = intrusion_detection_capability(y_true, y_pred)
    idc_90 = intrusion_detection_capability(y_true, y_pred_90)

    print("Intrusion Detection Capability:", "DT",idc_dt,"90",idc_90)
    
    # PR AUC and ROC AUC for Percentile and DT
    a = average_precision_score(y_true, y_proba) #pr auc; do i need to invert it? finde nicht wirklich aussage kräftig. Sehr hohe werte für Imbalance
    b = roc_auc_score(y_true, y_proba) #ROC AUC is suboptimal; do i need to invert it? seems not dunno.
    c = roc_auc_score(y_true,errors_full)
    d = average_precision_score(y_true,errors_full)
    #print("Testing if it is right ",y_pred[0],y_true[0],y_proba)
    # Clasification Report
    #cr90 = classification_report(y_true,y_pred_90)
    crc = classification_report(y_true,y_pred)
    
    print(crc)
    print(f"ROC AUC DT",b, "ROC AUC Error", c)
    print(f"Fpr90",float("{:.2f}".format(fpr90)),"Fnr90",float("{:.2f}".format(fnr90)),"FPR DT",float("{:.2f}".format(fprclass)),"FNR DT",
          float("{:.2f}".format(fnrclass)))
    mcc_dt = matthews_corrcoef(y_true,y_pred)
    mcc_90 = matthews_corrcoef(y_true,y_pred_90)
    print("MCC DT", mcc_dt, "MCC 90", mcc_90)
    #export DecisionTree for inference on mcu
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
    # for MCU inference, values used for Testing
    with open("values.h", "w") as f:            
              f.write(f"float mu_val = {mu}f\n")
              f.write(f"float sigma_val = {sigma}f\n")
              f.write(f"float threshold = {threshold}f\n")




    # Construct and return reply Message
    # Return the evaluation metrics
    return MetricRecord({
        "threshold": float(threshold),
      #  "threshold tr_star": float(tr_star),#"classification-report": classreport,
        "fpr90": float("{:.2f}".format(fpr90)),
        "fnr90": float("{:.2f}".format(fnr90)),
        "MCC DT": float("{:.2f}".format(mcc_dt)),
        "MCC 90": float("{:.2f}".format(mcc_90)),
        "FPR DT": float("{:.2f}".format(fprclass)),
        "FNR DT": float("{:.2f}".format(fnrclass)),
        "ROC AUC DT": float("{:.2f}".format(b)),
        "ROC AUC Error": float("{:.2f}".format(c)),
        "PR AUC DT": float("{:.2f}".format(a)),
        "PR AUC Error": float("{:.2f}".format(d)),
        "IDC DT": float("{:.2f}".format(idc_dt)),
        "IDC Error": float("{:.2f}".format(idc_90)),
        "num-examples": len(y_true),
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


