"""if-and-auto: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from if_and_auto2.task import Autoencoder
from if_and_auto2.task import test as test_fn
from if_and_auto2.task import train as train_fn
from if_and_auto2.dataset_load import load_cross_data,load_mono_dataset
#dependencies for autoencoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve,roc_curve,auc,average_precision_score,classification_report

import time
# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Training Time
    start_time = time.time()

    # Load the model and initialize it with the received weights
    input_dim: int = context.run_config["input-dim"]

    model = Autoencoder(input_dim)

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    which_dataset: int = context.run_config["which_dataset"]

    partition = partition_id 
    # which_dataset = 0 is TonIoT; everything else is IoTID20 just like 
    #trainloader, validaton_loader, _ ,_, _,_,_ = load_mono_dataset(partition, num_partitions,which_dataset=which_dataset) # do i need dynamic batchsize? WiP
    # which_dataset 0 for Training TonIoT -> Testing IoTID20; everything else for Training IoTID20 -> Testing TonIoT
    trainloader, validaton_loader, _ ,_, _,_,_ = load_cross_data(partition, num_partitions,which_dataset=1) # do i need dynamic batchsize? WiP

    # Call the training function
    train_loss, val_loss = train_fn(
        model, # Model Autoencoder
        trainloader,validaton_loader, # training and validation set
        partition, # Partition id
        context.run_config["local-epochs"],# epochs
        msg.content["config"]["lr"], #learning rate
        context.run_config["mu"], #mu for Fedprox
        device,
    )
    end_time = time.time()
    training_time = end_time - start_time
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num-examples": len(trainloader),
        "avg_training_time": float("{:.2f}".format(training_time)),  # New metric
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record,})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Testing Time
    start_time = time.time()

    # Load the model and initialize it with the received weights

    model = Autoencoder(67)

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    which_dataset: int = context.run_config["which_dataset"]

    partition = partition_id 
    """ which_dataset = 0 is TonIoT; everything else is IoTID20 """
    #_,_,X_test_full, X_test_validation, y_true,X_train_classifier,y_class = load_mono_dataset(partition, num_partitions,which_dataset=which_dataset)  # do i need dynamic batchsize? WiP
    """ which_dataset 0 for Training TonIoT -> Testing IoTID20; everything else for Training IoTID20 -> Testing TonIoT """
    _,_,X_test_full, X_test_validation, y_true,X_train_classifier,y_class = load_cross_data(partition, num_partitions,which_dataset=1) 
    # Call the evaluation function
    threshold, y_pred_90, tr_star, y_tr_star, errors_full, errors_val,y_pred,y_proba,_,_,_ = test_fn(
        model, # Autoencoder
        X_test_full, X_test_validation, # Data for Testing
        partition, # Partition ID
        device, 
        X_train_classifier,y_class # Data for Decision Tree
    )

 
    # Metrics for Client Eval
    # Confusion matrix
    cm90 = confusion_matrix(y_true, y_pred_90)
    cm_class = confusion_matrix(y_true,y_pred)
 
  
    print(f"\nConfusion Matrix Classify and Device Number {partition}:")
    print(cm_class)
    
    # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT
    fpr90=cm90[0][1]/ (cm90[0][0]+cm90[0][1])
    fnr90=cm90[1][0]/ (cm90[1][1]+cm90[1][0])

    fprclass=cm_class[0][1] / (cm_class[0][0]+cm_class[0][1])
    fnrclass=cm_class[1][0] / (cm_class[1][1]+cm_class[1][0])

    #  ROC AUC for Percentile and DT
    b = roc_auc_score(y_true, y_proba) 
    c = roc_auc_score(y_true,errors_full)

    print(f"Device {partition} ROC AUC DT",b, "ROC AUC Error", c)
    print(f"Device {partition} Fpr90",float("{:.2f}".format(fpr90)),"Fnr90",float("{:.2f}".format(fnr90)),"FPR DT",
          float("{:.2f}".format(fprclass)),"FNR DT",float("{:.2f}".format(fnrclass)))



    end_time = time.time()
    testing_time = end_time - start_time
    
    # Construct and return reply Message
    # Return the evaluation metrics
    metrics = {
        "threshold": float(threshold),
       # "threshold tr_star": float(tr_star),#"classification-report": classreport,
        "fpr90": float("{:.2f}".format(fpr90)),
        "fnr90": float("{:.2f}".format(fnr90)),
       # "fprstar": float("{:.2f}".format(fprstar)),
      #  "fnrstar": float("{:.2f}".format(fnrstar)),
        "FPR DT": float("{:.2f}".format(fprclass)),
        "FNR DT": float("{:.2f}".format(fnrclass)),
        #"PR AUC": float("{:.2f}".format(a)),
        "ROC AUC DT": float("{:.2f}".format(b)),
        "ROC AUC Error": float("{:.2f}".format(c)),
       # "PR AUC Error": float("{:.2f}".format(d)),
       # "Precision":float(("{:.2f}".format(pr_precísion))),
        "avg_testing_time": float("{:.2f}".format(testing_time)),  # New metric
        "num-examples": len(y_true),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
