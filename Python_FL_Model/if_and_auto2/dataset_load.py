"""if-and-auto: A Flower / PyTorch app."""

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def log1p_static_scale(df):
    """
    Applies Log1p and divides by a static constant.
    Prevents feature misalignment in Non-IID scenarios.
    """
    # 1. Select only numeric columns (exclude labels/IDs)
    # If you already dropped them, you can just use df.columns
    if hasattr(df, 'values'):
            X = df.values
    else:
        X = df
    
    # 2. Apply Log1p: ln(1 + x)
    # This compresses the massive range of flow duration/bytes
    # 0 -> 0.0
    # 1,000,000 -> 13.8
    # 100,000,000 -> 18.4
    X = np.log1p(np.clip(X, 0, None))
    
    # 3. Divide by Static Constant (20.0)
    # We use 20.0 because log(Max Network Value) is roughly 18-19.
    # This forces almost all data into the [0.0, 1.0] range.
    X_scaled = X /20
    
    return X_scaled.astype(np.float32)
rng = np.random.default_rng(seed=42)

def load_cross_data(partition_id: int, num_partitions: int, which_dataset: int):
    if which_dataset == 0:


        
        dfs=[]
        j = partition_id *5
        if (partition_id >= 2):
            for i in range(5):
                
            
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+60+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        else:    
            for i in range(5):
                
                
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
    

        df_attack = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_{partition_id+1}.csv")
            
        
        attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_{partition_id+1}.csv")

    

        bengin = pd.read_csv(f"small_IoTID20_dataset_benign_clean/split_{partition_id+1}.csv")
        print("Training BoTIoT, Testing IoTID20")
    elif which_dataset == 1:
        """Load partition IoTID20 data."""

        dfs=[]
        j = partition_id *5
        if (partition_id >= 2):
            for i in range(5):
                
            
                df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/split_{partition_id+8+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        else:    
            for i in range(5):
                
                
                df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
    

        df_attack = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_{partition_id+1}.csv")
        
            
        attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_{partition_id+1}.csv")

        bengin = pd.read_csv(f"small_BoTIoT_dataset_benign_clean/split_{partition_id+1}.csv") 
        print("Training IoTID20, Testing BoTIoT")


    X_test_attacks = pd.concat([bengin,attacks.sample(n= int((len(bengin)*0.1)),random_state=42)])


    print(f"Before Training {partition_id}",len(df_benign))
    print(f"Before Testing {partition_id}",len(X_test_attacks))
    # Features that are not used 
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']


    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    df_attack.columns = df_attack.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')    
    
    # "Feature Selection"
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)

    
    
    df_attack = df_attack.drop([col for col in label_col if col in df_attack.columns], axis=1)

    
    # Remove features
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)


    # label attack and benign data of Testing data for evaluation 
    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0)
    else: 
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # remove features
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    df_attack = df_attack.drop(columns=["Label"])
    print(f"After Training {partition_id}",len(df_benign))
    print(f"After Testing {partition_id}",len(X_test_attacks))
    # this is just to check
    num_ones = target.sum()
    num_zeros = len(target) - num_ones

    print(f"Benign Testing {partition_id}",num_zeros)
    print(f"Anomaly Testing {partition_id}",num_ones)
    # can add earlier to make the label_col smaller. Makes it so the feature names are written the same
  
    # split training data into Training and validation set
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_benign_class   = train_test_split(X_train_, test_size=0.2,random_state=42)


    # Scaling
    scaler = StandardScaler() # StandardScaler(clip = True) #it does combat the exploding values of val loss and tr_star threshold, but is it good for anomaly detection? no it shouldn't be
    X_train = log1p_static_scale(X_train_)
    
    X_Validation = log1p_static_scale(X_val)
    X_benign_class = log1p_static_scale(X_benign_class)

    df_attack = df_attack.sample(len(X_benign_class),random_state = 42)
    df_attack = log1p_static_scale(df_attack)
    
    X_train_classifier = np.vstack([X_benign_class, df_attack])
    y_class = np.hstack([
    np.zeros(len(X_benign_class)),  # Benign holdout
    np.ones(len(df_attack))   # All attacks
    ])
    #
    y_true = target
    
    # Reuse same scaler
    print("Training Data amount:",len(X_train))
    X_test_full = log1p_static_scale(X_test_attacks)

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))
  
    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_classifier,y_class


def load_mono_dataset(partition_id: int, num_partitions: int,which_dataset:int):
    """Load partition BoTIoT data."""



    if which_dataset ==0 :
        dfs = []
        i=0
        j = partition_id *5
        if (partition_id >= 2):
            for i in range(5):
                
            
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+60+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        else:    
            for i in range(5):
                
                
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        print(f"Client {partition_id} Mean Duration: {df_benign['Flow Duration'].mean()}")

        X_test_attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_{partition_id+1}.csv")
        print("load_data_BoTIoT split")
    elif which_dataset ==1 :
        dfs = []
        i=0
        j = partition_id *5
        if (partition_id >= 2):
            for i in range(5):
                
            
                df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/split_{partition_id+8+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        else:    
            for i in range(5):
                
                
                df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        print(f"Client {partition_id} Mean Duration: {df_benign['Flow_Duration'].mean()}")


        X_test_attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_{partition_id+1}.csv")
        print("load_data_IoTID20 Split")
    
        
        
        
    # Handle NaNs / Infs
    print(f"Before Training {partition_id}",len(df_benign))
    print(f"Before Testing {partition_id}",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    


    
    # Remove constant / near-constant features
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)

    # Clean data
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)

    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    else:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0) #brauch ich nicht in dieser Function
        
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    print(f"After Training {partition_id}",len(df_benign))
    print(f"After Testing {partition_id}",len(X_test_attacks))

    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_test   = train_test_split(X_train_, test_size=0.25,random_state=42)

    scaler = StandardScaler()  # StandardScaler(clip = True) #it does combat the exploding values of val loss and tr_star threshold, but is it good for anomaly detection? no it shouldn't be
    X_train = log1p_static_scale(X_train_)

    X_Validation = log1p_static_scale(X_val)
    sample_attack = int(len(X_test)*0.1)
    X_test_attacks = rng.choice(X_test_attacks , size=sample_attack, axis=0, replace=False)
    X_test_full = np.vstack([X_test, X_test_attacks])
    # Reuse same scaler
    y_true_early = np.hstack([
    np.zeros(len(X_test)),  # Benign holdout
    np.ones(len(X_test_attacks))   # All attacks
    ])

    X_test_full = log1p_static_scale(X_test_full)
    X_test_full,X_train_classifier,y_true, y_class = train_test_split(X_test_full,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)

    print("Samples train",len(X_train),"val",len(X_Validation),"X_train_classifier",len(X_train_classifier),"attacksx",len(X_test_attacks))
    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_classifier,y_class



def load_centralized_dataset(which_dataset):
    """Load partition IoTID20 data."""

    global fds_counter
    global fds
    global ads
    

    if which_dataset ==0 :
        dfs=[]
        i=0
        for i in range(5):
            
             
            df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)
        print(f"Client Mean Duration: {df_benign['Flow Duration'].mean()}")

        X_test_attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")
        calibration_split = pd.concat([pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv"),pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_8.csv")],ignore_index=True)

        print("load_data_BoTIoT 90/10")
    elif which_dataset ==1 :
        dfs=[]
        i=0
        for i in range(5):
            
             
            df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)
    

        X_test_attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")
        calibration_split = pd.concat([pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_15.csv"),pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_8.csv")],ignore_index=True)
        print("load_data_IoTID20 90/10")
    

    print(f"Before Training ",len(df_benign))
    print(f"Before Testing",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    


    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')
    calibration_split.columns = calibration_split.columns.str.replace('_', ' ')
    calibration_split = calibration_split.groupby('Label').sample(n=50, random_state=42)
    
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)

    # Clean data
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)
    #X_test_attacks = X_test_attacks[available_features]
    calibration_split = calibration_split.drop([col for col in label_col if col in calibration_split.columns], axis=1)

    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    else:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0) #brauch ich nicht in dieser Function
        
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    calibration_split = calibration_split.drop(columns=["Label"])
    print(f"After Training",len(df_benign))
    print(f"After Testing",len(X_test_attacks))
    

    # Split df_benign into Validation(X_val) and Benign data for Test(X_test)
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_test   = train_test_split(X_train_, test_size=0.25,random_state=42)

    
    scaler = StandardScaler() # StandardScaler(clip = True) #it does combat the exploding values of val loss and tr_star threshold, but is it good for anomaly detection? no it shouldn't be
    X_train = log1p_static_scale(X_train_)
    
    calibration_split = log1p_static_scale(calibration_split)

    X_Validation = log1p_static_scale(X_val)
    
    # changing the Ratio of how many Anomaly samples in Testing
    sample_attack = int(len(X_test)*0.1)    
    X_test_attacks = rng.choice(X_test_attacks , size=sample_attack, axis=0, replace=False)

    X_test_full = np.vstack([X_test, X_test_attacks])
    # Reuse same scaler
    y_true_early = np.hstack([
    np.zeros(len(X_test)),  # Benign holdout
    np.ones(len(X_test_attacks))   # All attacks
    ])

    #X_test_benign = log1p_static_scale(X_test.values)
    X_test_full = log1p_static_scale(X_test_full)
    X_test_full,X_train_classifier,y_true, y_class = train_test_split(X_test_full,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early)

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))

    lenge = len(X_test_full)
#Exporting Datasets into .h files for MCU deployment
    y_mcu = y_true
# --- Write the C Header File ---
    filename = f"mcu_test_data.h"
    print(f"Exporting {lenge} samples to {filename}...")

    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_TEST_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_NUM_FEATURES = {X_test_full.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_test_x[{lenge}][{X_test_full.shape[1]}] = {{\n")
        for i, row in enumerate(X_test_full):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.6f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Write Y data (Labels)
        f.write(f"const int mcu_test_y[{lenge}] = {{\n    ")
        y_str = ", ".join([str(int(val)) for val in y_mcu])
        f.write(y_str)
        f.write("\n};\n")
# --- Write the C Header File Calibration--- 
    filename = f"calibration_data.h"    
    lenge = len(calibration_split)
   
    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_TEST_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_NUM_FEATURES = {calibration_split.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_test_x[{lenge}][{X_test_full.shape[1]}] = {{\n")
        for i, row in enumerate(calibration_split):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.6f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

    print("Export complete!")

    
    
    print("Samples train",len(X_train),"val",len(X_Validation),"X_train_classifier",len(X_train_classifier),"attacksx",len(X_test_attacks))

    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_classifier,y_class,scaler

def load_crossdataset(which_dataset):
    """Load partition BoTIoT data."""
    if which_dataset == 0:

        dfs = []
        for i in range(5):
            
             
            df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)

    

        df_attack = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")
                    
        attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")

        bengin = pd.read_csv(f"small_IoTID20_dataset_benign_clean/split_1.csv")
        print("load_data")
    if which_dataset == 1:
        """Load partition IoTID20 data."""
        #os.chdir("B:/B_Projekt2/CiC/IoTID20/small_IoTID20_dataset_benign_clean_noleak")
        dfs = []

        for i in range(5):
            
             
            df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)

    
        #os.chdir("B:/B_Projekt2/CiC/IoTID20/small_IoTID20_dataset_allattacks_clean")

        df_attack = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")
        
    
            
        #os.chdir("B:/B_Projekt2/CiC/BoTIoT/small_BoTIoT_dataset_allattacks_clean")
        
        attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")

    
        #os.chdir("B:/B_Projekt2/CiC/BoTIoT/small_BoTIoT_dataset_benign_clean")

        bengin = pd.read_csv(f"small_BoTIoT_dataset_benign_clean/split_1.csv") 
        print("load_data2")

    X_test_attacks = pd.concat([bengin,attacks.sample(n= int((len(bengin)*0.1)),random_state=42)])

    print(f"Before Training ",len(df_benign))
    print(f"Before Testing ",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    df_attack.columns = df_attack.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')    
    
    # "Feature Selection"
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)

    
    df_attack = df_attack.drop([col for col in label_col if col in df_attack.columns], axis=1)

    
    # Remove features
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)

    # label attack and benign data of Testing data for evaluation 
    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0)
    else: 
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # remove features
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    df_attack = df_attack.drop(columns=["Label"])
    print(f"After Training ",len(df_benign))
    print(f"After Testing ",len(X_test_attacks))
    # this is just to check
    num_ones = target.sum()
    num_zeros = len(target) - num_ones

    print(f"Benign Testing ",num_zeros)
    print(f"Anomaly Testing ",num_ones)
    # can add earlier to make the label_col smaller. Makes it so the feature names are written the same
  
    # split training data into Training and validation set
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_benign_class   = train_test_split(X_train_, test_size=0.2,random_state=42)


    # Scaling
    scaler = StandardScaler() # StandardScaler(clip = True) #it does combat the exploding values of val loss and tr_star threshold, but is it good for anomaly detection? no it shouldn't be
    X_train = log1p_static_scale(X_train_)

    X_Validation = log1p_static_scale(X_val)
    X_benign_class = log1p_static_scale(X_benign_class)

    df_attack = df_attack.sample(len(X_benign_class),random_state = 42)
    df_attack = log1p_static_scale(df_attack)
    
    X_train_classifier = np.vstack([X_benign_class, df_attack])
    y_class = np.hstack([
    np.zeros(len(X_benign_class)),  # Benign holdout
    np.ones(len(df_attack))   # All attacks
    ])
    #
    y_true = target
    
    # Reuse same scaler
    print("Training Data amount:",len(X_train))
    X_test_full = log1p_static_scale(X_test_attacks)

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))



    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_classifier,y_class


    
