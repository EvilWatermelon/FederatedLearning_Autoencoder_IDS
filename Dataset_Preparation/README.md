# Prepare the Dataset
## Download Dataset
Download IoTID20: https://sites.google.com/view/iot-network-intrusion-dataset/home
Download CiC-BoTIoT: https://espace.library.uq.edu.au/view/UQ:c80fccd

## Split in Benign and Anomaly

run 'split_while_dataset.py' 2x times per Datasets. To split into pure Benign and pure Anomaly

## Split in á 1000 Sample splits

run 'split_in_splits.py' for each Subset of pure Benign and Anomaly.

## Split in á 1000 Samples for Training and Server eval

run 'split_in_splits_sorted.py' for pure Benign Subset of each Dataset.

the '_glo' files are used in Server Eval and the resulting Splits used for Training.
