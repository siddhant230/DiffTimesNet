# DiffTimesNet

## Step 0: Create a new VirtualEnv and install requirements
```
python -m venv venv

activate for windows:
source venv/Scripts/bin/activate

activate for linux:
. venv/bin/activate

pip install -r requirements.txt
```

## Step 1: Download the data
This script will download the data (all datasets) save in required foder.
```
python get_data.py
```

## Step 2: Train and evaluate model.
 We provide the experiment scripts for all benchmarks under the folder ./scripts/. You can reproduce the experiment results as the following examples:
#### long-term forecast
```bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh```
#### short-term forecast
```bash ./scripts/short_term_forecast/TimesNet_M4.sh```
#### imputation
```bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh```
#### anomaly detection
```bash ./scripts/anomaly_detection/PSM/TimesNet.sh```
#### classification
```bash ./scripts/classification/TimesNet.sh```