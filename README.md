Repository for **Kaggle TPS August 2021**

## Steps to execute:

1. Clone the source code from github under <PROJECT_HOME> directory.
> git clone https://github.com/arnabbiswas1/k_tab_sept_roc_auc_binary_classification_KFold.git

This will create the following directory structure:
> <PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold

2. Create conda env:
> conda env create --file environment.yml

3. Go to the raw data directory at `<PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold/data/raw`. Download dataset from Kaggle:
> kaggle competitions download -c tabular-playground-series-sep-2021

4. Unzip the data:
> unzip tabular-playground-series-sep-2021.zip

5. Set the value of variable `HOME_DIR` at `<PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold/src/config/constants.py` with the absolute path of `<PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold`

6. To process raw data into parquet format, go to `<PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold/src`. Execute the following:
> python -m scripts.process_raw_data

7. To trigger feature engineering, go to `<PROJECT_HOME>/k_tab_sept_roc_auc_binary_classification_KFold/src`. Execute the following:
> python -m scripts.create_fetaures


## Note:

Following is needed for visualizing plots for optuna using plotly (i.e. plotly dependency):

> jupyter labextension install jupyterlab-plotly@4.14.3
