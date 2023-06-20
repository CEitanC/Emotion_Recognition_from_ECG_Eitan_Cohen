# Create the Dataset - Need to Do One Time

## 1. CSV Files
Make sure you have the CSV files after following the steps in the Extract_samples directory. Locate them in this directory inside an internal directory named csv_files.

## 2. Open Jupyter Notebook
Open the `pre_processing_and_SSL_model.ipynb` Jupyter Notebook.

## 3. Run Data Extraction
Run the section titled "Extract the Data - Need Only One Time for Creating the SWELL Dataset" in the notebook.

## 4. Swell Dataset
The `swell_dict.npy` file will appear inside the `./swell_dataset` folder.


# Recover SSL Model

## 1. Open Jupyter Notebook
Open the `pre_processing_and_SSL_model.ipynb` Jupyter Notebook.

## 2. Create Empty Folders
Create the following empty folders inside this directory:
- data_folder
- summaries
- output
- models

## 3. Run SSL Model
Run the section titled "Run SSL Model" in the notebook.

## 4. Results
The results of the running will appear in the `STR` folders, which will be created during the process.
