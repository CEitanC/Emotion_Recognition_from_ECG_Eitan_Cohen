## Instructions for Running and Restoring the Project

To run and restore the project, please follow the steps below:

1. Download all Jupyter Notebook and Python files used in the project from the project's GitHub repository: [Emotion Recognition from ECG](https://github.com/CEitanC/Emotion_Recognition_from_ECG_Eitan_Cohen).
 Alternatively, you can run the command in the  terminal: `git clone https://github.com/CEitanC/Emotion_Recognition_from_ECG_Eitan_C`
2. To recover the project as it was done, follow the instructions in each directory's README file, pay attention to any additional dependencies, and use the provided create_folders.sh script for ensuring a valid working area. To ensure the create_folders.sh script is executable, use the `chmod +x create_folders.sh` command before executing it.
3. Use the CSV files located in the `pre_processing_and_recover_network` directory inside the `csv_files` directory. Alternatively, you can create them yourself by following these steps:
   - Download the raw data from the following link: [Raw Data](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624).
   - Convert the files in S00 format to CSV format, which will be used for further processing. To do this, navigate to the `Extract_samples` folder and run the script `main.m` from the S00 files folder of the project. Refer to the instructions in the readme file located in the `Extract_samples` folder for more details.

4. Preprocess the data and recover the SSL (Self-Supervised Learning) network. Go to the `pre_processing_and_recover_network` folder and run the notebook `pre_processing_and_SSL_model.ipynb`. Follow the instructions provided in the readme file of the `pre_processing_and_recover_network` folder.

5. Use the `pytorch_model` folder for recovering the fully-supervised network from the paper. You will find the notebooks related to the PyTorch model and the analysis of intermediate results for the research part. Please refer to the readme file inside the folder for further instructions.

6. Finally, go to the `TS_models` folder. It contains the notebooks for the research part of the project, including the models of Inception Time and the Transformer, as well as the notebooks for performance analysis of the models. In this folder, you will find a comparison of all the models, including splitting the samples randomly and splitting by the identification of the subject. Follow the instructions provided in the readme file of the `TS_models` folder for detailed guidance.

Please note that the instructions above assume you have the necessary dependencies and libraries installed to run the project successfully. Make sure to fulfill any additional requirements mentioned in the respective readme files within each folder.
