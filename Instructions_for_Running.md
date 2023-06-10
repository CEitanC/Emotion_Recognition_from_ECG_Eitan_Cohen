## Instructions for Running and Restoring the Project

To run and restore the project, please follow the steps below:
1. Download all Jupiter Notebook and Python files used in the project from the project's GitHub repository: https://github.com/CEitanC/Emotion_Recognition_from_ECG_Eitan_Cohen

2. Obtain the SWEEL samples used in the project by downloading the raw data from the following link: https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624

3. Convert the files in S00 format to CSV format, which will be used for further processing. To do this, navigate to the Extract_samples folder and run the script "run1.m" from the S00 files folder of the project.

4. Preprocess the data and recover the SSL (Self-Supervised Learning) network. Go to the pre_processing_and_recover_network folder and run the notebook "pre_processing_and_SSL_model.ipynb". Note that the sequence of creating the dataset in the NPY file needs to be executed only once, as the existing file will serve as the dataset file for all other runs of the project. This folder also contains several Python files that are partially used in other project notebooks.

5. In the pytorch_model folder, you will find the notebooks related to the PyTorch model and the analysis of intermediate results for the research part. Additionally, the folder includes the "ECG.py" library, which is relevant for future lectures on the PyTorch model in other notebooks.

6. The TS_models folder contains the notebooks for the research part of the project, including the models of Inception Time and the Transformer, as well as the notebooks for performance analysis of the models.

By following these steps, you should be able to successfully run and restore the project.
