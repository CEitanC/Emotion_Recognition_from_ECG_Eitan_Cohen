#  Directory Overview

This directory includes the following:

- Implementation and training of time series models - InceptionTime and Transformer.
- Comparison of InceptionTime, Transformer, and the conventional network from the paper:
    1) With the same random splitting of the samples into train and test sets.
    2) With the splitting of the samples into train and test sets based on the subject's identification.
- Examination of InceptionTime using the Leave One Out technique.
- Verification of InceptionTime's results using the 10-fold technique.

## Requirements Before Running This Directory

Make sure the following empty directories exist inside this directory:

- ten_fold_250
- ten_fold
- EX1
- EX2
- EX3
- EX4
- learners_EX1
- learners_EX2
- learners_EX3
- learners_EX4
- light_output
- lightning_saved_params </br>
You can use the script by running: `./create_folders.sh`

## Models Notebooks

- TST.ipynb
- InceptionTime.ipynb
- light_g.ipynb

Each notebook can be run with two options:
A) Random splitting of the samples into train and test sets - achieved by running the respective section under the title: "For random splits".
B) Splitting samples into train and test sets based on the subject's identification - achieved by running the respective section under the title: "For splitting subjects for Train and Test".

To change the current splitting option (A or B), you need to first run the `light_g.ipynb` notebook.

## InceptionTime10fold_250_epochs.ipynb and InceptionTime10fold.ipynb

These notebooks verify the results of InceptionTime to ensure it achieves better accuracy than the model in the paper. The verification is performed using the 10-fold technique, where the experiment is run 10 times with different splittings, and the average of the results is examined. The verification was conducted for both 100 and 250 epochs.

## incept_stress_leave_one.ipynb

This notebook provides a deep examination of the InceptionTime model for stress. It consists of two parts:

1) Training using the Leave One Out technique to check if the model achieves better results when splitting the dataset into train and test sets based on the subject's identification.
2) Hyperparameter changes to improve performance.


