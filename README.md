# Final Project B - Electrical and Computer Engineering Technion
## Emotion Recognition from ECG
### Eitan Cohen
### Supervised by Amro Abu-Saleh

## Project Goals:
    Research the option of emotion recognition from ECG
    Follow after an existing model and offer a way to improve performances.

## Repository Content:
Extract_samples: MATLAB script for converting the ECG samples from S00 to csv files.
pre_processing_and_recover_network:  reconstruction of the  SSL model from the paper including the pre-processing of the dataset. In Keras.
pytorch_model: implementation of the FS model using Pytorch and analysis of the result  for the research part of the project.
TS_models: the TST and Inception Time models including the experiment of trying to generalization.

The instructions for running the project can be found in the main directory of the project, in the file named "Instructions_for_Running.md."

## Credits
The projed is based on the paper - https://arxiv.org/abs/2002.03898
@misc{sarkar2020selfsupervised,
    title={Self-supervised ECG Representation Learning for Emotion Recognition},
    author={Pritam Sarkar and Ali Etemad},
    year={2020},
    eprint={2002.03898},
    archivePrefix={arXiv},
    primaryClass={eess.SP}
}

@misc{sarkar2020selfsupervised,
    title={Self-supervised ECG Representation Learning for Emotion Recognition},
    author={Pritam Sarkar and Ali Etemad},
    year={2020},
    eprint={2002.03898},
    archivePrefix={arXiv},
    primaryClass={eess.SP}
}

@INPROCEEDINGS{sarkar2019selfsupervised,
  author={P. {Sarkar} and A. {Etemad}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Self-Supervised Learning for ECG-Based Emotion Recognition}, 
  year={2020},
  volume={},
  number={},
  pages={3217-3221},}
  
  https://github.com/pritamqu/SSL-ECG/blob/master/LICENSE

## Sources:
•	Pritam Sarkar, Ali Etemad , “Self-supervised ECG Representation Learning for Emotion Recognition”, IEEE Transactions on Affective Computing, 2020. </br>
•	Hasnul, Muhammad Anas, Nor Azlina Ab. Aziz, Salem Alelyani, Mohamed Mohana, and Azlan Abd. Aziz. "Electrocardiogram-Based Emotion Recognition Systems and Their Applications in Healthcare—A Review" Sensors 21, no. 15: 5015. https://doi.org/10.3390/s21155015, 2021. </br>
•	Saskia Koldijk, Maya Sappelli, Suzan Verberne, Mark A. Neerincx, and Wessel Kraaij. 2014. The SWELL Knowledge Work Dataset for Stress and User Modeling Research. In Proceedings of the 16th International Conference on Multimodal Interaction (ICMI '14). Association for Computing Machinery, New York, NY, USA, 291–298. </br>
•	Ismail Fawaz, Hassan, Benjamin Lucas, Germain Forestier, Charlotte Pelletier, Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, and François Petitjean. "Inceptiontime: Finding alexnet for time series classification." Data Mining and Knowledge Discovery 34, no. 6 (2020): 1936-1962. </br>
•	Zerveas, George, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty, and Carsten Eickhoff. "A transformer-based framework for multivariate time series representation learning." In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 2114-2124. 2021.
