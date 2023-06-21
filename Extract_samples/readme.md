# Steps for Creating CSV Files for SWELL Datasets

## 1. Downloading S00 Files
 The raw data directory is [here](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-x55-69zp), </br>
 Navigate to Files-> Tree-> 0-Raw data -> D - Physiology -raw data -> Mobi signals_raw and filtered </br>
 Download all the S00 files except for the following files:
- pp10_8-10-2012-c2.S00
- pp15_15-10-2012_c2.S00
- pp20_29-10-2012_c1.S00
- pp22_1-11-2012_c3.S00
- pp25_7-11-2012_c2.S00
- pp4_25-9-2012_c1.S00
- pp8_4-10-2012_c2.S00
- pp11_9-10-2012_c1.S00
- pp15_15-10-2012_c3.S00
- pp20_29-10-2012_c2.S00
- pp23_1-11-2012_c2_cont.S00
- pp25_7-11-2012_c3.S00
- pp4_25-9-2012_c2.S00
- pp8_4-10-2012_c3.S00
- pp11_9-10-2012_c3.S00
- pp16_15-10-2012_c2.S00
- pp22_1-11-2012_c2.S00
- pp24_5-11-2012_c3.S00
- pp3_24-9-2012_c1.S00
- pp8_4-10-2012_c1.S00

## 2. Locating the S00 Files
Locate the downloaded S00 files in the directory where the MATLAB scripts are stored.

## 3. Running the MATLAB Script
Run the `main.m` script to create all the necessary CSV files.

