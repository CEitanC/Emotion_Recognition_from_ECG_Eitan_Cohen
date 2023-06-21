# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:29:28 2019

@author: Pritam
"""
from ast import walk
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import utils
from biosppy.signals import tools as tools
import glob
import shutil

#our
def csv_to_list(Path):
    with open(Path,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        my_list = []
        for row in lines:
            for e in row:
                my_list.append(float(e))
    return my_list

#our
def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    
    return filtered

#our
def clean_dir(path):
    files = glob.glob(path+'*')
    for f in files:
        os.remove(f)
#our
def clean_last_model():
    _,epoch = import_filenames('./models')
    print(epoch)
    
    for d in epoch:
        dir = './models/'+d
        shutil.rmtree(dir)    

def import_filenames(directory_path):
    """ 
    import all file names of a directory """

    print("dir path in import_filenames is ",directory_path)

    filename_list = []
    dir_list      = []
    for root, dirs, files in os.walk(directory_path, topdown=False):
        filename_list   = files     
        dir_list        = dirs
    # clean the ~lock
    for i in filename_list:
        if(("~" in i)or("read" in i)):
            filename_list.remove(i)
        



    return filename_list, dir_list
   
def normalize(x, x_mean, x_std):
    """ 
    perform z-score normalization of a signal """
    print("-----------normalize was used------------------")
    x_scaled = (x-x_mean)/x_std
    return x_scaled

def make_window(signal, fs, overlap, window_size_sec):
    """ 
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """
    print("-----------make_window was used------------------")

    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    segmented   = np.zeros((1, window_size), dtype = int)
    print("before, shape is: ",segmented[1:].shape)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        segment     = segment.reshape(1, len(segment))
        segmented   = np.append(segmented, segment, axis =0)
        start       = start + window_size - overlap
    print("finally, shape is: ",segmented[1:].shape)
    return segmented[1:]




        
        
def extract_swell_dataset(overlap_pct, window_size_sec, data_save_path, save):
    swell_path = "./csv_files/"
    swell_labels_path = "./label.xlsx"
    utils.makedirs(data_save_path)
    freq = 256
    window_size = window_size_sec * freq
    swell_file_names, _ = import_filenames(swell_path)
    
    person_name = []
    for i in swell_file_names:
        person_name.append(i[:i.find('_')])        
    person = np.unique(person_name)
    k = 0
    max_debug = -1
    swell_norm = np.empty((person.shape[0], 3))
    
    for i in tqdm(person):
        counter =0
        for j in swell_file_names:
            if j.split('_')[0]==i:
                signal = csv_to_list(swell_path + j)
                if counter == 0:
                    data = signal
                else:
                    data = np.vstack((data, signal))
                    counter = 1
        data = np.sort(data)
        tmp_max = np.max(data)
        if(tmp_max>max_debug):
            max_debug=tmp_max
        std = np.std(data[np.int(0.025*data.shape[0]) : np.int(0.975*data.shape[0])])
        mean = np.mean(data)
        print("for ", i," mean is ",mean,"std is ", std)
        swell_norm[k, :] = [np.int(i[2:]), mean, std]
        print("K is ",k)
        k = k+1 
        
    swell_dict = {}
    for i in tqdm(swell_file_names):
        name = np.int(i[2:i.find('_')])
        print("name is ",name)
        x_mean = swell_norm[np.where(swell_norm[:,0] == name)][:, 1][0]
        x_std  = swell_norm[np.where(swell_norm[:,0] == name)][:, 2][0]
        #our loading
        data_raw = []
        with open(swell_path + i,'r') as csvfile:
                    lines = csv.reader(csvfile, delimiter=',')
                    for row in lines:
                        for e in row:
                            data_raw.append(float(e))
        #data = np.loadtxt(swell_path + i) original version
        data = normalize(data_raw, x_mean, x_std) # this creates inf, the problem happen when std==0
        tmp_max = np.max(data)
        if(tmp_max>max_debug):
            max_debug=tmp_max
        data_windowed = make_window (data, freq, overlap_pct, window_size_sec)
        swell_dict.update({i: data_windowed})
        
    counter = 0;
    label = pd.ExcelFile(swell_labels_path)
    label_sheet_names = label.sheet_names
    print("label_sheet_names:")
    print(label_sheet_names)
    participant_labellings = pd.DataFrame
    print("Dataframe: ", participant_labellings)
    
    print('getting labels...')
    for i in tqdm(range(len(label_sheet_names))):
        participant_labellings = label.parse(label_sheet_names[i])
        if counter == 0:
            labels = participant_labellings
        else:
            labels = labels.append(participant_labellings, ignore_index = True, sort=False)
        counter = counter + 1;
    swell_labels = labels.drop_duplicates(subset = ['PP','Blok'], keep = 'last')
    swell_labels = swell_labels.reset_index(drop = True)
    counter = 0

    #adding csv names into labels
    swell_labels['filename'] = 'default'
    for i in swell_file_names:
        start = i.find('_')
        end = i.find('c')
        print("swell_labels['PP']:")
        print(swell_labels['PP'])
        print("i[:start].upper():")
        print(i[:start].upper())
        print("swell_labels['Blok']:")
        print(swell_labels['Blok'])
        print("int(i[end+1:-4]):")
        print(int(i[end+1:-4]))
        condition = (swell_labels['PP'] == i[:start].upper()) &(swell_labels['Blok'] == int(i[end+1:-4]))
        print("conditon is: ")
        print(condition)
        print("the type of con is ",type(condition))
        index = np.where(condition)[0]
        print("index is ",index)
        print("np.where(condition): ",np.where(condition))
        print("swell_labels['filename'].iloc[index[0]]")
        #print(swell_labels['filename'].iloc[index[0]])
        if len(index) != 0:
            swell_labels['filename'].iloc[index[0]] = i 
      #  print("swell_labels['filename'].iloc[index[0]]",swell_labels['filename'].iloc[index[0]]) 
    print('dict unpacking...')
    final_set = np.zeros((1, window_size+12), dtype = int)
    key_list = swell_dict.keys()
    
    for i in tqdm(key_list):
        new_key = np.float(i[i.find('pp')+2:i.find('_')] + "." + i[i.find('c')+1:-4])
        values = swell_dict[i]
        key = np.repeat(new_key, len(values))
        key = key.reshape(len(key), 1)
        label_set = swell_labels[(swell_labels['PP'] == i[:i.find('_')].upper()) & (swell_labels['Blok'] == np.int(i[i.find('c')+1:-4]))]
        label_set = label_set[['Valence_rc', 'Arousal_rc', 'Dominance', 'Stress', 'MentalEffort', 'MentalDemand', 'PhysicalDemand', 'TemporalDemand', 'Effort','Performance_rc', 'Frustration']]
        label_set = pd.concat([label_set]*len(values), ignore_index=True)
        label_set = np.asarray(label_set)
        print("---key ",key.shape, "\n label ",label_set.shape,"\n vals ",values.shape)
        label_set=np.resize(label_set, ( key.shape[0],label_set.shape[1]))
        signal_set = np.hstack((key, label_set, values))
        final_set = np.vstack((final_set, signal_set))    
    final_set = final_set[1:]
    max_val = np.max(final_set) #for debug
    if save:
        np.save(os.path.join(data_save_path,'swell_dict.npy'), final_set)          
   
    print(swell_dict)
    final_set = final_set.shape
    print("final_set size is:", final_set)
    print('swell files importing finished...')
    return final_set

       
def load_data(path):
    dataset = np.load(path, allow_pickle=True)     
    return dataset   


def swell_prepare_for_10fold(swell_data):
    
    ecg = swell_data[:, 12:]
    
    """ 'person.blok', 'Valence_rc', 'Arousal_rc', 'Dominance' """
    """ 'person.blok', 'Valence_rc', 'Arousal_rc', 'Dominance', 'Stress', 'MentalEffort', 'MentalDemand', 'PhysicalDemand', 'TemporalDemand', 'Effort','Performance_rc', 'Frustration' """

    person               = np.floor(swell_data[:,0])
    y_input_stress       = (swell_data[:, 0]*10 - np.round(swell_data[:, 0])*10).astype(int)
    y_arousal            = swell_data[:, 2]
    y_valence            = swell_data[:, 1]
    person               = person.reshape(-1, 1)
    y_input_stress       = y_input_stress.reshape(-1, 1)
    y_arousal            = y_arousal.astype(int).reshape(-1, 1)
    y_valence            = y_valence.astype(int).reshape(-1, 1)
    swell_data  = np.hstack((person, y_input_stress, y_arousal, y_valence, ecg))
    return swell_data 

def save_list(mylist, filename):
    for i in range(len(mylist)):
        temp = mylist[i]
        with open(filename, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(temp)
    return               


