%% 
clear all;
clc;
%%

Files=dir('*.S00');
for k=1:length(Files)
    file_S00 = Files(k).name;
    [ecgs, ~, ~] = myDoReadData(file_S00);
    len = strlength(file_S00);
    file_csv = replaceBetween(file_S00,len-3,len,".csv");
    ecg = ecgs.data{6,1};
    sampled = downsample(ecg, 8);
    writematrix(sampled,file_csv) ;
end


