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
    if k==1
        for i=1:8
            channel = ecgs.data{i,1};
            sampled = downsample(channel, 8);
            name = "channel_"+i+".csv";
            writematrix(sampled,name) ;
        end
    end
end

%% 
plot(sampled)
xlim([0,10e2])
%%


[a, portiHeartPre, b] = myDoReadData('pp9_4-10-2012_c1.S00'); 
%[a2, portiHeartPre, b] = myDoReadData('pp9_4-10-2012_c2.S00'); 
%[a3, portiHeartPre, b] = myDoReadData('pp9_4-10-2012_c3.S00'); 
%%
orgin = a.data{6,1};
sam8 = downsample(orgin, 8);
writematrix(sam8,'pp9.csv');


%%
plot(sam8)
xlim([0 1e3])

%%
c1 = a1.data{6,1};
c2 = a2.data{6,1};
c3 = a2.data{6,1};
%%
subplot(1,3,1);
plot(c1);
xlim([0 1e3])

subplot(1,3,2);
plot(c2);
xlim([0 1e3])

subplot(1,3,3);
plot(c3);

xlim([0 1e3])

%%
ecg1=a.data{1,1}; %not ECG, scaling: xlim[0 10e4]
ecg2=a.data{2,1}; %not ECG, scaling: xlim[0 10e4]
ecg3=a.data{3,1}; %not ECG, scaling: xlim[0 10e5]
ecg4=a.data{4,1}; %not ECG, squares. scaling: xlim[0 10e4]
ecg5=a.data{5,1}; %not ECG, scaling: xlim[0 10e3]
ecg6=a.data{6,1}; %probably the ECG, scaling: xlim[0 10e3]
ecg7=a.data{7,1}; %not ECG, scaling: xlim[0 10e5]
ecg8 = a.data{8,1};
%%
writematrix(ecg1,'ecg1.csv');
writematrix(ecg2,'ecg2.csv');
writematrix(ecg3,'ecg3.csv');
writematrix(ecg4,'ecg4.csv');
writematrix(ecg5,'ecg5.csv');
writematrix(ecg6,'ecg6.csv');
writematrix(ecg7,'ecg7.csv');
%%

sampled = downsample(ecg6, 256);

writematrix(sampled,'pp8_4-10-2012_c3.csv') ;

%writematrix(portiHeartPre,'not_sampled.csv') ;

%%

plot(ecg6)

xlim([0 10e3]);
%%
plot(sampled)
xlim([0 10e3]);

%%

figure(1)
subplot(3,3,1)
plot(ecg1)
xlim([0 10e4]);
title('ecg1');

subplot(3,3,2);
plot(ecg2);
xlim([0 10e4]);
title('ecg2');

subplot(3,3,3)
plot(ecg3)
xlim([0 10e5]);
title('ecg3');

subplot(3,3,4)
plot(ecg4)
xlim([0 10e4]);
title('ecg4');

subplot(3,3,5)
plot(ecg5)
xlim([0 10e3]);
title('ecg5');

subplot(3,3,6)
plot(ecg6)
xlim([0 10e3]);
title('ecg6');

subplot(3,3,7)
plot(ecg7)
xlim([0 10e5]);
title('ecg7');

subplot(3,3,8)
plot(ecg8)
xlim([0 10e3]);
title('ecg8');