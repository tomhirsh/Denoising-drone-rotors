% This code seperates audio (given by matrix as variable) into
% segments of length L and saves those segments in a sub-folder.
% The length of the output array is claculated by: L*fs (sample per second
% times length of segment (in sec)).

% recordings folder
rec_f_name = 'C:\Users\Tom\Documents\Learning_stuff\semester 8\Denoising_project\data\';
% recording file name to use
rec_name = 'Processed_data_sounds.mat';
% choose Mic
sub_f_name = 'Mic8_sounds';

% create subfolder to save processed data
mkdir(sub_f_name);

% load selected recording
selected_recording = sprintf('%s\%s', rec_f_name, rec_name);
load(selected_recording);

% calculate frames per second (fs)
fs = ceil(1/(T(2)-T(1)));

% length of segments (seconds)
L = 1;

% create and save segments
end_loop = floor(T(size(T))/L);
for i = 1:end_loop
    idx = (T >= L*(i-1)) & (T <= L*i);
    tmp = Mic8(idx);
    %if i == end_loop(1) % use only for rotors (first run getRpm)
    %    mean_rpm = rpm(i*fs);
    %else
    %    mean_rpm = mean(rpm(i*fs:(i+1)*fs));
    %end
    %name = sprintf('%s/%s_%.0f_%.0f.wav', sub_f_name, sub_f_name, i, mean_rpm); % use only for rotors (first run getRpm)
    name = sprintf('%s/%s_%.0f.wav', sub_f_name, sub_f_name, i); % use for sounds
    audiowrite(name,tmp,fs);
end