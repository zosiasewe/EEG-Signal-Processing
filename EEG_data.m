
%% Description

% Data set covers a project of Tasting 4 different materials
% Subjects : 5 participants with Closed nose and 5 participant with opened nose
% Materials used are discibed in the file "Label" as 1,2,3,4

% The data from each subjects consists : 160x1280x17 meaning: trails x time x channels

% What we want to achieve?
% We want to perform a binary classification o both opened and closed eyes
% We want to use ML algorithms for a robust classification of this
% closed/open eyes 0/1 problem
% For that we want to divide the problem into 5 subgroups:
    % Test all of the data (160 trails)
    % Test only TM1 ("Test material 1")
    % Test only TM2 ("Test material 2")
    % Test only TM3 ("Test material 3")
    % Test only TM4 ("Test material 4")


% After this We want to do a research on BCI P300 and if the size of the
% screen matters 


%% Data loading 
clc
clear all
close all


closed_path = fullfile(pwd, 'closed_nose');
opened_path = fullfile(pwd, 'opened_nose'); %pwd - current folder
load_single_var = @(fpath) struct2array(load(fpath));


% Closed nose data
AC_closed = load_single_var(fullfile(closed_path, "AC.mat"));
BA_closed = load_single_var(fullfile(closed_path, "BA.mat"));
BO_closed = load_single_var(fullfile(closed_path, "BO.mat"));
CMY_closed = load_single_var(fullfile(closed_path, "CMY.mat"));
JD_closed = load_single_var(fullfile(closed_path, "JD.mat"));
Labels_closed = load_single_var(fullfile(closed_path, "Label_closed.mat"));

% Opened nose data
HIO_opened    = load_single_var(fullfile(opened_path, "HIO.mat"));
KK_opened     = load_single_var(fullfile(opened_path, "KK.mat"));
KUA_opened    = load_single_var(fullfile(opened_path, "KUA.mat"));
SKT_opened    = load_single_var(fullfile(opened_path, "SK_T.mat"));
SM_opened     = load_single_var(fullfile(opened_path, "SM.mat"));
Labels_opened = load_single_var(fullfile(opened_path, "Label_opened.mat"));

% plot(AC_closed(1,:,1));
% title('AC Closed - CH1');
% xlabel('Time');
% ylabel('Amplitude');

 
% AC eyes closed : 1 trail 
AC_c_1 = squeeze(AC_closed(1,:,:)); %first trail
frequency = 250; %sampling frequency
time = (0:size(AC_c_1,1)-1) / frequency;
offset = 200;

figure(1)
hold on;
for channels = 1:size(AC_c_1, 2)
    plot(time, AC_c_1(:,channels) + (channels-1)*offset, 'DisplayName', ['Ch ' num2str(channels)]);
end

title("AC - eyes closed I trail")
ylabel("channels")
xlabel("time (s)");
xlim([0 5.12])
ylim([-100 3370])
yticks((0:size(AC_c_1,2)-1)*offset);
yticklabels({"CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8","CH9","CH10","CH11","CH12","CH13","CH14","CH15","CH16","CH17"});
grid on;
hold off;






%% Denosing









