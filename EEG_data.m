
%% Description
%{

Data set covers a project of Tasting 4 different materials
Subjects : 5 participants with Closed nose and 5 participant with opened nose
Materials used are discibed in the file "Label" as 1,2,3,4

The data from each subjects consists : 160x1280x17 meaning: trails x time x channels

The data was recorded using 512 Hz sampling frequency
Each epoch consists the recorded event of "tasting"

What we want to achieve?
- We want to perform a binary classification on both opened and closed nose
- We want to use ML algorithms for a robust classification of this closed/open nose 0/1 problem

For that we want to divide the problem into 5 subgroups:
    - Test all of the data (160 trails)
    - Test only TM1 ("Test material 1")
    - Test only TM2 ("Test material 2")
    - Test only TM3 ("Test material 3")
    - Test only TM4 ("Test material 4")


After this We want to do a research on BCI P300 and if the size of the
% screen matters 
%}

%% ---------------------------------------------------------------------------------------------------------------------------
clc
clear
close all

closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
tmp = load(closed_path);
data = struct2array(tmp);                                           % Size: [160 × 1280 × 17] = [trials × time × channels]
channel_names = load("Channel_names.mat");
channels = struct2array(channel_names);

% Defininition of variables
sampling_rate = 512;
n_samples = 1280;
n_trials = size(data, 1);                                           % 160 trials
n_time = size(data, 2);                                             % 1280 time points per trial
n_channels = size(data, 3);                                         % 17 channels
trial_duration_sec = n_time / sampling_rate;                        % 1280 / sampling rate
total_duration_sec = n_trials * trial_duration_sec;

data_perm = permute(data, [3, 2, 1]);                               % [channels × time × trials]
EEG_data_shaped = reshape(data_perm, n_channels, []);               % [17 × (1280*160)]

time = linspace(0, total_duration_sec, size(EEG_data_shaped, 2));   % time for entire signa
offset = 300;                                                       % separation between channels

%% Plot for the whole time range

figure;
hold on;

for ch = 1:n_channels
    plot(time, EEG_data_shaped(ch, :) + (ch - 1) * offset);
end

title('Continuous EEG plot');
subtitle('All 160 trials concatenated (AC, closed nose)')
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
xlim([0, total_duration_sec]);
grid on;
hold off;

%% Plot PSD for the whole set
figure;

%SNR
snr_raw_data = zeros(1, n_channels);
signal_band = [0.5 30];
noise_band = [45 100];

all_psds = [];
fprintf('SNR for Raw Data: \n')
for ch = 1:n_channels
    [psd_single, f] = pwelch(EEG_data_shaped(ch, :), [], [], [], sampling_rate);
    all_psds = [all_psds, psd_single];

    signal_indices = f >= signal_band(1) & f <= signal_band(2);
    P_signal = sum(psd_single(signal_indices));
    
    noise_indices = f >= noise_band(1) & f <= noise_band(2);
    P_noise = sum(psd_single(noise_indices));
    
    snr_raw_data(ch) = 10*log10(P_signal/P_noise);
    fprintf('SNR = %.2f dB for %s\n', snr_raw_data(ch), channels{ch});
end

avg_psd = mean(all_psds, 2);
avg_psd_db = 10*log10(avg_psd);

plot(f, avg_psd_db, 'b-', 'LineWidth', 0.1,'Color',"#7E2F8E");
title('Average PSD Across All Channels and Trials', 'FontSize', 10);
subtitle('(AC, closed nose)')
xlabel('Frequency (Hz)');
ylabel('PSD (dB)');
xlim([0 256]);
ylim([-80 40]);
grid on;

%% Plot for defined Epochs 
epoch_num = [40, 50, 60, 70];
time_2 = (0:n_samples-1) / sampling_rate;

figure;

for num_epochs = 1:length(epoch_num)
    subplot(2, 2, num_epochs);
    hold on;

    for ch = 1:n_channels
        plot(time_2, data(epoch_num(num_epochs), :, ch) + (ch - 1) * offset);
    end

    title(['Raw EEG — Trial ', num2str(epoch_num(num_epochs))]);
    subtitle('Before filtering (AC, closed nose)')
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels - 1) * offset);
    yticklabels(channels(1:n_channels));
    ylim([-offset, offset * n_channels]);
    xlim([0, 2.5]);
    grid on;
    hold off;
end



%% Filtering Data - Analysis
% low_numbers = [0.5, 0.4, 0.3];
% high_numbers = [30, 35, 40];
% 
% combination_idx = 1;
% n_filter_combinations = length(low_numbers) * length(high_numbers);
% filter_labels = cell(n_filter_combinations, 1);
% avg_psd_results = cell(n_filter_combinations, 1);
% colors = lines(n_filter_combinations);
% 
% %Stats
% results = [];
% filter_names = {};
% counter = 1;
% 
% 
% fprintf('\nFilter\t\t\tDelta\t\t\tAlpha\t\t\tBeta\t\t\tGamma\t\t\tNoise\n')
% 
% for z = 1:length(low_numbers)
%     for y = 1:length(high_numbers)
%         
%         nyquist = sampling_rate / 2;
%         low_cutoff = low_numbers(z);
%         high_cutoff = high_numbers(y);
%         filter_order = 2;
%         
%         filter_labels{combination_idx} = sprintf('HP=%.1f LP=%d', low_cutoff, high_cutoff);
%         
%         [b_hp, a_hp] = butter(filter_order, low_cutoff / nyquist, 'high');
%         [b_lp, a_lp] = butter(filter_order, high_cutoff / nyquist, 'low');
%         
%         EEG_filtered = zeros(size(EEG_data_shaped));
%         
%         for ch = 1:n_channels
%             hp_output = filtfilt(b_hp, a_hp, EEG_data_shaped(ch, :));
%             EEG_filtered(ch, :) = filtfilt(b_lp, a_lp, hp_output);
%         end
%         
%         EEG_filtered_reshaped = reshape(EEG_filtered, n_channels, n_samples, n_trials);
%         
%         all_psd = [];
%         
%         for chan = 1:n_channels
%             for trial = 1:n_trials
%                 
%                 trial_data = squeeze(EEG_filtered_reshaped(chan, :, trial));  % Single trail
%                 
%                 [psd_single, f] = pwelch(trial_data, [], [], [], sampling_rate);
%                 
%                 if isempty(all_psd)
%                     all_psd = psd_single;
%                     freq_vector = f;
%                 else
%                     all_psd = [all_psd, psd_single];
%                 end
%             end
%         end
%         
% 
%         delta_power = sum(all_psd(f >= 0.5 & f <= 4));
%         alpha_power = sum(all_psd(f >= 8 & f <= 13));
%         beta_power = sum(all_psd(f >= 13 & f <= 30));
%         gamma_power = sum(all_psd(f >= 30 & f <= 45));
%         noise_level = sum(all_psd(f >= 45 & f <= 50));
% 
%         avg_psd = mean(all_psd, 2);
%         avg_psd_results{combination_idx} = avg_psd;
%         
%         combination_idx = combination_idx + 1;
% 
%         results(counter, :) = [delta_power, alpha_power, beta_power, gamma_power, noise_level];
%         filter_names{counter} = sprintf('HP=%.1f LP=%d', low_cutoff, high_cutoff);
%         fprintf('%s\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\n', ...
%             filter_names{counter}, delta_power, alpha_power, beta_power, gamma_power, noise_level);
%         
%         counter = counter + 1;
% 
%     end
% end


%% Result HPF, LPF
low_numbers = 0.5;
high_numbers = 30;
nyquist = sampling_rate / 2;
[b_hp_r, a_hp_r] = butter(2, low_numbers / nyquist, 'high');
[b_lp_r, a_lp_r] = butter(2, high_numbers / nyquist, 'low');
EEG_filter = zeros(size(EEG_data_shaped));
    for ch = 1:n_channels
        hp_output = filtfilt(b_hp_r, a_hp_r, EEG_data_shaped(ch, :));
        EEG_filter(ch, :) = filtfilt(b_lp_r, a_lp_r, hp_output);
    end
        
EEG_filtered_result = reshape(EEG_filter, n_channels, n_samples, n_trials);


%% All time after filtering 
figure;
hold on;

for ch = 1:n_channels
    plot(time, EEG_filtered_result(ch, :) + (ch - 1) * offset);
end

title('Continuous EEG plot');
subtitle('After HPF, LPF (AC, closed nose)')
xlabel('Time (s)');
ylabel('Channels');
yticks((0:n_channels-1) * offset);
yticklabels(channels(1:n_channels));
ylim([-offset, offset * n_channels]);
xlim([0, total_duration_sec]);
grid on;
hold off;

% --------
figure;
fprintf('\n')
fprintf('SNR for Filtered Data: \n')
%SNR
snr_raw_data_f = zeros(1, n_channels);
signal_band = [0.5 30];
noise_band = [45 100];

all_psds = [];
for ch = 1:n_channels
    [psd_single, f] = pwelch(EEG_filtered_result(ch, :), [], [], [], sampling_rate);
    all_psds = [all_psds, psd_single];

    signal_indices_f = f >= signal_band(1) & f <= signal_band(2);
    P_signal_f = sum(psd_single(signal_indices_f));
    
    noise_indices_f = f >= noise_band(1) & f <= noise_band(2);
    P_noise_f = sum(psd_single(noise_indices_f));
    
    snr_raw_data_f(ch) = 10*log10(P_signal_f/P_noise_f);
    fprintf('SNR = %.2f dB for %s\n', snr_raw_data_f(ch), channels{ch});

end

avg_psd = mean(all_psds, 2);
avg_psd_db = 10*log10(avg_psd);

plot(f, avg_psd_db, 'b-', 'LineWidth', 0.1, 'Color',"#7E2F8E");
title('Average PSD Across All Channels and Trials', 'FontSize', 10);
subtitle("After HPF and LPF Filtering (AC, closed nose)")
xlabel('Frequency (Hz)');
ylabel('PSD (dB)');
xlim([0 256]);
ylim([-80 40]);
grid on;



%% Plot for defined Epochs - after filtering
epoch_num = [40, 50, 60, 70];
time_2 = (0:n_samples-1) / sampling_rate;

figure;

for num_epochs = 1:length(epoch_num)
    subplot(2, 2, num_epochs);
    hold on;

    for ch = 1:n_channels
        plot(time_2, EEG_filtered_result(ch, :,epoch_num(num_epochs)) + (ch - 1) * offset);
    end

    title(['Raw EEG — Trial ', num2str(epoch_num(num_epochs))]);
    subtitle('After HPF, LPF (AC - closed nose)')
    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels - 1) * offset);
    yticklabels(channels(1:n_channels));
    ylim([-offset, offset * n_channels]);
    xlim([0, 2.5]);
    grid on;
    hold off;
end

%% Artifact Removal - ICA

A = [];
x = EEG_filtered_result(:,:,:);
n_independent_components = 4;

x_reshaped = reshape(EEG_filtered_result, n_channels,[]);

n_independent_components = 4;  % Number of components to extract
[ica_components, W_ica, A_ica] = ICA(x_reshaped, n_independent_components, 1000, 1e-6, 1.0);

ica_reshaped = reshape(ica_components, n_independent_components, n_samples, n_trials);


%% Power Spectral Density ICA
figure();
for c = 1:n_independent_components
    subplot(2, 2, c);
    [psd_comp, f_comp] = pwelch(ica_components(c, :), [], [], [], sampling_rate);
    psd_db = 10*log10(psd_comp);
    
    plot(f_comp, psd_db, 'LineWidth', 1.5);
    title(['ICA Component ', num2str(c), ' - Power Spectral Density']);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    xlim([0, 100]);
    grid on;
end


%% Delete the artifacts
artifact_components = [];

A_clean = A_ica;
A_clean(:, artifact_components) = 0;

clean_components = ica_components;
clean_components(artifact_components, :) = 0;

clean_signal = A_clean * clean_components;
clean_eeg = reshape(clean_signal, n_channels, n_samples, n_trials);



%% Raw Feature Extraction (Time + Frequency + Nonlinear)

n_trials = size(clean_eeg, 3);
n_channels = size(clean_eeg, 1);

% Frequency bands
bands = [0.5 4; 4 8; 8 13; 13 30; 30 100];  % Delta, Theta, Alpha, Beta, Gamma
n_bands = size(bands, 1);

feature_matrix = [];
for trial = 1:n_trials
    trial_features = [];
    for ch = 1:n_channels
        x = squeeze(clean_eeg(ch, :, trial)); % single trial, single channel
        
        % Time Domain Features
        mu = mean(x);
        sigma2 = var(x);
        sigma = sqrt(sigma2);
        skew_val = skewness(x);
        kurt_val = kurtosis(x);
        RMS_val = sqrt(mean(x.^2));
        ZCR_val = sum(abs(diff(sign(x)))) / (2*(length(x)-1));
        MAD_val = mean(abs(x - mu));
        PtP_val = max(x) - min(x);
        v = diff(x);        % first difference
        a = diff(v);        % second difference
        vRMS = sqrt(mean(v.^2));
        aRMS = sqrt(mean(a.^2));

        time_features = [mu, sigma2, sigma, skew_val, kurt_val, RMS_val, ZCR_val, MAD_val, PtP_val, vRMS, aRMS];

        % Frequency Domain Features
        [pxx, f] = pwelch(x, [], [], [], sampling_rate);
        band_powers = zeros(1, n_bands);
        for b = 1:n_bands
            idx = f >= bands(b,1) & f <= bands(b,2);
            band_powers(b) = sum(pxx(idx));
        end
        [~, peak_idx] = max(pxx);
        f_peak = f(peak_idx);
        f_mean = sum(f.*pxx)/sum(pxx);
        SC = sum(f.*pxx)/sum(pxx);  % spectral centroid
        SS = sqrt(sum(((f-SC).^2).*pxx)/sum(pxx)); % spectral spread
        P_norm = pxx/sum(pxx);
        H = -sum(P_norm.*log(P_norm + eps)); % spectral entropy
        
        freq_features = [band_powers, f_peak, f_mean, SC, SS, H];

        % Nonlinear Features
        % Hjorth Parameters
        Activity = var(x);
        Mobility = sqrt(var(diff(x))/Activity);
        Complexity = sqrt(var(diff(diff(x)))/var(diff(x)))/Mobility;

        nonlinear_features = [Activity, Mobility, Complexity];

        %Combine features
        trial_features = [trial_features, time_features, freq_features, nonlinear_features];
    end
    feature_matrix = [feature_matrix; trial_features];
end

% Normalize Features
feature_matrix = (feature_matrix - mean(feature_matrix)) ./ std(feature_matrix);
fprintf('Feature matrix size: [%d x %d]\n', size(feature_matrix,1), size(feature_matrix,2));


%% GA

[n_trials, n_features] = size(feature_matrix);
epsilon = 1e-6;

n_pop = 30;
n_gen = 20;
mutation_rate = 0.1;

% Initialize population
pop = zeros(n_pop, 2*n_features);
pop(:, 1:n_features) = 0.5 + 1.5*rand(n_pop, n_features);  % scaling 0.5-2
pop(:, n_features+1:end) = randi([0,3], n_pop, n_features); % transformation 0-3

best_fitness = -Inf;
best_chrom = pop(1,:);

for gen = 1:n_gen
    fitness = zeros(n_pop,1);
    
    for i = 1:n_pop
        chrom = pop(i,:);
        
        % Apply chromosome
        X_trans = zeros(size(feature_matrix));
        scales = chrom(1:n_features);
        transforms = round(chrom(n_features+1:end));
        
        for f = 1:n_features
            x = feature_matrix(:,f) * scales(f);  % scale
            switch transforms(f)
                case 0
                    X_trans(:,f) = x;
                case 1
                    X_trans(:,f) = log(x + epsilon);
                case 2
                    X_trans(:,f) = sqrt(max(x + epsilon,0));
                case 3
                    X_trans(:,f) = x.^2;
            end
        end
        
        % Fuzzy membership: low, medium, high
        fuzzy_mat = zeros(n_trials, n_features, 3); % 3 membership levels
        for f = 1:n_features
            f_min = min(X_trans(:,f));
            f_max = max(X_trans(:,f));
            f_range = f_max - f_min + epsilon;
            
            % Triangular membership functions
            fuzzy_mat(:,f,1) = max(0, min(1, (0.5*(f_min+f_max) - X_trans(:,f)) / (0.5*f_range))); % low
            fuzzy_mat(:,f,2) = max(0, 1 - abs(X_trans(:,f) - 0.5*(f_min+f_max)) / (0.5*f_range));  % medium
            fuzzy_mat(:,f,3) = max(0, min(1, (X_trans(:,f) - 0.5*(f_min+f_max)) / (0.5*f_range))); % high
        end
        
        % Fuzzy entropy per feature
        entropy_sum = 0;
        for f = 1:n_features
            p = mean(squeeze(fuzzy_mat(:,f,:)),1);  % average membership across trials
            p = p / sum(p + epsilon);               % normalize
            entropy_sum = entropy_sum - sum(p.*log2(p + epsilon));
        end
        
        fitness(i) = entropy_sum;  % maximize entropy
    end
    
    [max_fit, idx] = max(fitness);
    if max_fit > best_fitness
        best_fitness = max_fit;
        best_chrom = pop(idx,:);
    end
    
    fprintf('Generation %d, Best Score = %.4f\n', gen, best_fitness);
    
    % Selection (tournament)
    new_pop = zeros(size(pop));
    for i = 1:n_pop
        idx1 = randi(n_pop);
        idx2 = randi(n_pop);
        if fitness(idx1) > fitness(idx2)
            new_pop(i,:) = pop(idx1,:);
        else
            new_pop(i,:) = pop(idx2,:);
        end
    end
    
    % Crossover (single point)
    for i = 1:2:n_pop-1
        if rand < 0.8
            cp = randi(2*n_features-1);
            temp1 = new_pop(i,cp+1:end);
            temp2 = new_pop(i+1,cp+1:end);
            new_pop(i,cp+1:end) = temp2;
            new_pop(i+1,cp+1:end) = temp1;
        end
    end
    
    % Mutation
    for i = 1:n_pop
        for j = 1:2*n_features
            if rand < mutation_rate
                if j <= n_features
                    new_pop(i,j) = 0.5 + 1.5*rand;
                else
                    new_pop(i,j) = randi([0,3]);
                end
            end
        end
    end
    
    pop = new_pop;
end

% transformed feature matrix
scales = best_chrom(1:n_features);
transforms = round(best_chrom(n_features+1:end));

X_transformed = zeros(size(feature_matrix));
for f = 1:n_features
    x = feature_matrix(:,f) * scales(f);
    switch transforms(f)
        case 0
            X_transformed(:,f) = x;
        case 1
            X_transformed(:,f) = log(x + epsilon);
        case 2
            X_transformed(:,f) = sqrt(max(x + epsilon,0));
        case 3
            X_transformed(:,f) = x.^2;
    end
end

fprintf('GA + Fuzzy Feature Extraction completed. Transformed matrix size: [%d x %d]\n', ...
    size(X_transformed,1), size(X_transformed,2));
