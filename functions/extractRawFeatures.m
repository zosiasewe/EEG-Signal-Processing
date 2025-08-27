function features = extractRawFeatures(EEG_data, EEG_filt_data, n_trials, sampling_rate, bands)

    % Number of frequency bands
    n_bands = size(bands,1);

    [n_channels, ~, ~] = size(EEG_data);
    n_time_features_per_channel = 11; % mu, var, std, skew, kurt, RMS, MAD, PtP, ZCR, vRMS, aRMS
    n_freq_features_per_channel = n_bands + 5; % band powers + f_peak + f_mean + SC + SS + H
    n_connectivity_pairs = nchoosek(n_channels,2);

    % Feature vector size
    n_features = (n_time_features_per_channel + n_freq_features_per_channel) * n_channels ...
                 + n_connectivity_pairs * 2;
    features = zeros(n_trials, n_features);

    % Loop over trials
    for trial = 1:n_trials
        % Use filtered data if available
        if ~isempty(EEG_data)
            trial_data = squeeze(EEG_data(:,:,trial));
        else
            trial_data = squeeze(EEG_filt_data(:,:,trial));
        end

        trial_features = [];

        % Time domain
        for ch = 1:n_channels
            x = trial_data(ch,:);
            mu = mean(x);
            sigma2 = var(x);
            sigma = sqrt(sigma2);
            skew_val = skewness(x);
            kurt_val = kurtosis(x);
            RMS_val = sqrt(mean(x.^2));
            MAD_val = mean(abs(x - mu));
            PtP_val = max(x) - min(x);
            ZCR_val = sum(abs(diff(sign(x)))) / (2*(length(x)-1));
            v = diff(x);
            a = diff(v);
            vRMS = sqrt(mean(v.^2));
            aRMS = sqrt(mean(a.^2));
            trial_features = [trial_features, mu, sigma2, sigma, skew_val, kurt_val, RMS_val, MAD_val, PtP_val, ZCR_val, vRMS, aRMS];
        end

        % Frequency domain
        for ch = 1:n_channels
            x = trial_data(ch,:);
            [pxx,f] = pwelch(x, [], [], [], sampling_rate);

            % Band powers
            band_powers = zeros(1, n_bands);
            for b = 1:n_bands
                idx = f >= bands(b,1) & f <= bands(b,2);
                band_powers(b) = sum(pxx(idx));
            end

            % Spectral features
            [~, peak_idx] = max(pxx);
            f_peak = f(peak_idx);
            f_mean = sum(f.*pxx)/sum(pxx);
            SC = f_mean; % spectral centroid
            SS = sqrt(sum(((f-SC).^2).*pxx)/sum(pxx));
            P_norm = pxx/sum(pxx);
            H = -sum(P_norm.*log(P_norm + eps));

            trial_features = [trial_features, band_powers, f_peak, f_mean, SC, SS, H];
        end

        % Connectivity
        for i = 1:n_channels
            for j = i+1:n_channels
                corr_val = corr(trial_data(i,:)', trial_data(j,:)');
                coh_val = abs(corr_val); % aproxy coherence
                trial_features = [trial_features, corr_val, coh_val];
            end
        end

        features(trial,:) = trial_features;
    end
end
