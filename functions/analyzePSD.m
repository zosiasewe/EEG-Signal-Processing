function analyzePSD(EEG_data, sampling_rate, n_channels, channels, subject_name, class, data_type)

    [psd_values_tmp, freq_axis] = pwelch(EEG_data(1,:), [], [], [], sampling_rate);
    n_freqs = length(freq_axis);
    all_psd_data = zeros(n_freqs, n_channels);
    snr_values = zeros(1, n_channels);
    
    signal_band = [0.5 30];
    noise_band = [45 100];

    % Compute PSD and SNR for each channel
    for ch = 1:n_channels
        [psd_values, ~] = pwelch(EEG_data(ch, :), [], [], [], sampling_rate);
        all_psd_data(:,ch) = psd_values;

        signal_power = sum(psd_values(freq_axis >= signal_band(1) & freq_axis <= signal_band(2)));
        noise_power  = sum(psd_values(freq_axis >= noise_band(1) & freq_axis <= noise_band(2)));
        snr_values(ch) = 10*log10(signal_power / noise_power);
    end

    % Plot average PSD
    avg_psd_db = 10*log10(mean(all_psd_data, 2));
% % %     figure; hold on;
% % %     plot(freq_axis, avg_psd_db, 'LineWidth', 1.2, 'Color', "#7E2F8E");
% % % 
% % % 
% % %     % Title based on data type
% % %     if strcmpi(data_type, 'raw')
% % %         title('Average Power Spectral Density - Raw Data');
% % %     elseif strcmpi(data_type, 'filt')
% % %         title('Average Power Spectral Density - Filtered Data');
% % %     else
% % %         title('Average Power Spectral Density');
% % %     end
% % % 
% % %     if strcmpi(class, 'opened')
% % %         subtitle(sprintf('Subject: %s (Opened Nose)', subject_name));
% % %     elseif strcmpi(class, 'closed')
% % %         subtitle(sprintf('Subject: %s (Closed Nose)', subject_name));
% % %     else
% % %         subtitle(sprintf('Subject: %s', subject_name));
% % %     end
% % % 
% % %     xlabel('Frequency (Hz)');
% % %     ylabel('PSD (dB)');
% % %     xlim([0 256]);
% % %     ylim([-80 40]);
% % %     grid on;

    % Display SNR values
    fprintf('    SNR values for %s (%s nose):\n', subject_name, class);
    for ch = 1:n_channels
        fprintf('      %s: %.2f dB\n', channels{ch}, snr_values(ch));
    end
end
