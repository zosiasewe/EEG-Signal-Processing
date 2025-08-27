function EEG_filtred = Filtering(EEG_Data, low_cutoff, high_cutoff, filter_order, n_channels, sampling_rate)
    nyquist_freq = sampling_rate / 2;
    [b_highpass, a_highpass] = butter(filter_order, low_cutoff / nyquist_freq, 'high');
    [b_lowpass, a_lowpass] = butter(filter_order, high_cutoff / nyquist_freq, 'low');

    [n_ch, n_time] = size(EEG_Data);  % EEG_Data: [channels x time]
    filtered_signal = zeros(n_ch, n_time);

    for ch = 1:n_ch
        highpass_signal = filtfilt(b_highpass, a_highpass, EEG_Data(ch, :));
        filtered_signal(ch, :) = filtfilt(b_lowpass, a_lowpass, highpass_signal);
    end

    EEG_filtred = filtered_signal; % return filtered [channels x time]
end
