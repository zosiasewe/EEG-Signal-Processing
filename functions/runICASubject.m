function clean_eeg = runICASubject(EEG_data, n_components, sampling_rate, subject_name, class, epoch_time)
    [n_channels, n_samples, n_trials] = size(EEG_data);
    clean_eeg = zeros(n_channels, n_samples, n_trials);

    %  [channels x (samples*trials)]
    EEG_2d = reshape(EEG_data, n_channels, []);

    [ica_components, W_matrix, A_matrix] = runICA(EEG_2d, n_components, 500, 1e-6, 1);

    %  [components x samples x trials]
    ica_reshaped = reshape(ica_components, n_components, n_samples, n_trials);

    trial_for_ica = 1;
%     plotICA(ica_reshaped, sampling_rate, epoch_time, subject_name, class, trial_for_ica);

    % Artifact removal
    artifact_components = [];
    if ~isempty(artifact_components)
        A_clean = A_matrix;
        A_clean(:, artifact_components) = 0;
        clean_components = ica_components;
        clean_components(artifact_components,:) = 0;
        clean_signal = A_clean * clean_components;
        clean_eeg = reshape(clean_signal, n_channels, n_samples, n_trials);
    else
        clean_eeg = EEG_data;
    end
end
