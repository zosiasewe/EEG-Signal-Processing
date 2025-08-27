function plotICA(ica_reshaped, sampling_rate, epoch_time, subject_name, class, trial_idx)
    n_components = size(ica_reshaped, 1);

    % PSD plot
    figure;
    for comp = 1:n_components
        subplot(2,2,comp);
        [psd_comp, freq_comp] = pwelch(ica_reshaped(comp, :, trial_idx), [], [], [], sampling_rate);
        plot(freq_comp, 10*log10(psd_comp), 'LineWidth', 1.5);
        title(sprintf('ICA Component %d - PSD', comp));
        subtitle(sprintf('Subject: %s (%s Nose)', subject_name, class));
        xlabel('Frequency (Hz)'); ylabel('Power (dB)');
        xlim([0 100]); grid on;
    end

    % Time series plot
    figure;
    offset = 50; hold on;
    for comp = 1:n_components
        plot(epoch_time, ica_reshaped(comp, :, trial_idx) + (comp-1)*offset);
    end
    title(sprintf('ICA Components - Trial %d', trial_idx));
    subtitle(sprintf('Subject: %s (%s Nose)', subject_name, class));
    xlabel('Time (s)'); ylabel('Components');
    yticks((0:n_components-1)*offset);
    ylim([-offset, n_components*offset]);
    grid on; hold off;
end
