function plotIndividualEpochs(EEG_data, epoch_samples, epoch_time, subject_name, class, n_channels, channel_offset, channels, data_type)
    figure;

    for trial_idx = 1:length(epoch_samples)
        subplot(2,2,trial_idx);
        hold on;

        trial_data = EEG_data(:, :, trial_idx);  % [channels x samples]


        for ch = 1:n_channels
            plot(epoch_time, trial_data(ch, :) + (ch-1)*channel_offset, 'LineWidth', 1.2);
        end

        if strcmpi(data_type, 'raw')
            title(sprintf('Trial %d - Raw EEG', epoch_samples(trial_idx)));
        elseif strcmpi(data_type, 'filt')
            title(sprintf('Trial %d - Filtered EEG (0.5-30 Hz)', epoch_samples(trial_idx)));
        else
            title(sprintf('Trial %d - EEG', epoch_samples(trial_idx)));
        end

        if strcmpi(class, 'opened')
            subtitle(sprintf('Subject: %s (Opened Nose)', subject_name));
        elseif strcmpi(class, 'closed')
            subtitle(sprintf('Subject: %s (Closed Nose)', subject_name));
        else
            subtitle(sprintf('Subject: %s', subject_name));
        end

        xlabel('Time (s)');
        ylabel('Channels');
        yticks((0:n_channels-1)*channel_offset);
        yticklabels(channels(1:n_channels));
        ylim([-channel_offset, channel_offset * n_channels]);
        xlim([0, max(epoch_time)]);
        grid on;
        hold off;
    end
end
