function plotContinuousEEG(time, EEG_data_subj, n_channels, channel_offset, subject_name, class, data_type, channels)

    figure;
    hold on;

    for ch = 1:n_channels
        plot(time, EEG_data_subj(ch, :) + (ch - 1) * channel_offset, 'LineWidth', 1.2);
    end
    
    % Title based on data type
    if strcmpi(data_type, 'raw')
        title('Continuous Raw EEG Data - All Trials');
    elseif strcmpi(data_type, 'filt')
        title('Continuous Filtered EEG Data (0.5-30 Hz)');
    else
        title('Continuous EEG Data');
    end

    % Subtitle based on class
    if strcmpi(class, 'opened')
        subtitle(sprintf('Subject: %s (Opened Nose)', subject_name));
    elseif strcmpi(class, 'closed')
        subtitle(sprintf('Subject: %s (Closed Nose)', subject_name));
    else
        subtitle(sprintf('Subject: %s', subject_name));
    end

    xlabel('Time (s)');
    ylabel('Channels');
    yticks((0:n_channels-1) * channel_offset);
    yticklabels(channels(1:n_channels));
    ylim([-channel_offset, channel_offset * n_channels]);
    xlim([time(1), time(end)]);
    grid on;
    hold off;
end
