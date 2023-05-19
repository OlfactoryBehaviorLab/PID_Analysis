import numpy as np
import matplotlib.pyplot as plt

import DewanPID_Utils
import Dewan_Contamination_Utils
from Dewan_Contamination_Utils import normalize_data, baseline_shift, smooth_data, save_data, get_non_odor_trials, \
    get_passivation_rate, get_depassivation_rate, combine_indexes, get_intersection


## TODO Make each odor-concentration pair the same color


def main():
    # # # Configurables # # #
    max_tube_length = 1000
    min_concentration = 0.01
    sec_before = 1
    sec_after = 30
    cutoff_percentages = [5, 10, 25, 50, 75, 95]
    bad_trials = [13, 30, 31]
    # # # Configurables # # #

    file_path, file_stem, file_folder = DewanPID_Utils.get_file('R:/PID_CF/Contamination/RAW Files/odorContaminationSLimoneneCONCSeries_sess1_D2023_5_17T15_2_52.h5')
    h5_file = DewanPID_Utils.open_h5_file(file_path)
    # Open our data file

    data_type_6 = []
    data_type_7 = []
    data_type_8 = []
    # Empty list for our trial data
    plot_type_7 = []
    plot_type_8 = []


    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_6_trials = np.where(trials['trialtype'] == 6)[0]
    type_7_trials = np.where(trials['trialtype'] == 7)[0]
    type_8_trials = np.where(trials['trialtype'] == 8)[0]

    # We are only interested when trialtype is six. The first trial is usually a 0/1
    odor_name = DewanPID_Utils.decode_list(trials['odor'])
    # odor_names are stored as bytes and need to be decoded using utf-8
    odor_indexes = get_non_odor_trials(odor_name)  # Remove Blanks and MO if present
    odor_concentration = trials['odorconc']
    good_concentrations = np.where(odor_concentration > min_concentration)[0]
    # Filter out concentrations under our cutoff; sometimes the low concentrations are not good due to signal:noise
    tube_length = trials['Tube']
    good_tubes = np.where(tube_length < max_tube_length)[0]
    # Filter out tube lengths over a certain length
    good_trials = combine_indexes(type_6_trials, type_7_trials, type_8_trials, odor_indexes, good_concentrations, good_tubes)
    # Find the common items between all our cutoffs/filters
    odor_name = odor_name[good_trials]
    carrier_flowrate = trials['Carrier_flowrate'][good_trials]
    diluter_flowrate = trials['Dilutor_flowrate'][good_trials]
    pass_valve_off_time = trials['PassOffTime'][good_trials]
    pass_valve_on_time = trials['PassOnTime'][good_trials]
    trial_duration = trials['trialdur'][good_trials]
    # ^^^ Get all data out

    number_of_trials = len(good_trials)
    fig, ax = plt.subplots()
    # Create empty plot to put traces into

    unique_conc, type_7_trial_per_concentration, type_8_trial_per_concentration = \
        Dewan_Contamination_Utils.get_trial_concentration_pairs(odor_concentration, type_7_trials, type_8_trials)


    for i, current_concentration in enumerate(unique_conc):  # Loop through all of our trials

        for trial in type_7_trial_per_concentration[i]:
            if i in bad_trials:  # Provide a way to skip corrupted trials
                continue


            trial_number = good_trials[i]
            trial_name = trial_names[trial_number]
            event_data, sniff_data = DewanPID_Utils.get_sniff_data(h5_file, trial_name)
            # Get sniff data out of H5 File; this contains our actual PID measurement
            sniff_samples = event_data['sniff_samples']
            packet_sent_time = event_data['packet_sent_time']
            sniff_data_array, time_stamp_array = DewanPID_Utils.condense_packets(sniff_data,
                                                                                 sniff_samples, packet_sent_time)
            # The packets come out in chunks, we want to linearize them into a long list to pick and choose from
            # Probably should change this one day as this seems inefficient
            # final_valve_on_time = pass_valve_off_time[i] - trial_duration[i] - 1
            # final_valve_on_time_msec = final_valve_on_time / 1000

            passivation_start_time = pass_valve_on_time[i]  # When passivation starts in msec
            passivation_stop_time = pass_valve_off_time[i]  # When passivation ends in msec
            TOI_start = passivation_start_time - (sec_before * 1000)
            TOI_end = passivation_stop_time + (sec_after * 1000)
            # We don't want all the data, so cut it before and after the passivation times

            roi_index = DewanPID_Utils.get_roi(TOI_start, TOI_end, time_stamp_array)
            # Get the ROI from the time_stamp_array in msecs
            # Get all the data between our cutoff values
            sniff_data_array = sniff_data_array[roi_index]
            time_stamp_array = time_stamp_array[roi_index]
            passivation_start_index = np.where(passivation_start_time == time_stamp_array)[0][0]
            passivation_end_index = np.where(passivation_stop_time == time_stamp_array)[0][0]
            # Get indexes for the start/stop time before it's converted to msec
            time_stamp_array = (time_stamp_array - passivation_start_time) / 1000
            # Convert time_stamp_array to seconds for plotting

            smoothed_data = smooth_data(sniff_data_array)  # Run a moving window average

            baseline_shifted_data = baseline_shift(smoothed_data)  # Do a baseline shift so the lowest value is 0

            normalized_data = normalize_data(baseline_shifted_data, sniff_data_array, passivation_end_index)
            # Normalize all the data between 0-100
            # troughless_data, passivation_start = linear_extrapolation(normalized_data)
            # Remove the weird pressure spike by identifying the troughs and running linear interpolation
            # Might not need this anymore 5/17/23 ACP

        passivation_auc, passivation_time_delay = get_passivation_rate(time_stamp_array, normalized_data,
                                                                       passivation_start_index, passivation_end_index)
        # Get passivation areas and time delays for the rising curve

        depassivation_aucs, depassivation_time_delays = get_depassivation_rate(time_stamp_array, normalized_data,
                                                                               passivation_end_index, cutoff_percentages)
        # Get the depassivation areas and time delays for all the cutoff values
        ax.plot(time_stamp_array, normalized_data)
        x_start = time_stamp_array[passivation_start_index]
        x_end = time_stamp_array[passivation_end_index]

        # Plot the traces on top of each other

        result = [odor_name[i], odor_concentration[i], tube_length[i], diluter_flowrate[i], carrier_flowrate[i],
                  passivation_auc, passivation_time_delay]
        # Compile all of our data into a list for each trial

        for z, each in enumerate(depassivation_time_delays):
            result.append(depassivation_aucs[z])
            result.append(each)
        # The depassivation values come out as a list, so unpack them and add them to the list

        if i in type_6_trials:
            result.extend('6')
            data_type_6.append(result)
        elif i in type_7_trials:
            data = normalized_data[passivation_end_index+75:passivation_end_index+20000]
            plot_type_7.append(data)
            result.extend('7')
            data_type_7.append(result)
        elif i in type_8_trials:
            data = normalized_data[passivation_end_index:passivation_end_index+20000]
            plot_type_8.append(data)
            result.extend('8')
            data_type_8.append(result)

        # Add our data for this trial to the complete list of data for export
    data = []
    for each in data_type_6:
        data.append(each)
    for each in data_type_7:
        data.append(each)
    for each in data_type_8:
        data.append(each)
    h5_file.close()
    save_data(data, file_folder, file_stem, fig, cutoff_percentages)
    # Output the data and the figure
    plot_type_7 = np.mean(plot_type_7, axis=0)[:5000]
    plot_type_8 = np.mean(plot_type_8, axis=0)[:5000]

    intersect = get_intersection(plot_type_7, plot_type_8)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(plot_type_7)), plot_type_7)
    ax1.plot(np.arange(len(plot_type_8)), plot_type_8, color='r')

    if len(intersect) < 2:
        ax1.axvline(intersect, color='k')
    fig1.show()
    fig.show()
    # Display the figure

if __name__ == '__main__':
    main()
