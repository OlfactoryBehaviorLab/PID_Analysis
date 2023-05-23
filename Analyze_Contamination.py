import numpy as np
import matplotlib.pyplot as plt

import Dewan_Contamination_Utils
import Dewan_PID_Utils
from Dewan_Contamination_Utils import normalize_data, baseline_shift, smooth_data, save_data, get_non_odor_trials, \
    get_passivation_rate, get_depassivation_rate, combine_indexes, get_concentration_type_pairs, \
    parse_concentration_data


## TODO Make each odor-concentration pair the same color


def main():
    # # # Configurables # # #
    max_tube_length = 1000
    min_concentration = 0.01
    sec_before = 1
    sec_after = 30
    cutoff_percentages = [5, 10, 25, 50, 75, 95]
    bad_trials = [14]
    # 14 for butyl acetate
    # S Limonene 13, 30, 31
    # # # Configurables # # #

    file_path, file_stem, file_folder = Dewan_PID_Utils.get_file()
    h5_file = Dewan_PID_Utils.open_h5_file(file_path)
    # Open our data file

    type_6_results = []
    type_7_results = []
    type_8_results = []
    # Empty list for our trial data

    all_raw_data = []
    passivation_end_index_list = []

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_6_trials = np.where(trials['trialtype'] == 6)[0]
    type_7_trials = np.where(trials['trialtype'] == 7)[0]
    type_8_trials = np.where(trials['trialtype'] == 8)[0]
    # We are only interested when trialtype is six. The first trial is usually a 0/1
    odor_name = Dewan_PID_Utils.decode_list(trials['odor'])
    # odor_names are stored as bytes and need to be decoded using utf-8
    odor_indexes = get_non_odor_trials(odor_name)  # Remove Blanks and MO if present
    odor_concentration = trials['odorconc']
    good_concentrations = np.where(odor_concentration > min_concentration)[0]
    # Filter out concentrations under our cutoff; sometimes the low concentrations are not good due to signal:noise
    tube_length = trials['Tube']
    good_tubes = np.where(tube_length < max_tube_length)[0]
    # Filter out tube lengths over a certain length
    good_trials = combine_indexes(type_6_trials, type_7_trials, type_8_trials, odor_indexes, good_concentrations,
                                  good_tubes)
    # type_6_trials = [each for each in type_6_trials if each in good_trials]
    # type_7_trials = [each for each in type_7_trials if each in good_trials]
    # type_8_trials = [each for each in type_8_trials if each in good_trials]
    # type_6_trials = [each for each in type_6_trials if each not in bad_trials]
    # type_7_trials = [each for each in type_7_trials if each not in bad_trials]
    # type_8_trials = [each for each in type_8_trials if each not in bad_trials]

    # Make sure anything that was filtered out by good_trials is reflected in the trial lists

    # Find the common items between all our cutoffs/filters
    # odor_name = odor_name[good_trials]
    # carrier_flowrate = trials['Carrier_flowrate'][good_trials]
    # diluter_flowrate = trials['Dilutor_flowrate'][good_trials]
    # pass_valve_off_time = trials['PassOffTime'][good_trials]
    # pass_valve_on_time = trials['PassOnTime'][good_trials]
    # odor_concentration = odor_concentration[good_trials]

    carrier_flowrate = trials['Carrier_flowrate']
    diluter_flowrate = trials['Dilutor_flowrate']
    pass_valve_off_time = trials['PassOffTime']
    pass_valve_on_time = trials['PassOnTime']

    # ^^^ Get all data out

    number_of_trials = len(good_trials)
    fig, ax = plt.subplots()
    # Create empty plot to put traces into
    new_type_7_trial = []
    new_type_8_trial = []
    type_8_pass_end = []
    type_7_pass_end = []

    for i, trial in enumerate(good_trials):  # Loop through all of our trials
        if trial in bad_trials:  # Provide a way to skip corrupted trials
            continue

        # trial_number = good_trials[i]
        trial_name = trial_names[trial]
        event_data, sniff_data = Dewan_PID_Utils.get_sniff_data(h5_file, trial_name)
        # Get sniff data out of H5 File; this contains our actual PID measurement
        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']
        sniff_data_array, time_stamp_array = Dewan_PID_Utils.condense_packets(sniff_data,
                                                                              sniff_samples, packet_sent_time)
        # The packets come out in chunks, we want to linearize them into a long list to pick and choose from
        # Probably should change this one day as this seems inefficient
        # final_valve_on_time = pass_valve_off_time[i] - trial_duration[i] - 1
        # final_valve_on_time_msec = final_valve_on_time / 1000

        passivation_start_time = pass_valve_on_time[trial]  # When passivation starts in msec
        passivation_stop_time = pass_valve_off_time[trial]  # When passivation ends in msec
        TOI_start = passivation_start_time - (sec_before * 1000)
        TOI_end = passivation_stop_time + (sec_after * 1000)
        # We don't want all the data, so cut it before and after the passivation times

        roi_index = Dewan_PID_Utils.get_roi(TOI_start, TOI_end, time_stamp_array)
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
                                                                               passivation_end_index,
                                                                               cutoff_percentages)
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
            type_6_results.append(result)
        elif i in type_7_trials:
            passivation_end_index_list.append(passivation_end_index)
            all_raw_data.append(normalized_data)
            new_type_7_trial.append(len(all_raw_data)-1)
            result.extend('7')
            type_7_results.append(result)
        elif i in type_8_trials:
            passivation_end_index_list.append(passivation_end_index)
            all_raw_data.append(normalized_data)
            new_type_8_trial.append(len(all_raw_data)-1)
            result.extend('8')
            type_8_results.append(result)


        # Add our data for this trial to the complete list of data for export
    # data_to_save = []
    # for each in type_6_results:
    #     data_to_save.append(each)
    # for each in type_7_results:
    #     data_to_save.append(each)
    # for each in type_8_results:
    #     data_to_save.append(each)

    h5_file.close()
    # fig.show()
    # Display the figure

    # save_data(data_to_save, file_folder, file_stem, fig, cutoff_percentages)
    # Output the data and the figure

    # # # New Functionality with Concentration Averaging # # #

    temp = []

    for j, each in enumerate(all_raw_data):
        cutoff = passivation_end_index_list[j]
        temp.append(each[cutoff:])

    all_raw_data = temp
    # Cuts each row to its depassivation end time
    raw_data_max = np.min([len(each) for each in all_raw_data])
    # We need to trim all lists to the shortest time
    truncated_raw_data = [each[:raw_data_max] for each in all_raw_data]
    # Shorten all the lists to the minimum and maximum indexes
    end_timestamp = len(truncated_raw_data[0]) - 75
    time_stamp_array = (np.arange(end_timestamp) / 1000)
    # Generate X-values from 0 -> length of the arrays; we start at zero since all data before the
    # passivation_off_valve is chopped off
    truncated_raw_data = np.array(truncated_raw_data)

    unique_concentrations, type_7_trial_pairs, type_8_trial_pairs = get_concentration_type_pairs(odor_concentration,
                                                                                                 new_type_7_trial,
                                                                                                 new_type_8_trial)
    # We want to get the indexes for each trial type for each concentration
    for i in range(len(unique_concentrations)):
        fig2, ax2 = plt.subplots()
        for each in type_7_trial_pairs[i]:
            ax2.plot(time_stamp_array, truncated_raw_data[each][:-75], color='b')
        for each in type_8_trial_pairs[i]:
            ax2.plot(time_stamp_array, truncated_raw_data[each][75:], color='r')
        fig2.show()

    allFig, allAx = plt.subplots()
    allFig.suptitle('All Concentrations')

    control_AUC = []
    control_time_delay = []
    test_AUC = []
    test_time_delay = []

    for i, each in enumerate(unique_concentrations):
        line_types = ['solid', 'dotted', 'dashed', 'dashdot']
        plot, axis = plt.subplots()
        plot.suptitle(f'Concentration: {each}')
        concentration_type_7_trials = type_7_trial_pairs[i]
        concentration_type_8_trials = type_8_trial_pairs[i]
        type_7_data_average = np.mean(truncated_raw_data[concentration_type_7_trials], axis=0)[75:]
        type_8_data_average = np.mean(truncated_raw_data[concentration_type_8_trials], axis=0)[:-75]
        # We need to shift the data by ~75 ms, so they overlap. Dunno why.... Timestamp shifted above
        control_aucs, control_time_delays = get_depassivation_rate(time_stamp_array,
                                                                   type_7_data_average, 0,
                                                                   cutoff_percentages)

        test_aucs, test_time_delays = get_depassivation_rate(time_stamp_array,
                                                             type_8_data_average, 0,
                                                             cutoff_percentages)

        control_AUC.append(control_aucs)
        control_time_delay.append(control_time_delays)

        test_AUC.append(test_aucs)
        test_time_delay.append(test_time_delays)

        axis.plot(time_stamp_array, type_7_data_average, linestyle=line_types[i], color='b')
        axis.plot(time_stamp_array, type_8_data_average, linestyle=line_types[i], color='r')
        allAx.plot(time_stamp_array, type_7_data_average, linestyle=line_types[i], color='b')
        allAx.plot(time_stamp_array, type_8_data_average, linestyle=line_types[i], color='r')

        plot.show()

    auc_ratios, auc_differences, time_ratios, time_differences = parse_concentration_data(control_AUC,
                                                                                          control_time_delay, test_AUC,
                                                                                          test_time_delay)

    Dewan_Contamination_Utils.save_concentration_data(file_folder, file_stem, odor_name[0], tube_length[0],
                                                      unique_concentrations, cutoff_percentages, auc_ratios,
                                                      auc_differences, time_ratios, time_differences)

    allFig.show()


if __name__ == '__main__':
    main()
