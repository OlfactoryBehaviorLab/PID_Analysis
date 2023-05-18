import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import signal

import DewanPID_Utils

BIT_CONVERSION_FACTOR = 4.8828


# TODO Make each odor-concentration pair the same color
# TODO Passofftime - trialdur == "fv on time" for the new setup
# TODO Control==7 Tube == 8 trialTypes

def get_auc(x, y):
    trap = np.trapz(y, x)
    # Use trapezoid rule to take integral of x,y data
    return trap


def normalize_data(baseline_shift_data, sniff_data_array, pass_end_index):
    # max_index = np.where(pass_valve_off == time_stamp_array)[0][0]  # Find the index for the passivation end time
    max_value_first_index = pass_end_index - 500
    max_value_second_index = pass_end_index - 100
    max_value = np.mean(sniff_data_array[max_value_first_index:max_value_second_index])
    # Get an average of the maximum values 400msec before passivation ends

    normalize_factor = max_value * 0.01
    # Calculate our normalization factor (max / 100)
    output_data = np.divide(baseline_shift_data, normalize_factor)
    # Divide all datapoints by normalization factor
    return output_data


def baseline_shift(data):
    baseline = abs(np.min(data))
    data = np.subtract(data, baseline)
    return data


def smooth_data(sniff_data, window_size=15, mode='reflect'):
    smoothed_data = uniform_filter1d(sniff_data, size=window_size, mode=mode)
    # Run moving window average on datas
    return smoothed_data


def bits_2_volts(bit_data, gain, carrier):
    gain_adjusted = bit_data / gain
    flow_dilution_adjusted = gain_adjusted / (carrier / 900)
    bits_to_volt = flow_dilution_adjusted * BIT_CONVERSION_FACTOR
    # If we want the graph to be in mV, the data needs to be adjusted by the gain, flow rate, and BIT_CONVERSION

    return bits_to_volt


def generate_header(cutoff_percentages):
    column_labels = ['Odor Name', 'Odor Concentration', 'Tube Length', 'Dilutor', 'Carrier', 'Passivation AUC',
                     'Passivation Time']
    # List of labels that are not repeated
    for each in cutoff_percentages:
        auc_label = f'Depassivation AUC ({each})'
        time_delay_label = f'Depassivation Time ({each})'
        column_labels.append(auc_label)
        column_labels.append(time_delay_label)
    # Generate labels for each Passivation Cutoff
    column_labels.append('Trial Type')

    return column_labels


def save_data(data, file_path, file_stem, fig, cutoff_percentages):
    column_labels = generate_header(cutoff_percentages)
    # Generate headers for dataframe
    output = pd.DataFrame(data, columns=column_labels)
    # Put data into dataframe
    csv_path = os.path.join(file_path, f'{file_stem}-CONTAMINATION.csv')
    tiff_path = os.path.join(file_path, f'{file_stem}.tiff')
    # Generate file paths for image and dcsv
    output.to_csv(csv_path, index=False)
    fig.savefig(tiff_path, dpi=600)
    # Save files to disk


def are_keys_in_string(keys, string):
    for key in keys:
        if np.logical_and((len(key) == len(string)), (key in string)):
            return True
    return False
    # Look for a string in a set of keys


def get_non_odor_trials(odor_names):
    keys = ['blank', 'mo', 'h2o']

    indexes = [are_keys_in_string(keys, str.lower(item)) for item in odor_names]
    indexes = np.logical_not(indexes)
    # See if there are any non-odor names in odors_names
    good_indexes = np.arange(len(odor_names))[indexes]
    # Flip the indexes to get the good ones
    return good_indexes


def get_troughs(normalized_data):
    troughs = signal.argrelextrema(normalized_data[1000:1300], np.less_equal, order=5)[0]
    # Find low points throughout the data and return the indexes
    good_indexes = DewanPID_Utils.get_roi(10, 300, troughs)
    # Sometimes it erroneously makes the trough too early,
    # This makes sure it is always in the middle of the
    # passivation Time; sometimes the curves are very stretched, hence checking out to 300
    troughs = troughs[good_indexes]
    # Even if we get two adjacent numbers, they're close enough to just take the first/last
    t1 = troughs[0] + 1000
    t2 = troughs[-1] + 1000
    # Return the first index and last index, they should represent our two troughs
    # Add 1000 to offset our troughs back to our original index
    return t1, t2


def linear_extrapolation(normalized_data):
    t1, t2 = get_troughs(normalized_data)
    # Get the indexes of our troughs
    num_nums = t2 - t1
    # Number of items to generate
    new_y = np.linspace(normalized_data[t1], normalized_data[t2], num=num_nums)
    # Linearly generate numbers between the first and second trough, forms a straight line
    # new_normalized = np.copy(normalized_data)
    normalized_data[t1:t2] = new_y
    # Replace the old data with the newly generated points
    return normalized_data, t1


def get_passivation_rate(time_stamp_array, normalized_data, passivation_start_index, passivation_end_index):

    # start_roi = time_stamp_array[passivation_start_index]  # Get time in seconds for the start of passivation
    # end_roi = time_stamp_array[passivation_end_index]  # Get time in seconds for the end of passivation

    # Our initial point is where the passivation starts
    # roi = DewanPID_Utils.get_roi(start_roi, end_roi, time_stamp_array)
    # Get the ROI for the passivation, from the start point to 1000msec before the pass valve switches off
    x_vals = time_stamp_array[passivation_start_index:passivation_end_index]
    y_vals = normalized_data[passivation_start_index:passivation_end_index]
    passivation_auc = get_auc(x_vals, y_vals)
    # Calculate AUC for the passivation area
    passivation_time_delay = np.sum(np.diff(x_vals))
    # Add up all the time differences to get the delay for the passivation
    return passivation_auc, passivation_time_delay


def get_depassivation_rate(time_stamp_array, normalized_data, passivation_off_index, cutoff_percentages):
    # off_index = np.where(time_stamp_array == pass_off_time)[0]
    # The index where the passivation valve switches off
    max_value = normalized_data[passivation_off_index]
    # Passivation starts exactly where the valve switches off
    cutoff_values = np.multiply(max_value, np.divide(cutoff_percentages, 100))
    # Calculate the various percentages of the max value
    auc_vals = []
    time_diffs = []

    for each in cutoff_values:
        roi = DewanPID_Utils.get_roi(each, max_value, normalized_data)
        time_diff = roi[-1] - passivation_off_index
        # Get the roi for each cutoff
        # The time difference is where the roi ends - where the valve switches off
        x_vals = time_stamp_array[roi]
        y_vals = normalized_data[roi]

        auc_vals.append(get_auc(x_vals, y_vals))
        time_diffs.append(time_diff)
        # Calculate the AUC and append the data to our return list

    return auc_vals, time_diffs


def combine_indexes(type_6, type_7, type_8, odor_trials, good_conc, good_tube):
    good_indexes_1 = np.intersect1d(good_tube, odor_trials)
    good_indexes_2 = np.intersect1d(good_conc, good_indexes_1)
    good_trials = np.concatenate((type_6, type_7, type_8))
    temp_array_1 = np.intersect1d(good_trials, good_indexes_2)
    temp_array_1 = np.sort(temp_array_1)
    return temp_array_1


def main():
    # # # Configurables # # #
    max_tube_length = 1000
    min_concentration = 0.01
    sec_before = 1
    sec_after = 30
    cutoff_percentages = [5, 10, 25, 50, 75, 95]
    # # # Configurables # # #

    file_path, file_stem, file_folder = DewanPID_Utils.get_file()
    h5_file = DewanPID_Utils.open_h5_file(file_path)
    # Open our data file

    data_type_6 = []
    data_type_7 = []
    data_type_8 = []
    # Empty list for our trial data

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

    for i in range(0, number_of_trials):  # Loop through all of our trials
        print(i)
        if i == 30:
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
            result.extend('7')
            data_type_7.append(result)
        elif i in type_8_trials:
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

    fig.show()
    # Display the figure


if __name__ == '__main__':
    main()
