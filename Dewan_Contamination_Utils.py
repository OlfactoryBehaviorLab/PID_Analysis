import os

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

import Dewan_PID_Utils

BIT_CONVERSION_FACTOR = 4.8828


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


def get_troughs(normalized_data):
    troughs = signal.argrelextrema(normalized_data[1000:1300], np.less_equal, order=5)[0]
    # Find low points throughout the data and return the indexes
    good_indexes = Dewan_PID_Utils.get_roi(10, 300, troughs)
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


def bits_2_volts(bit_data, gain, carrier):
    gain_adjusted = bit_data / gain
    flow_dilution_adjusted = gain_adjusted / (carrier / 900)
    bits_to_volt = flow_dilution_adjusted * BIT_CONVERSION_FACTOR
    # If we want the graph to be in mV, the data needs to be adjusted by the gain, flow rate, and BIT_CONVERSION

    return bits_to_volt


def generate_standard_header(cutoff_percentages):
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
    column_labels = generate_standard_header(cutoff_percentages)
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
        roi = Dewan_PID_Utils.get_roi(each, max_value, normalized_data)
        time_diff = roi[-1] - passivation_off_index
        # Get the roi for each cutoff
        # The time difference is where the roi ends - where the valve switches off
        x_vals = time_stamp_array[roi]
        y_vals = normalized_data[roi]

        auc_vals.append(get_auc(x_vals, y_vals))
        time_diffs.append(time_diff)
        # Calculate the AUC and append the data to our return list

    return auc_vals, time_diffs


def combine_indexes(type_6, type_7, odor_trials, good_conc, good_tube):
    good_indexes_1 = np.intersect1d(good_tube, odor_trials)
    good_indexes_2 = np.intersect1d(good_conc, good_indexes_1)
    good_trials = np.concatenate((type_6, type_7))
    temp_array_1 = np.intersect1d(good_trials, good_indexes_2)
    temp_array_1 = np.sort(temp_array_1)
    return temp_array_1


def get_concentration_type_pairs(odor_concentration, type_7_trials, type_8_trials):
    unique_concentrations = np.unique(odor_concentration)
    type_7_concentration_trials = []
    type_8_concentration_trials = []
    # for each in unique_concentrations:
    #     type_7_concentration_trials.append(np.where(each == odor_concentration[type_7_trials])[0])
    #     type_8_concentration_trials.append(np.where(each == odor_concentration[type_8_trials])[0])

    for each in unique_concentrations:
        t7_temp = []
        t8_temp = []
        for t7 in type_7_trials:
            if odor_concentration[t7] == each:
                t7_temp.append(t7)
        for t8 in type_8_trials:
            if odor_concentration[t8] == each:
                t8_temp.append(t8)

        type_7_concentration_trials.append(t7_temp)
        type_8_concentration_trials.append(t8_temp)

    return unique_concentrations, type_7_concentration_trials, type_8_concentration_trials


def parse_concentration_data(control_AUC_array, control_time_delay_array, test_AUC_array, test_time_delay_array):
    auc_ratios = []
    auc_differences = []
    time_ratios = []
    time_differences = []
    for i, control_auc_vals in enumerate(control_AUC_array):
        control_time_vals = control_time_delay_array[i]
        test_AUC_vals = test_AUC_array[i]
        test_time_vals = test_time_delay_array[i]

        auc_ratio = np.divide(control_auc_vals, test_AUC_vals)
        auc_difference = np.subtract(test_AUC_vals, control_auc_vals)

        time_ratio = np.divide(control_time_vals, test_time_vals)
        time_difference = np.subtract(test_time_vals, control_time_vals)

        auc_ratios.append(auc_ratio)
        auc_differences.append(auc_difference)
        time_ratios.append(time_ratio)
        time_differences.append(time_difference)

    return auc_ratios, auc_differences, time_ratios, time_differences


def generate_concentration_header(odor, tube_length, cutoff_percentages):
    header = ['Concentration']

    for each in cutoff_percentages:
        auc_ratio = f'AUC Ratio({each})'
        auc_difference = f'AUC Difference({each})'
        time_ratio = f'Time Ratio({each})'
        time_difference = f'Time Difference({each})'

        header.append(auc_ratio)
        header.append(auc_difference)
        header.append(time_ratio)
        header.append(time_difference)

    header.append(f'Odor: {odor}')
    header.append(f'Tube Length {tube_length}mm')

    return header


def save_concentration_data(file_path, file_stem, odor_name, tube_length, unique_concentrations, cutoffs, auc_ratios, auc_differences,
                            time_ratios, time_differences):
    header = generate_concentration_header(odor_name, tube_length, cutoffs)
    data = []

    for i, each in enumerate(unique_concentrations):
        # Loop through all the concentrations, each row in the data is for a specific concentration
        concentration_result = [each]
        for j, auc_ratio in enumerate(auc_ratios[i]):
            # Loop through the items in each row, each item is for a specific cutoff
            # We rotate through the cutoffs since that is how the header is arranged
            concentration_result.append(auc_ratio)
            concentration_result.append(auc_differences[i][j])
            concentration_result.append(time_ratios[i][j])
            concentration_result.append(time_differences[i][j])

        concentration_result.append("")
        concentration_result.append("")

        data.append(concentration_result)

    data = pd.DataFrame(data, columns=header)
    csv_path = os.path.join(file_path, f'{file_stem}-CONCENTRATION.csv')
    # Generate file paths for image and csv
    data.to_csv(csv_path, index=False)


def get_on_off_times(trial_data):
    passivation_on_times = []
    passivation_off_times = []

    for time in range(1, 31):
        column_name = str(time)
        passivation_on_times.append(trial_data[f'{column_name}PassOnTime'])
        passivation_off_times.append(trial_data[f'{column_name}PassOffTime'])

    passivation_on_times = np.transpose(passivation_on_times)
    passivation_off_times = np.transpose(passivation_off_times)

    return passivation_on_times, passivation_off_times

