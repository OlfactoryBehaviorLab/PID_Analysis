import os

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

import DewanPID_Utils
from Analyze_Contamination import BIT_CONVERSION_FACTOR


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


def get_intersection(curve_1, curve_2):
    intersects = np.where(np.isclose(curve_1, curve_2, rtol=1e-3, atol=1e-3))[0]
    return intersects
