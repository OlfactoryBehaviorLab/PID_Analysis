import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import signal

import DewanPID_Utils

BIT_CONVERSION_FACTOR = 4.8828


# TODO Make each odor-concentration pair the same color

def get_auc(x, y):
    trap = np.trapz(y, x)
    # Use trapezoid rule to take integral of x,y data
    return trap


def normalize_data(sniff_data):
    normalize_factor = np.max(sniff_data) * 0.01
    # Calculate our normalization factor (max / 100)
    output_data = np.divide(sniff_data, normalize_factor)
    # Divide all datapoints by normalization factor
    return output_data


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
    # See if there are any non-odor names in odors_names
    indexes = np.logical_not(indexes)
    # Flip the indexes to get the good ones
    return indexes


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


def get_passivation_rate(time_stamp_array, troughless_data, pass_off_time, passivation_start):
    start_roi = time_stamp_array[passivation_start]
    # Our initial point is where the passivation starts
    roi = DewanPID_Utils.get_roi(start_roi, pass_off_time, time_stamp_array)
    # Get the ROI for the passivation, from the start point to 1000msec before the pass valve switches off
    x_vals = time_stamp_array[roi]
    y_vals = troughless_data[roi]
    passivation_auc = get_auc(x_vals, y_vals)
    # Calculate AUC for the passivation area
    passivation_time_delay = np.sum(np.diff(time_stamp_array[roi]))
    # Add up all the time differences to get the delay for the passivation
    return passivation_auc, passivation_time_delay


def get_depassivation_rate(time_stamp_array, troughless_data, pass_off_time, cutoff_percentages):
    off_index = np.where(time_stamp_array == pass_off_time)[0]
    # The index where the passivation valve switches off
    max_value = troughless_data[off_index]
    # Passivation starts exactly where the valve switches off
    cutoff_values = np.multiply(max_value, np.divide(cutoff_percentages, 100))
    # Calculate the various percentages of the max value
    auc_vals = []
    time_diffs = []

    for each in cutoff_values:
        roi = DewanPID_Utils.get_roi(each, max_value, troughless_data)
        time_diff = roi[-1] - off_index
        # Get the roi for each cutoff
        # The time difference is where the roi ends - where the valve switches off
        x_vals = time_stamp_array[roi]
        y_vals = troughless_data[roi]

        auc_vals.append(get_auc(x_vals, y_vals))
        time_diffs.append(time_diff[0])
        # Calculate the AUC and append the data to our return list

    return auc_vals, time_diffs


def main():
    # # # Configurables # # #
    max_tube_length = 200
    min_concentration = 1
    sec_before = 1
    sec_after = 10
    cutoff_percentages = [5, 10, 25, 50, 75, 95]
    # # # Configurables # # #

    file_path, file_stem, file_folder = DewanPID_Utils.get_file()
    h5_file = DewanPID_Utils.open_h5_file(file_path)
    # Open our data file

    data = []
    # Empty list for our trial data

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_6_trials = np.where(trials['trialtype'] == 6)[0]
    # We are only interested when trialtype is six. The first trial is usually a 0/1

    odor_name = DewanPID_Utils.decode_list(trials['odor'][type_6_trials])
    # odor_names are stored as bytes and need to be decoded using utf-8
    odor_indexes = get_non_odor_trials(odor_name)  # Remove Blanks and MO if present

    type_6_trials = np.array(type_6_trials)[odor_indexes]  # Select only the odor trials

    odor_concentration = trials['odorconc'][type_6_trials]
    good_concentrations = np.where(odor_concentration > min_concentration)[0]
    # Filter out concentrations under our cutoff; sometimes the low concentrations are not good due to signal:noise
    odor_concentration = odor_concentration[good_concentrations]
    type_6_trials = type_6_trials[good_concentrations]

    tube_length = trials['Tube'][type_6_trials]
    good_tubes = np.where(tube_length < max_tube_length)[0]
    # Filter out tube lengths over a certain length

    type_6_trials = type_6_trials[good_tubes]

    odor_name = odor_name[type_6_trials]
    final_valve_on_time = trials['fvOnTime'][type_6_trials]
    pid_gain = trials['PIDGain'][type_6_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_6_trials]
    diluter_flowrate = trials['Dilutor_flowrate'][type_6_trials]
    pass_valve_off_time = trials['PassOffTime'][type_6_trials]
    # ^^^ Get all data out

    num_type_6_trials = len(type_6_trials)

    fig, ax = plt.subplots()
    # Create empty plot to put traces into

    for i in range(num_type_6_trials):  # Loop through all of our trials
        trial_number = type_6_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = DewanPID_Utils.get_sniff_data(h5_file, trial_name)
        # Get sniff data out of H5 File; this contains our actual PID measurement

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = DewanPID_Utils.condense_packets(sniff_data,
                                                                             sniff_samples, packet_sent_time)
        # The packets come out in chunks, we want to linearize them into a long list to pick and choose from

        final_valve_on = final_valve_on_time[i]
        pass_valve_off = (pass_valve_off_time[i] - final_valve_on) / 1000

        TOI_start = final_valve_on - sec_before * 1000
        TOI_end = final_valve_on + sec_after * 1000
        # We don't want all the data, so cut it off some before and some after the final valve goes off

        roi_index = DewanPID_Utils.get_roi(TOI_start, TOI_end, time_stamp_array)
        # Get all the data between our cutoff values

        sniff_data_array = sniff_data_array[roi_index]
        time_stamp_array = time_stamp_array[roi_index]
        time_stamp_array = (time_stamp_array - final_valve_on) / 1000

        smoothed_data = smooth_data(sniff_data_array)  # Run a moving window
        normalized_data = normalize_data(smoothed_data)  # Normalize all the data between 0-100
        troughless_data, passivation_start = linear_extrapolation(normalized_data)
        # Remove the weird pressure spike by identifying the troughs and running linear interpolation

        # ax.axvline(x=time_stamp_array[t1])
        # ax.axvline(x=time_stamp_array[t2])
        passivation_auc, passivation_time_delay = get_passivation_rate(time_stamp_array, normalized_data,
                                                                       pass_valve_off, passivation_start)
        # Get passivation areas and time delays for the rising curve

        depassivation_aucs, depassivation_time_delays = get_depassivation_rate(time_stamp_array, normalized_data,
                                                                               pass_valve_off, cutoff_percentages)
        # Get the depassivation areas and time delays for all the cutoff values

        ax.plot(time_stamp_array, troughless_data)
        # Plot the traces on top of each other

        result = [odor_name[i], odor_concentration[i], tube_length[i], diluter_flowrate[i], carrier_flowrate[i],
                  passivation_auc, passivation_time_delay]
        # Compile all of our data into a list for each trial

        for z, each in enumerate(depassivation_time_delays):
            result.append(depassivation_aucs[z])
            result.append(each)
        # The depassivation values come out as a list, so unpack them and add them to the list

        data.append(result)
        # Add our data for this trial to the complete list of data for export

    h5_file.close()

    save_data(data, file_folder, file_stem, fig, cutoff_percentages)
    # Output the data and the figure

    fig.show()
    # Display the figure


if __name__ == '__main__':
    main()
