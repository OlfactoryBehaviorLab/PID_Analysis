import os

import numpy as np
import pandas as pd

import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import signal, interpolate

import DewanPID_Utils

BIT_CONVERSION_FACTOR = 4.8828


# TODO Rising Side (Do everything in reverse) Go to 95% of max
# TODO Save Normalization Figures
# TODO Make each odor-concentration pair the same color

def get_auc(x, y):
    trap = np.trapz(y, x)
    return trap


def normalize_data(sniff_data):
    normalize_factor = np.max(sniff_data) * 0.01
    output_data = np.divide(sniff_data, normalize_factor)

    return output_data


def smooth_data(sniff_data, window_size=15, mode='reflect'):
    smoothed_data = uniform_filter1d(sniff_data, size=window_size, mode=mode)
    return smoothed_data


def bits_2_volts(bit_data, gain, carrier):
    gain_adjusted = bit_data / gain
    flow_dilution_adjusted = gain_adjusted / (carrier / 900)
    bits_to_volt = flow_dilution_adjusted * BIT_CONVERSION_FACTOR

    return bits_to_volt


def find_upper_bound(sniff_data, lower_bound):
    maximum = np.max(sniff_data[lower_bound:])
    cutoff = maximum * 0.05
    cutoff_indexes = np.where(sniff_data[lower_bound:] > cutoff)[0]
    upper_bound = cutoff_indexes[-1] + lower_bound

    return upper_bound


def find_lower_bound(time_stamps):
    end_time = 2.1
    region_indexes = np.where(time_stamps >= end_time)[0]
    lower_bound = region_indexes[0]
    return lower_bound


def generate_header(cutoff_percentages):
    column_labels = ['Odor Name', 'Odor Concentration', 'Tube Length', 'Dilutor', 'Carrier', 'Passivation AUC',
                     'Passivation Time']

    for each in cutoff_percentages:
        auc_label = f'Depassivation AUC ({each})'
        time_delay_label = f'Depassivation Time ({each})'
        column_labels.append(auc_label)
        column_labels.append(time_delay_label)

    return column_labels

def save_data(data, file_path, file_stem, fig, cutoff_percentages):
    column_labels = generate_header(cutoff_percentages)

    output = pd.DataFrame(data, columns=column_labels)

    csv_path = os.path.join(file_path, f'{file_stem}-CONTAMINATION.csv')
    tiff_path = os.path.join(file_path, f'{file_stem}.tiff')

    output.to_csv(csv_path, index=False)
    fig.savefig(tiff_path, dpi=600)


def are_keys_in_string(keys, string):
    return any(key in string for key in keys)


def get_non_odor_trials(odor_names):
    keys = ['blank', 'mo', 'h2o']
    indexes = [are_keys_in_string(keys, str.lower(item)) for item in odor_names]
    indexes = np.logical_not(indexes)

    return indexes


def get_troughs(normalized_data):
    troughs = signal.argrelextrema(normalized_data[1000:1300], np.less_equal, order=5)[0]
    good_indexes = DewanPID_Utils.get_roi(10, 300, troughs)
    # Sometimes it erroneously makes the trough too early,
    # This makes sure it is always in the middle of the
    # passivation Time; sometimes the curves are very stretched, hence checking out to 300
    troughs = troughs[good_indexes]
    # Even if we get two adjacent numbers, they're close enough to just take the first/last
    t1 = troughs[0] + 1000
    t2 = troughs[-1] + 1000
    return t1, t2


def linear_extrapolation(normalized_data):
    t1, t2 = get_troughs(normalized_data)
    x_vals = np.arange(0, len(normalized_data))
    num_nums = t2-t1
    new_y = np.linspace(normalized_data[t1], normalized_data[t2], num=num_nums)
    new_normalized = np.copy(normalized_data)
    new_normalized[t1:t2] = new_y

    return new_normalized, t1


def get_passivation_rate(time_stamp_array, troughless_data, pass_off_time, passivation_start):
    start_roi = time_stamp_array[passivation_start]
    roi = DewanPID_Utils.get_roi(start_roi, pass_off_time, time_stamp_array)
    x_vals = time_stamp_array[roi]
    y_vals = troughless_data[roi]
    passivation_auc = get_auc(x_vals, y_vals)
    passivation_time_delay = np.sum(np.diff(time_stamp_array[roi]))
    return passivation_auc, passivation_time_delay

    return passivation_auc

def get_depassivation_rate(time_stamp_array, troughless_data, pass_off_time, cutoff_percentages):
    off_index = np.where(time_stamp_array == pass_off_time)[0]
    max_value = troughless_data[off_index]
    cutoff_values = np.multiply(max_value, np.divide(cutoff_percentages, 100))

    auc_vals = []
    time_diffs = []

    for each in cutoff_values:
        roi = DewanPID_Utils.get_roi(each, max_value, troughless_data)
        time_diff = roi[-1] - off_index
        x_vals = time_stamp_array[roi]
        y_vals = troughless_data[roi]

        auc_vals.append(get_auc(x_vals, y_vals))
        time_diffs.append(time_diff[0])

    return auc_vals, time_diffs


def main():
    max_tube_length = 200
    min_concentration = 1
    sec_before = 1
    sec_after = 10
    cutoff_percentages = [5, 10, 25, 50, 75, 95]


    file_path, file_stem, file_folder = DewanPID_Utils.get_file()
    h5_file = DewanPID_Utils.open_h5_file(file_path)

    data = []

    sec_before = 1
    sec_after = 10

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_6_trials = np.where(trials['trialtype'] == 6)[0]

    odor_name = DewanPID_Utils.decode_list(trials['odor'][type_6_trials])
    odor_indexes = get_non_odor_trials(odor_name)

    type_6_trials = np.array(type_6_trials)[odor_indexes]

    odor_concentration = trials['odorconc'][type_6_trials]
    good_concentrations = np.where(odor_concentration > min_concentration)[0]
    odor_concentration = odor_concentration[good_concentrations]
    odor_name = odor_name[odor_indexes][good_concentrations]
    final_valve_on_time = trials['fvOnTime'][type_6_trials][good_concentrations]
    pid_gain = trials['PIDGain'][type_6_trials][good_concentrations]
    carrier_flowrate = trials['Carrier_flowrate'][type_6_trials][good_concentrations]
    diluter_flowrate = trials['Dilutor_flowrate'][type_6_trials][good_concentrations]
    pass_valve_off_time = trials['PassOffTime'][type_6_trials][good_concentrations]
    tube_length = trials['Tube'][type_6_trials][good_concentrations]

    good_tubes = np.where(tube_length < max_tube_length)[0]
    type_6_trials = type_6_trials[good_tubes]
    num_type_6_trials = len(good_tubes)

    fig, ax = plt.subplots()

    for i in range(num_type_6_trials):
        trial_number = type_6_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = DewanPID_Utils.get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = DewanPID_Utils.condense_packets(sniff_data,
                                                                             sniff_samples, packet_sent_time)
        final_valve_on = final_valve_on_time[i]
        pass_valve_off = (pass_valve_off_time[i] - final_valve_on) / 1000

        TOI_start = final_valve_on - sec_before * 1000
        TOI_end = final_valve_on + sec_after * 1000

        roi_index = DewanPID_Utils.get_roi(TOI_start, TOI_end, time_stamp_array)
        sniff_data_array = sniff_data_array[roi_index]
        time_stamp_array = time_stamp_array[roi_index]
        time_stamp_array = (time_stamp_array - final_valve_on) / 1000

        pass_off_index = np.where(time_stamp_array > pass_valve_off)[0][0]
        smoothed_data = smooth_data(sniff_data_array)
        normalized_data = normalize_data(smoothed_data)
        troughless_data, passivation_start = linear_extrapolation(normalized_data)

        # ax.axvline(x=time_stamp_array[t1])
        # ax.axvline(x=time_stamp_array[t2])
        passivation_auc, passivation_time_delay = get_passivation_rate(time_stamp_array, normalized_data,
                                                                       pass_valve_off, passivation_start)
        depassivation_aucs, depassivation_time_delays = get_depassivation_rate(time_stamp_array, normalized_data,
                                                                               pass_valve_off, cutoff_percentages)

        ax.plot(time_stamp_array, troughless_data)

        # ax.fill_between(x_component, y_component, color='r')

        result = [odor_name[i], odor_concentration[i], tube_length[i], diluter_flowrate[i], carrier_flowrate[i],
                  passivation_auc, passivation_time_delay]

        for z, each in enumerate(depassivation_time_delays):
            result.append(depassivation_aucs[z])
            result.append(each)

        data.append(result)

    h5_file.close()

    save_data(data, file_folder, file_stem, fig, cutoff_percentages)

    fig.show()



if __name__ == '__main__':
    main()
