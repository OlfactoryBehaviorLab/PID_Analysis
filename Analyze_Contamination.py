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


def save_data(data, file_path, file_stem, fig):
    column_labels = ['Odor Name', 'Odor Concentration', 'Tube Length', 'Dilutor', 'Carrier', 'Passivation',
                     'Depassivation']
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
    good_indexes = DewanPID_Utils.get_roi(10, 150, troughs)
    # Sometimes it erroneously makes the trough too early,
    # This makes sure it is always in the middle of the
    # passivation Time
    troughs = troughs[good_indexes]
    t1 = troughs[0] + 1000
    t2 = troughs[-1] + 1000
    return t1, t2


def spline_extrapolation(normalized_data):
    t1, t2 = get_troughs(normalized_data)

    x_vals = np.arange(0, len(normalized_data))
    new_xvals = np.arange(t1, t2)

    function = interpolate.UnivariateSpline(x_vals, normalized_data, k=1)
    new_y = function(new_xvals)

    # new_normalized = np.copy(normalized_data)
    normalized_data[t1:t2] = new_y

    return normalized_data, t1


def get_passivation_rate(time_stamp_array, troughless_data, pass_off_time, passivation_start):
    start_roi = time_stamp_array[passivation_start]
    roi = DewanPID_Utils.get_roi(start_roi, pass_off_time, time_stamp_array)
    x_vals = time_stamp_array[roi]
    y_vals = troughless_data[roi]
    passivation_auc = get_auc(x_vals, y_vals)

    return passivation_auc


def get_depassivation_rate(time_stamp_array, troughless_data, pass_off_time):
    off_index = np.where(time_stamp_array == pass_off_time)[0]
    max_value = troughless_data[off_index]
    cutoff_value = max_value * 0.95

    roi = DewanPID_Utils.get_roi(cutoff_value, max_value, troughless_data)

    x_vals = time_stamp_array[roi]
    y_vals = troughless_data[roi]

    depassivation_auc = get_auc(x_vals, y_vals)

    return depassivation_auc


def main():
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
    good_trials = np.where(odor_concentration > 3.5)[0]

    num_type_6_trials = len(good_trials)
    odor_concentration = odor_concentration[good_trials]
    odor_name = odor_name[odor_indexes][good_trials]
    final_valve_on_time = trials['fvOnTime'][type_6_trials][good_trials]
    pid_gain = trials['PIDGain'][type_6_trials][good_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_6_trials][good_trials]
    diluter_flowrate = trials['Dilutor_flowrate'][type_6_trials][good_trials]
    pass_valve_off_time = trials['PassOffTime'][type_6_trials][good_trials]
    tube_length = trials['Tube'][type_6_trials][good_trials]

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

        smoothed_data = smooth_data(sniff_data_array)
        normalized_data = normalize_data(smoothed_data)
        troughless_data, passivation_start = spline_extrapolation(normalized_data)

        passivation_auc = get_passivation_rate(time_stamp_array, troughless_data, pass_valve_off, passivation_start)
        depassivation_auc = get_depassivation_rate(time_stamp_array, troughless_data, pass_valve_off)

        ax.plot(time_stamp_array, troughless_data)

        # ax.fill_between(x_component, y_component, color='r')

        result = [odor_name[i], odor_concentration[i], tube_length[i], diluter_flowrate[i], carrier_flowrate[i],
                  passivation_auc, depassivation_auc]
        data.append(result)

    h5_file.close()

    save_data(data, file_folder, file_stem, fig)

    fig.show()



if __name__ == '__main__':
    main()
