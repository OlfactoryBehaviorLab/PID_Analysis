import numpy as np
import pandas as pd
import PID_with_Plotting
import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

BIT_CONVERSION_FACTOR = 4.8828

# TODO Smooth Data -> Moving Window 5-10
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

def smooth_data(sniff_data, window_size=10, mode='reflect'):
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

def save_csv(data, file_stem):
    column_labels = ['Odor Name', 'Odor Concentration', 'AUC', 'Time Delay']
    output = pd.DataFrame(data, columns=column_labels)
    output.to_csv(f'.\\{file_stem}-CONTAMINATION.csv', index=False)

def are_keys_in_string(keys, string):
    return any(key in string for key in keys)

def get_non_odor_trials(odor_names):
    keys = ['blank', 'mo', 'h2o']
    indexes = [are_keys_in_string(keys, str.lower(item)) for item in odor_names]
    indexes = np.logical_not(indexes)

    return indexes

def main():
    file_path, file_stem = PID_with_Plotting.get_file()
    h5_file = PID_with_Plotting.open_h5_file(file_path)

    data = []

    sec_before = 1
    sec_after = 10

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_2_trials = np.where(trials['trialtype'] == 2)[0]

    odor_name = PID_with_Plotting.decode_list(trials['odor'][type_2_trials])
    odor_indexes = get_non_odor_trials(odor_name)

    type_2_trials = np.array(type_2_trials)[odor_indexes]
    num_type_2_trials = len(type_2_trials)

    odor_name = odor_name[odor_indexes]
    final_valve_on_time = trials['fvOnTime'][type_2_trials]
    odor_concentration = trials['odorconc'][type_2_trials]
    pid_gain = trials['PIDGain'][type_2_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_2_trials]
    fig, ax = plt.subplots()

    for i in tqdm.trange(num_type_2_trials):
        trial_number = type_2_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = PID_with_Plotting.get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = PID_with_Plotting.condense_packets(sniff_data,
                                                                                sniff_samples, packet_sent_time)
        TOI_start = final_valve_on_time[i] - sec_before * 1000
        TOI_end = final_valve_on_time[i] + sec_after * 1000

        roi_index = PID_with_Plotting.get_roi(TOI_start, TOI_end, time_stamp_array)
        sniff_data_array = sniff_data_array[roi_index]
        time_stamp_array = time_stamp_array[roi_index]
        time_stamp_array = (time_stamp_array - final_valve_on_time[i]) / 1000
        end_baseline = int(sec_before * 1000 - 100)
        baseline = np.mean(sniff_data_array[100:end_baseline])
        sniff_data_array = sniff_data_array - baseline

        sniff_data_array = bits_2_volts(sniff_data_array, pid_gain[i], carrier_flowrate[i])

        smoothed_data = smooth_data(sniff_data_array)

        normalized_data = normalize_data(smoothed_data)

        integral_lower_bound = find_lower_bound(time_stamp_array)
        integral_upper_bound = find_upper_bound(normalized_data, integral_lower_bound)

        y_component = normalized_data[integral_lower_bound:integral_upper_bound]
        x_component = time_stamp_array[integral_lower_bound:integral_upper_bound]

        ax.plot(time_stamp_array, normalized_data)
        #ax.fill_between(x_component, y_component, color='r')

        time_delay = integral_upper_bound - integral_lower_bound

        AUC = get_auc(x_component, y_component)

        result = [odor_name[i], odor_concentration[i], AUC, time_delay]

        data.append(result)

    h5_file.close()

    fig.show()

    save_csv(data, file_stem)


if __name__ == '__main__':
    main()
