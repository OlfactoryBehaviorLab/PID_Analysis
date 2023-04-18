import numpy as np
import pandas as pd
import PID_with_Plotting
import matplotlib.pyplot as plt
from sklearn import preprocessing

BIT_CONVERSION_FACTOR = 4.8828


def get_auc(x, y):
    trap = np.trapz(y, x)
    return trap


def normalize_data(sniff_data):
    #scaler = preprocessing.MinMaxScaler()
    #input_data = sniff_data.reshape(-1, 1)
    #normalized = scaler.fit_transform(input_data)
    #output_data = normalized.reshape(-1)
    normalize_factor = sniff_data[1800:1900]
    normalize_factor = np.mean(normalize_factor)
    output_data = sniff_data / normalize_factor

    return output_data


def bits_2_volts(bit_data, gain, carrier):
    gain_adjusted = bit_data / gain
    flow_dilution_adjusted = gain_adjusted / (carrier / 900)
    bits_to_volt = flow_dilution_adjusted * BIT_CONVERSION_FACTOR

    return bits_to_volt


def find_upper_bound(sniff_data, lower_bound):
    minimum = np.min(sniff_data[lower_bound:])
    cutoff = minimum * 1.01
    cutoff_indexes = np.where(sniff_data[lower_bound:] > cutoff)[0]
    #print(sniff_data[lower_bound:])
    upper_bound = cutoff_indexes[-1] + lower_bound

    return upper_bound


def find_lower_bound(time_stamps):
    end_time = 2
    region_indexes = np.where(time_stamps >= end_time)[0]
    lower_bound = region_indexes[0]
    return lower_bound


def save_csv(data, file_stem):
    column_labels = ['Odor Name', 'Odor Concentration', 'AUC']
    output = pd.DataFrame(data, columns=column_labels)
    output.to_csv(f'.\\{file_stem}-CONTAMINATION.csv', index=False)


def main():
    file_path, file_stem = PID_with_Plotting.get_file()
    h5_file = PID_with_Plotting.open_h5_file(file_path)

    data = []

    sec_before = 1
    sec_after = 10

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_2_trials = np.where(trials['trialtype'] == 2)[0]
    num_type_2_trials = len(type_2_trials)

    final_valve_on_time = trials['fvOnTime'][type_2_trials]
    odor_concentration = trials['odorconc'][type_2_trials]
    pid_gain = trials['PIDGain'][type_2_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_2_trials]
    odor_name = PID_with_Plotting.decode_list(trials['odor'][type_2_trials])

    for i in range(num_type_2_trials):
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

        normalized_data = normalize_data(sniff_data_array)

        sniff_data_array = bits_2_volts(normalized_data, pid_gain[i], carrier_flowrate[i])

        integral_lower_bound = find_lower_bound(time_stamp_array)
        integral_upper_bound = find_upper_bound(sniff_data_array, integral_lower_bound)
        print(integral_lower_bound, integral_upper_bound)

        y_component = sniff_data_array[integral_lower_bound:integral_upper_bound]
        x_component = time_stamp_array[integral_lower_bound:integral_upper_bound]
        fig, ax = plt.subplots()
        #fig2,ax2 = plt.subplots()
        #ax2.plot(time_stamp_array, sniff_data_array)
        ax.plot(time_stamp_array, sniff_data_array)
        ax.fill_between(x_component, y_component, color='red')
        fig.show()
        #fig2.show()
        AUC = get_auc(x_component, y_component)

        result = [odor_name[i], odor_concentration[i], AUC]

        data.append(result)

    h5_file.close()

    save_csv(data, file_stem)


if __name__ == '__main__':
    main()
