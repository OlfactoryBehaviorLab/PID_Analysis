import numpy as np
import matplotlib.pyplot as plt

import Dewan_PID_Utils
from Dewan_Contamination_Utils import normalize_data, baseline_shift, smooth_data, save_data, get_non_odor_trials, \
    get_passivation_rate, get_depassivation_rate, combine_indexes, get_concentration_type_pairs, \
    parse_concentration_data, get_on_off_times



def main():
    # # # Configurables # # #
    max_tube_length = 1000
    min_concentration = 0.01
    # sec_before = 1
    # sec_after = 30
    bad_trials = [14]
    # # # Configurables # # #

    file_path, file_stem, file_folder = Dewan_PID_Utils.get_file()
    h5_file = Dewan_PID_Utils.open_h5_file(file_path)
    # Open our data file

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_6_trials = np.where(trials['trialtype'] == 6)[0]
    type_7_trials = np.where(trials['trialtype'] == 7)[0]
    odor_name = Dewan_PID_Utils.decode_list(trials['odor'])
    odor_concentration = trials['odorconc']
    tube_length = trials['Tube']

    good_odor_indexes = get_non_odor_trials(odor_name)  # Remove Blanks and MO if present
    good_concentration_indexes = np.where(odor_concentration > min_concentration)[0]
    good_tube_indexes = np.where(tube_length < max_tube_length)[0]

    good_trials = combine_indexes(type_6_trials, type_7_trials, good_odor_indexes, good_concentration_indexes,
                                  good_tube_indexes)

    carrier_flowrate = trials['Carrier_flowrate']
    diluter_flowrate = trials['Dilutor_flowrate']
    pass_valve_off_time = trials['PassOffTime']
    pass_valve_on_time = trials['PassOnTime']
    gain = trials['gain']

    pass_on_times, pass_off_times = get_on_off_times(trials)

    trial_data = []

    for each in good_trials:
        if each in bad_trials:
            pass

        differences = []

        trial_name = trial_names[each]
        event_data, sniff_data = Dewan_PID_Utils.get_sniff_data(h5_file, trial_name)
        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']
        sniff_data_array, time_stamp_array = Dewan_PID_Utils.condense_packets(sniff_data,
                                                                              sniff_samples, packet_sent_time)

        for pulse in range(len(pass_on_times)):

            pulse_start_time = pass_on_times[pulse]
            pulse_end_time = pass_off_times[pulse]
            roi_start = pulse_start_time - 300
            roi_end = pulse_end_time
            data_indexes = Dewan_PID_Utils.get_roi(pulse_start_time, pulse_end_time, time_stamp_array)

            pulse_data = sniff_data_array[data_indexes]
            time_stamp_array = time_stamp_array[data_indexes]

            baseline = np.mean(pulse_data[:300])
            pulse_mean = np.mean(pulse_data[300:])
            differences.append(pulse_mean - baseline)

        trial_data.append(differences)

if __name__ == '__main__':
    main()
