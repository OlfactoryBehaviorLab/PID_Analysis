import numpy as np

from Utils import Dewan_PID_Utils
from Dewan_Contamination_Utils import get_non_odor_trials, \
    combine_indexes

from Utils.Dewan_Contamination_Utils_V2 import get_on_off_times, plot_pulse_differences, fit_function



def main():
    # # # Configurables # # #
    max_tube_length = 1000
    min_concentration = 0.01
    baseline_length = 300  # in msec
    post_time = 0
    bad_trials = []
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
    trial_duration = trials['trialdur']
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
            roi_start = pulse_start_time - baseline_length
            roi_end = pulse_end_time + post_time
            data_indexes = Dewan_PID_Utils.get_roi(roi_start, roi_end, time_stamp_array)

            pulse_data = sniff_data_array[data_indexes]
            time_stamp_array = time_stamp_array[data_indexes]

            baseline = np.mean(pulse_data[:300])
            pulse_mean = np.mean(pulse_data[300:])
            differences.append(pulse_mean - baseline)

        trial_data.append(differences)


    for each in trial_data:
        coefficients, _ = fit_function(each)
        plot_pulse_differences(each, trial_duration[0], coefficients)


if __name__ == '__main__':
    main()
