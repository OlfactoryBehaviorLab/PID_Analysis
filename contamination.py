import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import HDFStore

def get_sniff_data(h5_file, trial_name):
    event_data = h5_file[trial_name + '/Events']
    sniff_data = h5_file[trial_name + '/sniff']

    return event_data, sniff_data

def open_h5_file(path: str):
    h5_file = h5py.File(path, 'r')
    return h5_file


def main():
    file_path = '.\ContaminationH5s\odorOctylamine_sess1_D2023_2_10T12_24_38.h5'
    h5_file = open_h5_file(file_path)

    num_sec_before_fv = 0.5
    num_sec_after_fv = 1.5

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_2_trials = np.where(trials['trialtype'] == 2)[0]

    num_type_2_trials = len(type_2_trials)

    final_valve_on_time = trials['fvOnTime'][type_2_trials]
    odor_concentration = trials['odorconc'][type_2_trials]
    pid_pump = trials['PIDPump'][type_2_trials]
    pid_spacer = trials['PIDSpace'][type_2_trials]
    pid_gain = trials['PIDGain'][type_2_trials]
    odor_vial = trials['odorvial'][type_2_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_2_trials]
    dilutor_flowrate = trials['Dilutor_flowrate'][type_2_trials]
    odor_name = trials['odor'][type_2_trials]


    for i in range(1):
        trial_number = type_2_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = get_sniff_data(h5_file, trial_name)
        sniff_data = [each for each in sniff_data]

        num_packets = len(sniff_data)

        sniff_samples = sniff_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']
        num_data_points = sum(sniff_samples)

        current_index = 0

        sniff_data_array = np.zeros((num_data_points, 1))
        time_stamp_array = np.copy(sniff_data_array)
        time_stamp_array_plot = np.copy(sniff_data_array)
        sniff_data_array_plot = np.copy(sniff_data_array)

        for packet in range(num_packets):
            current_packet = sniff_data[packet]
            packet_size = sniff_samples[packet]
            end_time = packet_sent_time[packet]
            sniff_data_array[current_index:current_index+packet_size] = current_packet
            time_stamp_array[current_index:current_index+packet_size] = np.arange((end_time-packet_size), (end_time+1))
            current_index += 1

    TOI_start = final_valve_on_time[i] - num_sec_before_fv * 1000
    TOI_end = final_valve_on_time[i] + num_sec_after_fv * 1000

    plot_sec_before_fv = 1
    plot_sec_after_fv = 4

    TOI_start = final_valve_on_time[i] - plot_sec_before_fv * 1000
    TOI_end = final_valve_on_time[i] + plot_sec_after_fv * 1000


    #h5_file.close()

if __name__ == "__main__":
    main()
