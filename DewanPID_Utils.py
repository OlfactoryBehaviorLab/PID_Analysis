import os

import PySimpleGUI as sg
import h5py
import numpy as np


def get_sniff_data(h5_file, trial_name):
    event_data = h5_file[trial_name + '/Events']
    sniff_data = h5_file[trial_name + '/sniff'][:]
    #sniff_data = [each for each in sniff_data]

    return event_data, sniff_data


def open_h5_file(path: str):
    try:
        h5_file = h5py.File(path, 'r')
    except OSError or FileNotFoundError:
        sg.popup_error("Unable to open file. Quitting!")
        quit()
    else:
        return h5_file


def decode_list(items: list):
    decoded = [item.decode('utf-8') for item in items]
    return np.array(decoded)


def condense_packets(sniff_data, sniff_samples, packet_sent_time):
    first_time_point = packet_sent_time[0] - sniff_samples[0] + 1
    end_time_point = packet_sent_time[-1] + 1
    time_stamp_array = np.arange(first_time_point, end_time_point)
    sniff_data_array = np.concatenate(sniff_data).ravel()

    return sniff_data_array, time_stamp_array


def get_roi(ROI_Start: int, ROI_End: int, data_array):
    roi_index = np.where(np.logical_and(data_array > ROI_Start, data_array < ROI_End))[0]
    return roi_index


def get_file(path=None) -> (str, str):
    if path is None:
        filename = sg.popup_get_file("Select H5 File...", file_types=(("H5 Files", "*.h5"), ("All Files", "*.*")))
    else:
        filename = path
    file_stem = os.path.basename(filename)
    file_folder = os.path.dirname(filename)
    return filename, file_stem, file_folder



