import h5py
import os
import pandas as pd
import numpy as np
import PySimpleGUI as sg


def get_sniff_data(h5_file, trial_name):
    event_data = h5_file[f'{trial_name}/Events']
    sniff_data = h5_file[f'{trial_name}/sniff'][:]
    # Pull data out of h5 file by index

    return event_data, sniff_data


def open_h5_file(path: str):
    try:
        h5_file = h5py.File(path, 'r')
    except (OSError, FileNotFoundError):
        sg.popup_error("Unable to open file. Quitting!")
        quit()
    else:
        return h5_file
    # Try to open the h5 file, if you cannot, error out and close the program


def decode_list(items: list):
    decoded = [item.decode('utf-8') for item in items]
    return np.array(decoded)
    # utf-8 decode each item in list


def condense_packets(sniff_data, sniff_samples, packet_sent_time):
    first_time_point = packet_sent_time[0] - sniff_samples[0] + 1
    # Packet_sent_time it the time of the last packet, subtract the first sniff_sample length to get the true start time
    end_time_point = packet_sent_time[-1] + 1
    # End point is infacte last time point, since it's the end of the last packet
    time_stamp_array = np.arange(first_time_point, end_time_point)
    # Generate a list for the time_stamp_array,
    sniff_data_array = np.concatenate(sniff_data).ravel()
    # Combine all the sniff_data into a single list

    return sniff_data_array, time_stamp_array


def get_roi(ROI_Start: int, ROI_End: int, data_array):
    roi_index = np.where(np.logical_and(data_array > ROI_Start, data_array < ROI_End))[0]
    # Find all indexes between two values
    return roi_index


def get_file(path=None) -> (str, str, str):
    if path is None:
        filename = sg.popup_get_file("Select H5 File...", file_types=(("H5 Files", "*.h5"), ("All Files", "*.*")))
    else:
        filename = path
    file_stem = os.path.basename(filename)
    file_folder = os.path.dirname(filename)
    # Get file stem to name output files; and file folder to save output files to
    return filename, file_stem, file_folder


def save_data(file_name_stem, file_folder, data, fig):
    column_labels = ['OdorConcentration', 'PIDPump', 'PIDGain', 'PeakPIDResponse', ' AveragePIDResponse', 'odorVial',
                     'Carrier_flowrate', 'Diluter_flowrate', 'PIDSpace', 'OdorName']
    output = pd.DataFrame(data, columns=column_labels)
    file_path = os.path.join(file_folder, 'CSV', f'{file_name_stem}.csv')
    fig_path = os.path.join(file_folder, 'Figures', f'{file_name_stem}.png')
    fig.savefig(fig_path, dpi=600)
    output.to_csv(file_path, index=False)


def save_figure(file_name_stem, file_folder, fig):
    fig_path = os.path.join(file_folder, 'Figures', f'{file_name_stem}.pdf')
    fig.savefig(fig_path, dpi=300)
