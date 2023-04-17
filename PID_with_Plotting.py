import h5py
import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import sympy
plt.rcParams['figure.dpi'] = 600


def get_sniff_data(h5_file, trial_name):
    event_data = h5_file[trial_name + '/Events']
    sniff_data = h5_file[trial_name + '/sniff']
    sniff_data = [each for each in sniff_data]

    return event_data, sniff_data


def open_h5_file(path: str):
    try:
        h5_file = h5py.File(path, 'r')
    except OSError or FileNotFoundError:
        sg.popup_error("Unable to open file. Quitting!")
        quit()
    else:
        return h5_file


def decode_list(items):
    decoded = [item.decode('utf-8') for item in items]
    return decoded


def condense_packets(sniff_data, sniff_samples, packet_sent_time):
    sniff_data_array = []
    time_stamp_array = []

    num_packets = len(sniff_data)
    for packet in range(num_packets):
        current_packet = sniff_data[packet]
        packet_size = sniff_samples[packet]
        end_time = packet_sent_time[packet]
        sniff_data_array.extend(current_packet)
        time_stamp_addition = np.arange((end_time - packet_size + 1), end_time + 1)
        time_stamp_array.extend(time_stamp_addition)

    return np.array(sniff_data_array), np.array(time_stamp_array)


def get_roi(TOI_start, TOI_end, time_stamp_array):
    roi_index = np.where(np.logical_and(TOI_start < time_stamp_array, time_stamp_array < TOI_end))[0]
    return roi_index


def get_file():
    filename = sg.popup_get_file("Select H5 File...", file_types=(("H5 Files", "*.h5"), ("All Files", "*.*")))
    return filename


def save_csv(file_name_stem, data):
    column_labels = ['OdorConcentration', 'PIDPump', 'PIDGain', 'PeakPIDResponse', ' AveragePIDResponse', 'odorVial',
                     'Carrier_flowrate', 'Diluter_flowrate', 'PIDSpace', 'OdorName']
    output = pd.DataFrame(data, columns=column_labels)
    output.to_csv(f'.\\{file_name_stem}.csv', index=False)

def func(x, a, b, c):
    return a * np.power(x, -b) + c

def get_fit(x, y):

    fitted, pcov = curve_fit(func, x, y, maxfev=100000)

    modelPredictions = func(x, *fitted)

    absError = modelPredictions - y

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y))

    print('Parameters:', fitted)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)
    print(f'Formula: y = {fitted[0]} * x^{fitted[1]} + {fitted[2]}')

    return fitted[0], fitted[1], fitted[2]

def get_derivatives_log(x, y):
    logx = np.log(x)
    logy = np.log(y)

    #function = sympy.Function('y = a * x^-b + c')

    derivs = np.gradient(logy)/np.gradient(logx)
    #derivs = sympy.diff(function, x)

    return derivs

def main():

    #file_path = get_file()
    file_path = '.\\ContaminationH5s\\odorOctylamine_sess1_D2023_2_10T12_24_38.h5'
    file_name_stem = os.path.basename(file_path)[:-3]
    h5_file = open_h5_file(file_path)

    num_sec_before_fv = 0.5
    num_sec_after_fv = 1.5

    plot_sec_before_fv = 1
    plot_sec_after_fv = 6

    data = []

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_2_trials = np.where(trials['trialtype'] == 2)[0]
    num_type_2_trials = len(type_2_trials)

    final_valve_on_time = trials['fvOnTime'][type_2_trials]
    odor_concentration = trials['odorconc'][type_2_trials]
    pid_pump = decode_list(trials['PIDPump'][type_2_trials])
    pid_spacer = trials['PIDSpace'][type_2_trials]
    pid_gain = trials['PIDGain'][type_2_trials]
    odor_vial = trials['odorvial'][type_2_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_2_trials]
    dilutor_flowrate = trials['Dilutor_flowrate'][type_2_trials]
    odor_name = decode_list(trials['odor'][type_2_trials])

    fig, ax1 = plt.subplots()
    y_vals = []
    x_vals = []

    for i in [20]:
        trial_number = type_2_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = condense_packets(sniff_data, sniff_samples, packet_sent_time)

        time_stamp_array_plot = np.copy(time_stamp_array)
        sniff_data_array_plot = np.copy(sniff_data_array)

        TOI_start = final_valve_on_time[i] - num_sec_before_fv * 1000
        TOI_end = final_valve_on_time[i] + num_sec_after_fv * 1000

        TOI_start_plot = final_valve_on_time[i] - plot_sec_before_fv * 1000
        TOI_end_plot = final_valve_on_time[i] + plot_sec_after_fv * 1000

        roi_index = get_roi(TOI_start, TOI_end, time_stamp_array)
        sniff_data_array = sniff_data_array[roi_index]
        time_stamp_array = time_stamp_array[roi_index]
        end_baseline = int(num_sec_before_fv * 1000 - 100)
        baseline = np.mean(sniff_data_array[100:end_baseline])
        sniff_data_array = sniff_data_array - baseline

        plot_roi_index = get_roi(TOI_start_plot, TOI_end_plot, time_stamp_array_plot)
        sniff_data_array_plot = sniff_data_array_plot[plot_roi_index]
        time_stamp_array_plot = time_stamp_array_plot[plot_roi_index]
        base_ROI_plot = np.where(time_stamp_array_plot < (final_valve_on_time[i] - 50))[0]
        baseline_plot = np.mean(sniff_data_array_plot[base_ROI_plot])
        sniff_data_array_plot = sniff_data_array_plot - baseline_plot

        peak_PID_response = np.max(sniff_data_array)
        average_range_start = final_valve_on_time[i] + 500
        average_range_end = final_valve_on_time[i] + 1500
        average_ROI = get_roi(average_range_start, average_range_end, time_stamp_array)
        average_PID_response = np.mean(sniff_data_array[average_ROI])

        x_values = (time_stamp_array_plot - final_valve_on_time[i]) / 1000
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = sniff_data_array_plot / pid_gain[i]
        y_values = y_values / (carrier_flowrate[i] / 900)
        y_values = y_values * 4.8828
        y_vals.append(max(y_values))

        rising_curve_x = x_values[3050:3500]
        rising_curve_y = y_values[3050:3500]


        normalize_val = np.mean(y_values[2800:2900])

        rising_curve_y = rising_curve_y/normalize_val

        rising_curve_y = uniform_filter1d(rising_curve_y, size=25)

        a, b, c = get_fit(rising_curve_x, rising_curve_y)

        x_fitted = np.linspace(np.min(rising_curve_x), np.max(rising_curve_x), len(rising_curve_y))
        #y_fitted = a * np.log((rising_curve_x)- c) + b
        y_fitted = func(x_fitted, a, b, c)

        #you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....
        # Integrate from 2s -> 0.1% of max
        
        #slopes = get_derivatives_log(x_fitted, y_fitted)

        #slopes = a * np.power((x_fitted * b), (b-1))
        #print(slopes)
        #cutoff = np.where(np.abs(slopes) == np.min(np.absolute(slopes)))[0]
        #print(f'Cutoff: {cutoff}')

        fig2, ax2 = plt.subplots()
        ax2.plot(rising_curve_x,  rising_curve_y)
        ax2.plot(x_fitted, y_fitted)
        fig2.show()

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = [odor_concentration[i], pid_pump[i], pid_gain[i], peak_PID_response, average_PID_response,
                    odor_vial[i], carrier_flowrate[i], dilutor_flowrate[i], pid_spacer[i], odor_name[i]]

        data.append(row_data)

    h5_file.close()

    ax1.set_ylim([0, max(y_vals) + (max(y_vals) * 0.05)])
    ax1.set_xlim([min(x_vals), max(x_vals)])

    x_ticks = np.arange(round(min(x_vals)), round(max(x_vals)) + 1)

    plt.xticks(x_ticks)

    ax1.axvline(x=0.5, color='k')
    ax1.axvline(x=1.5, color='k')
    ax1.axvline(x=2, color='r')
    ax1.axvline(x=0, color='r')

    fig.show()

    save_csv(file_name_stem, data)


if __name__ == "__main__":
    main()
