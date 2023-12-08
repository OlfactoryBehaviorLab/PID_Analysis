import matplotlib.pyplot as plt
import numpy as np
from Utils import Dewan_PID_Utils

plt.rcParams['figure.dpi'] = 600


def main():

    file_path, file_name_stem, file_folder = Dewan_PID_Utils.get_file()
    h5_file = Dewan_PID_Utils.open_h5_file(file_path)

    num_sec_before_fv = 1
    num_sec_after_fv = 2.5

    plot_sec_before_fv = 1
    plot_sec_after_fv = 6

    data = []

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_2_trials = np.where(trials['trialtype'] == 2)[0]
    num_type_2_trials = len(type_2_trials)

    final_valve_on_time = trials['fvOnTime'][type_2_trials]
    odor_concentration = trials['odorconc'][type_2_trials]
    pid_pump = Dewan_PID_Utils.decode_list(trials['PIDPump'][type_2_trials])
    pid_spacer = trials['PIDSpace'][type_2_trials]
    pid_gain = trials['PIDGain'][type_2_trials]
    odor_vial = trials['odorvial'][type_2_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_2_trials]
    dilutor_flowrate = trials['Dilutor_flowrate'][type_2_trials]
    odor_name = Dewan_PID_Utils.decode_list(trials['odor'][type_2_trials])

    fig, ax1 = plt.subplots()
    y_vals = []
    x_vals = []

    for i in range(num_type_2_trials):
        trial_number = type_2_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = Dewan_PID_Utils.get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = Dewan_PID_Utils.condense_packets(sniff_data, sniff_samples,
                                                                              packet_sent_time)

        time_stamp_array_plot = np.copy(time_stamp_array)
        sniff_data_array_plot = np.copy(sniff_data_array)

        TOI_start = final_valve_on_time[i] - num_sec_before_fv * 1000
        TOI_end = final_valve_on_time[i] + num_sec_after_fv * 1000

        TOI_start_plot = final_valve_on_time[i] - plot_sec_before_fv * 1000
        TOI_end_plot = final_valve_on_time[i] + plot_sec_after_fv * 1000

        roi_index = Dewan_PID_Utils.get_roi(TOI_start, TOI_end, time_stamp_array)
        sniff_data_array = sniff_data_array[roi_index]
        time_stamp_array = time_stamp_array[roi_index]
        end_baseline = int(num_sec_before_fv * 1000 - 100)
        baseline = np.mean(sniff_data_array[100:end_baseline])
        sniff_data_array = sniff_data_array - baseline

        plot_roi_index = Dewan_PID_Utils.get_roi(TOI_start_plot, TOI_end_plot, time_stamp_array_plot)
        sniff_data_array_plot = sniff_data_array_plot[plot_roi_index]
        time_stamp_array_plot = time_stamp_array_plot[plot_roi_index]
        base_ROI_plot = np.where(time_stamp_array_plot < (final_valve_on_time[i] - 50))[0]
        baseline_plot = np.mean(sniff_data_array_plot[base_ROI_plot])
        sniff_data_array_plot = sniff_data_array_plot - baseline_plot

        peak_PID_response = np.max(sniff_data_array)
        average_range_start = final_valve_on_time[i] + 500
        average_range_end = final_valve_on_time[i] + 1500
        average_ROI = Dewan_PID_Utils.get_roi(average_range_start, average_range_end, time_stamp_array)
        average_PID_response = np.mean(sniff_data_array[average_ROI])

        x_values = (time_stamp_array_plot - final_valve_on_time[i]) / 1000
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = sniff_data_array_plot / pid_gain[i]
        y_values = y_values / (carrier_flowrate[i] / 900)
        y_values = y_values * 4.8828
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....

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

    Dewan_PID_Utils.save_data(file_name_stem, file_folder, data, fig)
    fig.show()

if __name__ == "__main__":
    main()