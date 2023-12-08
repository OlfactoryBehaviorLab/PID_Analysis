import matplotlib.pyplot as plt
import numpy as np
from Utils import Dewan_PID_Utils
from tqdm import trange

plt.rcParams['figure.dpi'] = 600


def main():

    file_path, file_name_stem, file_folder = Dewan_PID_Utils.get_file()
    h5_file = Dewan_PID_Utils.open_h5_file(file_path)

    time_to_average = 1500  # Time in MS to average before the FV turns off
    extra_plot_end = 2000
    data = []

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_3_trials = np.where(trials['trialtype'] == 3)[0]  # CF is type 3
    type_9_trials = np.where(trials['trialtype'] == 9)[0]  # Pure check is type 9
    CF_trials = np.hstack((type_3_trials, type_9_trials))  # Stick em together
    num_trials = len(CF_trials)

    final_valve_on_time = trials['fvOnTime'][CF_trials]
    odor_concentration = trials['odorconc'][CF_trials]
    pid_pump = Dewan_PID_Utils.decode_list(trials['PIDPump'][CF_trials])
    pid_spacer = trials['PIDSpace'][CF_trials]
    pid_gain = trials['PIDGain'][CF_trials]
    odor_vial = trials['odorvial'][CF_trials]
    carrier_flowrate = trials['Carrier_flowrate'][CF_trials]
    dilutor_flowrate = trials['Dilutor_flowrate'][CF_trials]
    odor_name = Dewan_PID_Utils.decode_list(trials['odor'][CF_trials])
    odor_preduration = trials['odorpreduration'][CF_trials]
    odor_duration = trials['trialdur'][CF_trials]
    fig, ax1 = plt.subplots()
    y_vals = []
    x_vals = []

    for i in trange(num_trials):
        trial_number = CF_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = Dewan_PID_Utils.get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = Dewan_PID_Utils.condense_packets(sniff_data, sniff_samples,
                                                                              packet_sent_time)

        FV_on_time = final_valve_on_time[i]
        odor_dur = odor_duration[i]

        baseline_start = FV_on_time - 1100  # Baseline is from 1100-100ms before the FV
        baseline_end = FV_on_time - 100
        baseline_index = Dewan_PID_Utils.get_roi(baseline_start, baseline_end, time_stamp_array)

        baseline = np.mean(sniff_data_array[baseline_index])
        sniff_data_array = np.subtract(sniff_data_array, baseline)  # Baseline shift the data

        peak_PID_response = np.max(sniff_data_array)  # Get max response
        average_range_end = FV_on_time + odor_dur  # The end of the average will be the entire odor duration
        average_range_start = average_range_end - time_to_average  # We will start at the end, and backup a bit

        average_response_indexes = Dewan_PID_Utils.get_roi(average_range_start, average_range_end, time_stamp_array)
        average_PID_response = np.mean(sniff_data_array[average_response_indexes])  # Average the ROI

        plot_end = average_range_end + extra_plot_end
        # We basically will plot the same space, but with some buffer at the end
        plot_roi = Dewan_PID_Utils.get_roi(baseline_start, plot_end, time_stamp_array)

        x_values = (time_stamp_array[plot_roi] - final_valve_on_time[i]) / 1000  # Reference times to FV on
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = sniff_data_array[plot_roi] / pid_gain[i]
        # Scale data by gain
        y_values = y_values / (carrier_flowrate[i] / 900)
        # Scale data by flow dilution
        y_values = y_values * 4.8828
        # Convert bits -> volts
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = [odor_concentration[i], pid_pump[i], pid_gain[i], peak_PID_response, average_PID_response,
                    odor_vial[i], carrier_flowrate[i], dilutor_flowrate[i], odor_preduration[i], odor_duration[i],
                    pid_spacer[i], odor_name[i]]
        # Combine the data


        data.append(row_data)

    h5_file.close()

    ax1.set_ylim([0, max(y_vals) + (max(y_vals) * 0.05)])
    ax1.set_xlim([min(x_vals), max(x_vals)])

    x_ticks = np.arange(round(min(x_vals)), round(max(x_vals)) + 1)

    plt.xticks(x_ticks)

    average_line_end = odor_duration[-1] / 1000
    average_line_start = average_line_end - (time_to_average / 1000)

    ax1.axvline(x=(average_line_end+0.03), color='r')
    ax1.axvline(x=0, color='r')

    ax1.axvline(x=(average_line_end), color='k')
    ax1.axvline(x=average_line_start, color='k')

    Dewan_PID_Utils.save_data(file_name_stem, file_folder, data, fig)
    fig.show()


if __name__ == "__main__":
    main()
