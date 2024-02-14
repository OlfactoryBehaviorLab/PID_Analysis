import matplotlib.pyplot as plt
import numpy as np
from Utils import Dewan_PID_Utils_V2, Dewan_MAT_Parser

plt.rcParams['figure.dpi'] = 600


def main():
    data = []
    y_vals = []
    x_vals = []
    fig, ax1 = plt.subplots()

    file_path, file_name_stem, file_folder = Dewan_PID_Utils_V2.get_file()
    bpod_data = Dewan_MAT_Parser.parse_mat(file_path)

    # experiment_type = bpod_data['experiment']['session_type']
    settings = bpod_data['settings']
    num_trials = len(settings)

    for i in range(num_trials):

        trial_settings = settings.iloc[i]
        odor_duration = trial_settings['odor_duration']
        gain = trial_settings['pid_gain']
        carrier_flowrate = trial_settings['carrier_MFC']

        trial_data = bpod_data['data'].iloc[i]

        baseline = np.mean(trial_data['baseline_bits'])
        odor_data = trial_data['odor_bits']
        odor_data_baseline_shift = np.subtract(odor_data, baseline)

        peak_PID_response = np.max(odor_data_baseline_shift)
        average_PID_response = np.mean(odor_data_baseline_shift)

        # plot_end = average_range_end + extra_plot_end
        # plot_roi = Dewan_PID_Utils.get_roi(baseline_start, plot_end, time_stamp_array)
        num_data_points = len(odor_data_baseline_shift)
        x_values = np.linspace(-2, settings.iloc[0]['odor_duration']/1000, num_data_points)

        #x_values = (time_stamp_array[plot_roi] - final_valve_on_time[i]) / 1000
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = odor_data_baseline_shift / gain
        y_values = y_values / (carrier_flowrate / 900)
        y_values = y_values * 4.8828
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = [odor_concentration[i], pid_pump[i], pid_gain[i], peak_PID_response, average_PID_response,
                    odor_vial[i], carrier_flowrate[i], dilutor_flowrate[i], odor_preduration[i], odor_duration[i],
                    pid_spacer[i], odor_name[i]]

        data.append(row_data)


    ax1.set_ylim([0, max(y_vals) + (max(y_vals) * 0.05)])
    ax1.set_xlim([min(x_vals), max(x_vals)])

    x_ticks = np.arange(round(min(x_vals)), round(max(x_vals)) + 1)

    plt.xticks(x_ticks)

    average_line_end = odor_duration[-1] / 1000
    average_line_start = average_line_end - (time_to_average / 1000)

    ax1.axvline(x=(average_line_end + 0.03), color='r')
    ax1.axvline(x=0, color='r')

    ax1.axvline(x=(average_line_end), color='k')
    ax1.axvline(x=average_line_start, color='k')

    Dewan_PID_Utils.save_data(file_name_stem, file_folder, data, fig)
    fig.show()

if __name__ == "__main__":
    main()
