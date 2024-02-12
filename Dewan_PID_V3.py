import matplotlib.pyplot as plt
import numpy as np
from Utils import Dewan_PID_Utils_V2, Dewan_MAT_Parser

plt.rcParams['figure.dpi'] = 600


def main():
    data = []
    y_vals = []
    x_vals = []

    time_to_average = 1500  # Time in MS to average before the FV turns off
    extra_plot_end = 2000

    file_path, file_name_stem, file_folder = Dewan_PID_Utils_V2.get_file()
    bpod_data = Dewan_MAT_Parser.parse_mat(file_path)

    experiment_type = bpod_data['experiment']['session_type']
    settings = bpod_data['settings']


    fig, ax1 = plt.subplots()


    for i, trial in enumerate(settings):


        FV_on_time = final_valve_on_time[i]
        odor_dur = odor_duration[i]

        baseline_start = FV_on_time - 1100
        baseline_end = FV_on_time - 100
        baseline_index = Dewan_PID_Utils.get_roi(baseline_start, baseline_end, time_stamp_array)

        baseline = np.mean(sniff_data_array[baseline_index])
        sniff_data_array = np.subtract(sniff_data_array, baseline)

        peak_PID_response = np.max(sniff_data_array)
        average_range_end = FV_on_time + odor_dur
        average_range_start = average_range_end - time_to_average

        average_response_indexes = Dewan_PID_Utils.get_roi(average_range_start, average_range_end, time_stamp_array)
        average_PID_response = np.mean(sniff_data_array[average_response_indexes])

        plot_end = average_range_end + extra_plot_end
        plot_roi = Dewan_PID_Utils.get_roi(baseline_start, plot_end, time_stamp_array)

        x_values = (time_stamp_array[plot_roi] - final_valve_on_time[i]) / 1000
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = sniff_data_array[plot_roi] / pid_gain[i]
        y_values = y_values / (carrier_flowrate[i] / 900)
        y_values = y_values * 4.8828
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = [odor_concentration[i], pid_pump[i], pid_gain[i], peak_PID_response, average_PID_response,
                    odor_vial[i], carrier_flowrate[i], dilutor_flowrate[i], odor_preduration[i], odor_duration[i],
                    pid_spacer[i], odor_name[i]]

        data.append(row_data)

    h5_file.close()

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
