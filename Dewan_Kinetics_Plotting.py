import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import Dewan_PID_Utils
from tqdm import trange

plt.rcParams['figure.dpi'] = 600
plt.rcParams['pdf.fonttype'] = 42


def main():

    # # # Graph Configurables # # #
    LINE_COLOR = 'grey'
    BOX_COLOR = 'royalblue'
    BOX_TRANSPARENCY = 0.25
    plot_sec_before_fv = 1
    plot_sec_after_fv = 6
    # # # Graph Configurables # # #

    file_path, file_name_stem, file_folder = Dewan_PID_Utils.get_file()
    h5_file = Dewan_PID_Utils.open_h5_file(file_path)

    trials = h5_file['/Trials']
    trial_names = list(h5_file.keys())
    type_1_trials = np.where(trials['trialtype'] == 1)[0]
    num_type_1_trials = len(type_1_trials)

    final_valve_on_time = trials['fvOnTime'][type_1_trials]
    pid_gain = trials['PIDGain'][type_1_trials]
    carrier_flowrate = trials['Carrier_flowrate'][type_1_trials]

    fig, ax1 = plt.subplots()
    y_vals = []
    x_vals = []

    for i in trange(num_type_1_trials):
        trial_number = type_1_trials[i]
        trial_name = trial_names[trial_number]

        event_data, sniff_data = Dewan_PID_Utils.get_sniff_data(h5_file, trial_name)

        sniff_samples = event_data['sniff_samples']
        packet_sent_time = event_data['packet_sent_time']

        sniff_data_array, time_stamp_array = Dewan_PID_Utils.condense_packets(sniff_data, sniff_samples,
                                                                              packet_sent_time)

        time_stamp_array_plot = np.copy(time_stamp_array)
        sniff_data_array_plot = np.copy(sniff_data_array)

        TOI_start_plot = final_valve_on_time[i] - plot_sec_before_fv * 1000
        TOI_end_plot = final_valve_on_time[i] + plot_sec_after_fv * 1000

        plot_roi_index = Dewan_PID_Utils.get_roi(TOI_start_plot, TOI_end_plot, time_stamp_array_plot)
        sniff_data_array_plot = sniff_data_array_plot[plot_roi_index]
        time_stamp_array_plot = time_stamp_array_plot[plot_roi_index]
        base_ROI_plot = np.where(time_stamp_array_plot < (final_valve_on_time[i] - 50))[0]
        baseline_plot = np.mean(sniff_data_array_plot[base_ROI_plot])
        sniff_data_array_plot = sniff_data_array_plot - baseline_plot

        x_values = (time_stamp_array_plot - final_valve_on_time[i]) / 1000
        x_vals.append(max(x_values))
        x_vals.append(min(x_values))

        y_values = sniff_data_array_plot / pid_gain[i]
        y_values = y_values / (carrier_flowrate[i] / 900)
        y_values = y_values * 4.8828
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....
        ax1.plot(x_values, y_values, linewidth=0.5, color=LINE_COLOR)

    h5_file.close()
    ax1.set_ylim([0, max(y_vals) + (max(y_vals) * 0.05)])
    ax1.set_xlim([-1, 4])
    ax1.get_yaxis().set_visible(False)
    rect = patches.Rectangle((0, 0), 2, (max(y_vals)*1.05), color=BOX_COLOR, alpha=BOX_TRANSPARENCY)
    ax1.add_patch(rect)
    ax1.set_xlabel("Time (s)", fontfamily='arial', fontsize=12, fontweight='bold')
    x_ticks = np.arange(-1, 5)

    plt.xticks(x_ticks)

    Dewan_PID_Utils.save_figure(file_name_stem, file_folder, fig)
    fig.show()


if __name__ == "__main__":
    main()
