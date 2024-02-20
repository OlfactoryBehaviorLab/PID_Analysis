import matplotlib.pyplot as plt
import numpy as np
from Utils import Dewan_PID_Utils_V2, Dewan_MAT_Parser
import pandas as pd

plt.rcParams['figure.dpi'] = 600


def main():
    PID_Data = []
    y_vals = []
    x_vals = []
    fig, ax1 = plt.subplots()

    try:
        file_path, file_name_stem, file_folder = Dewan_PID_Utils_V2.get_file()
    except FileNotFoundError as e:
        print(e)
        return

    bpod_data = Dewan_MAT_Parser.parse_mat(file_path)

    # experiment_type = bpod_data['experiment']['session_type']
    settings = bpod_data['settings']
    num_trials = len(settings)

    for i in range(num_trials):

        trial_settings = settings.iloc[i]
        odor_duration = trial_settings['odor_duration']
        gain_str = trial_settings['pid_gain'][1:]
        gain = np.double(gain_str)
        carrier_flowrate = trial_settings['carrier_MFC']

        trial_data = bpod_data['data'].iloc[i]

        baseline_data = trial_data['baseline_bits']
        avg_baseline_data = np.mean(trial_data['baseline_bits'])
        odor_data = trial_data['odor_bits']
        end_data = trial_data['end_bits']

        baseline_data_baseline_shift = np.subtract(baseline_data, avg_baseline_data)  # Yes, mostly zeros
        odor_data_baseline_shift = np.subtract(odor_data, avg_baseline_data)
        end_data_baseline_shift = np.subtract(end_data, avg_baseline_data)

        peak_PID_response = np.max(odor_data_baseline_shift)
        average_PID_response = np.mean(odor_data_baseline_shift)

        pre_trial_len = len(baseline_data_baseline_shift)
        trial_len = len(odor_data_baseline_shift)
        post_trial_time = len(end_data_baseline_shift)

        number_items = pre_trial_len + trial_len + post_trial_time

        x_values = np.linspace(-2, 4,  number_items)

        y_values = np.hstack((baseline_data_baseline_shift, odor_data_baseline_shift, end_data_baseline_shift))
        y_values = y_values / gain
        y_values = y_values / (carrier_flowrate / 900)
        y_values = y_values * 4.8828 # TODO: Find new value for Bpod setup
        y_vals.append(max(y_values))

        # you can fit a line to any dataset if you try hard enough. In this case, 1,000,000 times....
        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = np.hstack((peak_PID_response, average_PID_response))

        PID_Data.append(row_data)


    ax1.set_ylim([0, (max(y_vals) * 1.05)])
    ax1.set_xlim([-2.5, 4.5])

    #x_ticks = np.arange(round(min(x_vals)), round(max(x_vals)) + 1)
    x_ticks = np.arange(-2, 4)
    plt.xticks(x_ticks)

    PID_Data = pd.DataFrame(PID_Data, columns=['PID Peak', 'PID Avg'])
    combined_data = settings.join(PID_Data)

    Dewan_PID_Utils_V2.save_data(file_name_stem, file_folder, combined_data, fig)

if __name__ == "__main__":
    main()
