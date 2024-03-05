import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import trange
from Utils import Dewan_PID_Utils_V2, Dewan_MAT_Parser


plt.rcParams['figure.dpi'] = 600
CF_ITI = 2 # The Trial ITI should always be 2s for a CF
TOTAL_PREDURATION = 2 # The total time before a trial will almost always be 2s before odor measurement

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

    experiment_params = bpod_data['experiment']
    experiment_type = experiment_params['session_type'][0]
    odor_name = experiment_params['odor'][0]
    experimenter_name = experiment_params['name'][0]

    settings = bpod_data['settings']
    num_trials = len(settings)

    print(f'Processing {experiment_type} for {odor_name} run by {experimenter_name}...')

    for i in trange(num_trials):

        trial_settings = settings.iloc[i]

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
        
        x_values = np.arange(-pre_trial_len, (trial_len + post_trial_time))

        y_values = np.hstack((baseline_data_baseline_shift, odor_data_baseline_shift, end_data_baseline_shift))
        y_values = y_values / gain
        y_values = y_values / (carrier_flowrate / 900)
        y_values = y_values * 4.8828 # TODO: Find new value for Bpod setup

        y_vals.append(max(y_values))
        x_vals.append(min(x_values))
        x_vals.append(max(x_values))

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = np.hstack((peak_PID_response, average_PID_response))

        PID_Data.append(row_data)


    x_max = max(x_values)
    x_min = min(x_values)

    x_tick_min = round(x_min, -3)
    x_tick_max = round(x_max, -3)

    x_ticks = np.linspace(x_tick_min, x_tick_max, 7)
    ax1.set_xticks(x_ticks, labels = np.arange(-2, 5))
    ax1.set_xlim([x_tick_min, x_tick_max * 1.05])
    ax1.set_ylim([-300, (max(y_vals) * 1.05)])

    ax1.set_xlabel('Time since FV (s)')
    ax1.set_ylabel('Signal (Trial - Baseline)')
    plt.title(f'{odor_name}-{experiment_type}-{experimenter_name}')

    plt.tight_layout()

    PID_Data = pd.DataFrame(PID_Data, columns=['PID Peak', 'PID Avg'])
    combined_data = settings.join(PID_Data)

    Dewan_PID_Utils_V2.save_data(file_name_stem, file_folder, combined_data, fig)

    print('Done processing!')

if __name__ == "__main__":
    main()
