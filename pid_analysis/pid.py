import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
import logging

from tqdm import trange
from pid_analysis.utils import tools, mat_parser

plt.set_loglevel (level = 'warning')

plt.rcParams['figure.dpi'] = 600
CF_ITI = 2  # The Trial ITI should always be 2s for a CF
TOTAL_PREDURATION = 2  # The total time before a trial will almost always be 2s before odor measurement

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def main():
    try:
        file_paths = tools.get_file()  # Get list of file(s)
    except FileNotFoundError:
        print(traceback.format_exc())
        return
    for file_container in file_paths:  # Loop through selected file(s)
        try:
            process_file(file_container)
        except Exception:
            print(traceback.format_exc())
            continue

    print('Done processing!')


def process_file(file_container):
    PID_Data = []
    y_vals = []
    x_vals = []
    volts = False
    fig, ax1 = plt.subplots()

    file_path = file_container['path']
    file_stem = file_container['stem']
    output_folder = file_container['folder']
    aIn_path = file_container['aIn']
    bpod_data = mat_parser.parse_mat(file_path, aIn_path)

    if aIn_path:
        volts = True

    if bpod_data is None:
        print(f'Error parsing mat file {file_path}')
        return

    experiment_params = bpod_data['experiment']
    experiment_type = experiment_params['session_type'][0]
    odor_name = experiment_params['odor'][0]
    experimenter_name = experiment_params['name'][0]
    num_trials = len(bpod_data['data'])
    settings = bpod_data['settings']
    trial_type = []

    print(f'Processing {experiment_type} for {odor_name} run by {experimenter_name}\n')
    for i in trange(num_trials):

        trial_settings = settings.iloc[i]
        trial_type = trial_settings['trial_type']
        gain_str = trial_settings['pid_gain'][1:]
        gain = np.double(gain_str)
        carrier_flowrate = trial_settings['carrier_MFC']
        solvent_duration = trial_settings['solvent_duration']

        trial_data = bpod_data['data'].iloc[i]
        if not volts:
            baseline_data = trial_data['baseline_bits']
            avg_baseline_data = np.mean(baseline_data)
            odor_data = trial_data['odor_bits']
            end_data = trial_data['end_bits']
        else:
            baseline_data = trial_data['baseline_volts']
            avg_baseline_data = np.mean(baseline_data)
            odor_data = trial_data['odor_volts']
            end_data = trial_data['end_volts']


        baseline_data_baseline_shift = np.subtract(baseline_data, avg_baseline_data)  # Yes, mostly zeros
        odor_data_baseline_shift = np.subtract(odor_data, avg_baseline_data)
        end_data_baseline_shift = np.subtract(end_data, avg_baseline_data)
        baseline_data_baseline_shift = baseline_data_baseline_shift[399:]
        # just grab 1 second

        peak_PID_response = np.max(odor_data_baseline_shift)
        average_PID_response = np.mean(odor_data_baseline_shift)

        pre_trial_len = len(baseline_data_baseline_shift)
        trial_len = len(odor_data_baseline_shift)
        post_trial_len = len(end_data_baseline_shift)

        if trial_type.lower() == "solvent":
            solvent1_data = trial_data["solvent1_volts"]
            solvent2_data = trial_data["solvent2_volts"]
            solvent1_data_baseline_shift = np.subtract(solvent1_data, avg_baseline_data)
            solvent2_data_baseline_shift = np.subtract(solvent2_data, avg_baseline_data)
            solvent1_data_len = len(solvent1_data_baseline_shift)
            solvent2_data_len = len(solvent2_data_baseline_shift)

            solvent1_duration_s = (solvent1_data_len // solvent_duration) * 1000
            # should indicate how many datapoints exist in 1s of data, as we only want the final second
            solvent1_data_baseline_shift_trimmed = solvent1_data_baseline_shift[solvent1_duration_s[0]:]

            trial_len = trial_len + len(solvent1_data_baseline_shift) + solvent2_data_len

            average_solvent1_response = np.mean(solvent1_data_baseline_shift_trimmed)
            peak_solvent1_response = np.max(solvent1_data_baseline_shift_trimmed)

            solvent2_part1 = solvent2_data_baseline_shift[:solvent2_data_len//2]
            solvent2_part2 = solvent2_data_baseline_shift[solvent2_data_len//2:]

            average_solvent2_p1_response = np.mean(solvent2_part1)
            peak_solvent2_p1_response = np.max(solvent2_part1)
            average_solvent2_p2_response = np.mean(solvent2_part2)
            peak_solvent2_p2_response = np.max(solvent2_part2)

        if post_trial_len > pre_trial_len:
            post_trial_len = pre_trial_len

        x_values = np.arange(-pre_trial_len, (trial_len + post_trial_len))

        if trial_type.lower() == "solvent":
            y_values = np.hstack((
                baseline_data_baseline_shift,
                solvent1_data_baseline_shift,
                odor_data_baseline_shift,
                solvent2_data_baseline_shift,
                end_data_baseline_shift[:post_trial_len],
            ))
        else:
            y_values = np.hstack((baseline_data_baseline_shift, odor_data_baseline_shift, end_data_baseline_shift[:post_trial_len]))

        y_values = y_values / gain
        y_values = y_values / (carrier_flowrate / 900)
        #y_values = y_values * 4.8828 # TODO: Find new value for Bpod setup

        y_vals.append(min(y_values))
        y_vals.append(max(y_values))
        x_vals.append(min(x_values))
        x_vals.append(max(x_values))

        ax1.plot(x_values, y_values, linewidth=0.5)

        row_data = np.hstack((peak_PID_response, average_PID_response))
        if trial_type.lower() == "solvent":
            row_data = np.hstack((
                row_data,
                average_solvent1_response, peak_solvent1_response,
                average_solvent2_p1_response, peak_solvent2_p1_response,
                average_solvent2_p2_response, peak_solvent2_p2_response,
            ))

        PID_Data.append(row_data)

    x_max = max(x_vals)
    x_min = min(x_vals)

    x_tick_min = round(x_min, -3)
    x_tick_max = round(x_max, -3)

    y_min = min(y_vals)
    y_max = max(y_vals)
    y_offset = abs(max(y_vals)) * 0.1
    y_max += y_offset

    if y_min == 0:
        y_min = -y_offset
    else:
        y_min -= y_offset

    if trial_type.lower() == "solvent":
        x_tick_labels = np.arange(-1, 8)
        x_tick_spacing = 9
    else:
        x_tick_labels = np.arange(-2, 5)
        x_tick_spacing = 7

    x_ticks = np.linspace(x_tick_min, x_tick_max, x_tick_spacing)
    ax1.set_xticks(x_ticks, labels = x_tick_labels)
    ax1.set_xlim([x_tick_min, x_tick_max])
    ax1.set_ylim([y_min , y_max])

    ax1.set_xlabel('Time since FV (s)')
    ax1.set_ylabel('Signal (Trial - Baseline)')
    plt.title(f'{odor_name}-{experiment_type}-{experimenter_name}')

    plt.tight_layout()

    if trial_type.lower() == "solvent":
        columns = [
                    'PID Peak', 'PID Avg',
                    'Solvent(pre odor) Peak', 'Solvent(pre odor) Avg',
                    'Solvent(post odor) 1 Peak', 'Solvent(post odor) 1 Avg',
                    'Solvent(post odor) 2 Peak', 'Solvent(post odor) 2 Avg'
                    ]
    else:
        columns = ["PID Peak", "PID Avg"]

    PID_Data = pd.DataFrame(PID_Data, columns=columns)
    combined_data = settings.join(PID_Data)

    tools.save_data(file_stem, output_folder, combined_data, fig)


if __name__ == "__main__":
    main()
