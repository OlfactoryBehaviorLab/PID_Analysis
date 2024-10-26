import pathlib
import scipy.io as sio
import pandas as pd
import numpy as np


def parse_mat(path: pathlib.Path):
    mat_file = load_mat(path)

    pid_data = {
        'settings': [],
        'bpod_info': [],
        'experiment': [],
        'data': [],
    }

    session_data = mat_file['SessionData'][0]
    trial_settings = parse_settings(session_data)
    session_info = parse_session_info(session_data)
    experiment_info = parse_experiment_info(session_data)
    trial_data = parse_analog_data(session_data)

    pid_data['settings'] = trial_settings
    pid_data['bpod_info'] = session_info
    pid_data['experiment'] = experiment_info
    pid_data['data'] = trial_data

    return pid_data


def parse_experiment_info(session_data) -> pd.DataFrame:
    exp_info = session_data['ExperimentParams'][0][0]
    exp_info = pd.DataFrame(exp_info)
    exp_info = exp_info.apply(lambda x: np.ravel(x[0]))

    return exp_info


def parse_settings(session_data) -> pd.DataFrame:
    settings = session_data['Settings'][0][0]  # Grab settings list that is deeply nested
    settings = pd.DataFrame(settings)  # Convert to dataframe
    settings = settings.map(lambda x: x[0])
    # Each data point is wrapped in a double array [[data]]; lambda function applied to each data point where the
    # double array is removed
    return settings


def parse_session_info(session_data) -> pd.DataFrame:
    info = session_data['Info'][0][0]
    info = pd.DataFrame(info)
    info = info.map(np.ravel)
    info = info.map(lambda x: x[0])

    firmware = info['Firmware']
    firmware = array_to_version_number(firmware)
    info['Firmware'] = firmware

    circuit_rev = info['CircuitRevision']
    circuit_rev = array_to_version_number(circuit_rev)
    info['CircuitRevision'] = circuit_rev

    return info


def parse_analog_data(session_data):
    analog_data_swap = session_data['analog_stream_swap'][0][0]
    analog_data_swap = preprocess_analog_swap(analog_data_swap)

    sync_bytes = get_sync_bytes(analog_data_swap)

    baseline_indices = sync_bytes['baseline'][0]
    FV_indices = sync_bytes['FV'][0]
    end_indices = sync_bytes['end'][0]
    iti_indices = sync_bytes['ITI'][0]

    trial_data = {  # Should really use more dicts; future Austin reports more dicts (10/25/24)
        'baseline_bits': [],
        'odor_bits': [],
        'baseline_volts': [],
        'odor_volts': [],
        'end_bits': [],
        'num_trials': 0
    }
    number_trials = len(FV_indices)

    for i in range(number_trials):
        try:
            start_index = baseline_indices[i]
            FV_index = FV_indices[i]
            end_index = end_indices[i]
            iti_start_index = iti_indices[i]

            if i == (number_trials - 1):
                iti_end_index = -1
            else:
                iti_end_index = baseline_indices[i + 1]

            baseline_data = analog_data_swap.iloc[start_index:FV_index]
            odor_data = analog_data_swap.iloc[FV_index:end_index]
            end_bits = analog_data_swap.iloc[iti_start_index:iti_end_index]

            baseline_data_bits = baseline_data['samples'].tolist()
            baseline_data_volts = baseline_data['samples_volts'].tolist()
            baseline_data_bits = np.hstack(baseline_data_bits)
            baseline_data_volts = np.hstack(baseline_data_volts)

            odor_data_bits = odor_data['samples'].tolist()
            odor_data_volts = odor_data['samples_volts'].tolist()
            odor_data_bits = np.hstack(odor_data_bits)
            odor_data_volts = np.hstack(odor_data_volts)

            end_bits = end_bits['samples'].tolist()
            end_bits = np.hstack(end_bits)

            trial_data['baseline_bits'].append(baseline_data_bits)
            trial_data['odor_bits'].append(odor_data_bits)

            trial_data['baseline_volts'].append(baseline_data_volts)
            trial_data['odor_volts'].append(odor_data_volts)

            trial_data['end_bits'].append(end_bits)
            trial_data['num_trials'] += 1
        except Exception as e:
            print(f'Error parsing trial {i}')

    trial_data = pd.DataFrame(trial_data)

    return trial_data


def preprocess_analog_swap(analog_data_swap):
    analog_data_swap = pd.DataFrame(analog_data_swap)
    analog_data_swap = analog_data_swap.apply(lambda x: np.ravel(x))
    columns_to_explode = list(analog_data_swap.keys().values[:2])
    analog_data_swap = analog_data_swap.explode(columns_to_explode)

    return analog_data_swap


def get_sync_bytes(analog_data_swap):
    sync_bytes = {
        'baseline': [],
        'start': [],
        'FV': [],
        'end': [],
        'ITI': []
    }

    all_sync_bytes = analog_data_swap['sync_indexes'].apply(lambda x: np.ravel(x))
    all_sync_bytes = [0 if sync_byte.size < 1 else sync_byte for sync_byte in all_sync_bytes]
    all_sync_bytes = remove_double_sync_bytes(all_sync_bytes)
    all_sync_bytes = np.array(all_sync_bytes)

    sync_bytes['baseline'] = np.where(all_sync_bytes == 67)  # B(aseline)
    sync_bytes['start'] = np.where(all_sync_bytes == 83)  # S(tart)
    sync_bytes['FV'] = np.where(all_sync_bytes == 70)  # F(inal Valve)
    sync_bytes['end'] = np.where(all_sync_bytes == 69)  # E(nd)
    sync_bytes['ITI'] = np.where(all_sync_bytes == 73)  # I(TI)
    lengths = [len(sync_bytes[each]) for each in sync_bytes.keys()]
    all_equal = np.array_equal(lengths, lengths)

    if not all_equal:
        raise "Error: unequal number of sync bytes, please repair sync data in MATLAB"

    sync_bytes_per_trial = pd.DataFrame(sync_bytes)

    return sync_bytes_per_trial


def remove_double_sync_bytes(all_sync_bytes):
    # Occasionally, two sync bytes will occupy one time point due to our polling frequency
    # We will take the first byte, and move it to the index n-1
    # While not perfectly accurate, our polling rate is significantly higher than the response rate of
    # the PID sensor, so this is not a concern

    for i, each in enumerate(all_sync_bytes):
        if isinstance(each, int):
            continue
        elif each.size == 1:
            all_sync_bytes[i] = each.item()
        else:
            all_sync_bytes[i - 1] = each[0]
            all_sync_bytes[i] = each[1]

    return all_sync_bytes


def array_to_version_number(array):
    array = array.apply(np.hstack).apply(np.ravel)[0]
    array = '.'.join(list(array.astype(str)))

    return array


def load_mat(path: pathlib.Path) -> dict:
    mat_file = []
    try:
        mat_file = sio.loadmat(str(path))
    except FileNotFoundError as fnfe:
        print(fnfe.strerror)
        print(f'File at path {path} does not exist!')
        return mat_file
    except TypeError as te:
        print(te.with_traceback)
        return mat_file

    return mat_file
