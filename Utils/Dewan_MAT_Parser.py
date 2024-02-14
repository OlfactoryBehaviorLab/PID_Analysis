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
    settings = settings.applymap(lambda x: x[0][0])
    # Each data point is wrapped in a double array [[data]]; lambda function applied to each data point where the
    # double array is removed
    return settings


def parse_session_info(session_data) -> pd.DataFrame:
    info = session_data['Info'][0][0]
    info = pd.DataFrame(info)
    info = info.applymap(np.ravel)
    info = info.applymap(lambda x: x[0])

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

    trial_data = {  # Should really use more dicts
        'baseline_bits': [],
        'odor_bits': [],
        'baseline_volts': [],
        'odor_volts': [],
    }

    for each in sync_bytes.index.tolist():
        indices = sync_bytes.iloc[each]

        baseline_indices = indices['baseline']

        if len(baseline_indices) > 0:  # If there is somehow no baseline indices, use the odor start index
            start_indices = baseline_indices
        else:
            start_indices = indices['start']

        FV_indices = indices['FV']
        end_indices = indices['end']

        baseline_data = analog_data_swap.iloc[start_indices:FV_indices]
        odor_data = analog_data_swap.iloc[FV_indices:end_indices]

        baseline_data_bits = baseline_data['samples'].tolist()
        baseline_data_volts = baseline_data['samples_volts'].tolist()
        baseline_data_bits = np.hstack(baseline_data_bits)
        baseline_data_volts = np.hstack(baseline_data_volts)

        odor_data_bits = odor_data['samples'].tolist()
        odor_data_volts = odor_data['samples_volts'].tolist()
        odor_data_bits = np.hstack(odor_data_bits)
        odor_data_volts = np.hstack(odor_data_volts)

        trial_data['baseline_bits'].append(baseline_data_bits)
        trial_data['odor_bits'].append(odor_data_bits)

        trial_data['baseline_volts'].append(baseline_data_volts)
        trial_data['odor_volts'].append(odor_data_volts)

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
        'end': []
    }

    all_sync_bytes = analog_data_swap['sync_indexes'].apply(lambda x: np.ravel(x))

    sync_bytes['baseline'] = all_sync_bytes.index[all_sync_bytes == 63]   # B(aseline)
    sync_bytes['start'] = all_sync_bytes.index[all_sync_bytes == 83]      # S(tart)
    sync_bytes['FV'] = all_sync_bytes.index[all_sync_bytes == 70]            # F(inal Valve)
    sync_bytes['end'] = all_sync_bytes.index[all_sync_bytes == 69]        # E(nd)

    lengths = [len(sync_bytes[each]) for each in sync_bytes.keys()]
    all_equal = np.array_equal(lengths, lengths)

    if not all_equal:
        raise "Error: unequal number of sync bytes, please repair sync data in MATLAB"

    sync_bytes_per_trial = pd.DataFrame(sync_bytes)

    return sync_bytes_per_trial


def array_to_version_number(array):
    array = array.apply(np.hstack).apply(np.ravel)[0]
    array = '.'.join(list(array.astype(str)))

    return array


def load_mat(path: pathlib.Path) -> dict:
    mat_file = []
    try:
        mat_file = sio.loadmat(str(path))
    except FileNotFoundError as e:
        print(e.strerror)
        print(f'File at path {path} does not exist!')
        return mat_file
    except TypeError as e2:
        print(e2.with_traceback)
        return mat_file

    return mat_file
