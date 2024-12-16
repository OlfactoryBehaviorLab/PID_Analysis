import pathlib
import mat73
import scipy.io as sio
import pandas as pd
import numpy as np
import traceback

BASELINE_BYTE = 67
ODOR_BYTE = 83
FV_BYTE = 70
KIN_BYTE = 75
END_BYTE = 69
ITI_BYTE = 73

def parse_mat(path: pathlib.Path, aIn_path: pathlib.Path):
    mat_file = []
    aIn_file = []

    print(f'Parsing {path.name}')

    mat_file = load_mat(path)

    if aIn_path:
        aIn_file = load_mat(aIn_path)

        if not aIn_file:
            print(f'Error opening aIn matfile {aIn_path}')
            return None

    if not mat_file:
        print(f'Error opening matfile {path}')
        return None

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

    if not aIn_file:
        trial_data = parse_analog_data(session_data)
    else:
        trial_data = parse_aIn_analog_data(aIn_file)

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


def gather_trim_data(data, slices):
    _data = []
    for start, end in slices:
        _slice = data[start:end]
        _data.append(_slice)

    min_length = min([len(row) for row in _data])
    trimmed_data = [row[:min_length] for row in _data]

    return trimmed_data


def get_trial_sync_bytes(sync_bytes, sync_indices, end_offset: int = 2000):
    baseline_indices = np.where(sync_bytes == BASELINE_BYTE)[0] # Start of every trial
    ITI_events = np.where(sync_bytes == ITI_BYTE)[0]  # End of every trial

    sync_byte_bins = list(zip(baseline_indices, ITI_events))
    sync_bytes_per_trial = {}

    for trial_num, (start_index, end_index) in enumerate(sync_byte_bins):  # Loop through all trials

        _trial_bytes = sync_bytes[start_index:end_index+1]
        _sync_indices = sync_indices[start_index:end_index+1]

        if KIN_BYTE in _trial_bytes:  # We can skip all trials that are "subsampled"
            continue
        else:
            baseline_start_index = _sync_indices[np.where(_trial_bytes == BASELINE_BYTE)[0]][0]
            odor_start_index = _sync_indices[np.where(_trial_bytes == ODOR_BYTE)[0]][0]
            odor_end_index = _sync_indices[np.where(_trial_bytes == END_BYTE)[0]][0]
            ITI_start_index = _sync_indices[np.where(_trial_bytes == ITI_BYTE)[0]][0]
            ITI_end_index = ITI_start_index + end_offset

            trial_dict = {
                'baseline': [baseline_start_index, odor_start_index],
                'odor': [odor_start_index, odor_end_index],
                'end': [ITI_start_index, ITI_end_index]
            }

            sync_bytes_per_trial[str(trial_num)] = trial_dict

    return sync_bytes_per_trial


def parse_aIn_analog_data(aIn_file):
    sync_bytes = aIn_file['SyncEvents']
    sync_indices = aIn_file['SyncEventTimes']
    samples = aIn_file['Samples']

    sync_bytes_per_trial = get_trial_sync_bytes(sync_bytes, sync_indices)

    baseline_events = np.where(sync_bytes == 67)[0]
    FV_events = np.where(sync_bytes == 70)[0]
    ITI_events = np.where(sync_bytes == 73)[0]

    baseline_indices = sync_indices[baseline_events].astype(int)
    FV_indices = sync_indices[FV_events].astype(int)
    ITI_indices = sync_indices[ITI_events].astype(int)

    baseline_periods = list(zip(baseline_indices, FV_indices))
    odor_periods = list(zip(FV_indices, ITI_indices))
    end_periods = list(zip(ITI_indices[:-1], baseline_indices[1:]))
    # Offset both lists by one so we can handle the last ITI index separate
    final_end = tuple((ITI_indices[-1], len(samples)))
    # The final trial will be the last ITI index to the end of the data
    end_periods.append(final_end)

    baseline_data = []
    odor_data = []
    end_data = []

    trimmed_baseline_data = gather_trim_data(samples, baseline_periods)
    trimmed_odor_data = gather_trim_data(samples, odor_periods)
    trimmed_end_data = gather_trim_data(samples, end_periods)

    trial_data = {
        'baseline_volts': trimmed_baseline_data,
        'odor_volts': trimmed_odor_data,
        'end_volts': trimmed_end_data
    }

    trial_data_df = pd.DataFrame(trial_data)

    return trial_data_df


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

            print(f'{i}: {start_index - FV_index, end_index, iti_start_index}')

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
        except Exception:
            print(f'Error processing trial {i}')
            # print(traceback.format_exc())

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
    _all_sync_bytes = []

    for sync_byte in all_sync_bytes:
        if sync_byte.size < 1:
            _all_sync_bytes.append(0)
        else:
            _all_sync_bytes.append(sync_byte)

   # all_sync_bytes = [0 if sync_byte.size < 1 else sync_byte for sync_byte in all_sync_bytes]
    _all_sync_bytes = remove_double_sync_bytes(_all_sync_bytes)
    _all_sync_bytes = np.array(_all_sync_bytes)

    sync_bytes['baseline'] = np.where(_all_sync_bytes == 67)  # B(aseline)
    sync_bytes['start'] = np.where(_all_sync_bytes == 83)  # S(tart)
    sync_bytes['FV'] = np.where(_all_sync_bytes == 70)  # F(inal Valve)
    sync_bytes['end'] = np.where(_all_sync_bytes == 69)  # E(nd)
    sync_bytes['ITI'] = np.where(_all_sync_bytes == 73)  # I(TI)
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
    except NotImplementedError:
        mat_file = mat73.loadmat(str(path))
    except FileNotFoundError:
        print(traceback.format_exc())
        return mat_file
    except TypeError:
        print(traceback.format_exc())
        return mat_file
    except Exception:
        print(traceback.format_exc())
        return mat_file

    return mat_file
