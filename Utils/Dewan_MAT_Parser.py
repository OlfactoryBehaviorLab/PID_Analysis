import pathlib
import scipy.io as sio
import pandas as pd
import numpy as np

def parse_mat(path: pathlib.Path):
    mat_file = load_mat(path)

    session_data = mat_file['SessionData'][0]
    trial_settings = parse_settings(session_data)
    session_info = parse_session_info(session_data)


def parse_settings(session_data) -> pd.DataFrame:
    settings = session_data['Settings'][0][0]  # Grab settings list that is deeply nested
    settings = pd.DataFrame(settings)  # Convert to dataframe
    settings = settings.map(lambda x: x[0][0])
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
    analog_data_swap = pd.DataFrame(analog_data_swap)
    analog_data_swap = analog_data_swap.apply(lambda x: np.ravel(x))
    columns_to_explode = list(analog_data_swap.keys().values[:2])
    analog_data_swap = analog_data_swap.explode(columns_to_explode)

    sync_bytes = get_sync_bytes(analog_data_swap)



def get_sync_bytes(analog_data_swap):
    sync_bytes = analog_data_swap['sync_indexes'].apply(lambda x: np.ravel(x))

    trial_start_bytes = sync_bytes.index[sync_bytes == 83]
    FV_on_bytes = sync_bytes.index[sync_bytes == 70]
    trial_end_bytes = sync_bytes.index[sync_bytes == 69]

    sync_bytes_per_trial = pd.DataFrame(np.transpose([trial_start_bytes, FV_on_bytes, trial_end_bytes]),
                                        columns=['Start', 'FV', 'End'])
    sync_bytes_columns = sync_bytes_per_trial.columns
    lengths = []

    for each in sync_bytes_columns:
        lengths.append(len(sync_bytes_per_trial[each].values))

    all_equal = np.array_equal(lengths, lengths)

    if not all_equal:
        raise "Error: unequal number of sync bytes, please repair sync data in MATLAB"

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

