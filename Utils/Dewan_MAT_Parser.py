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

    analog_data_swap = analog_data_swap.map(lambda x: np.ravel(x))

    start_indexes = analog_data_swap.index[analog_data_swap['sync_indexes'] == 83]
    FV_indexes = analog_data_swap.index[analog_data_swap['sync_indexes'] == 70]
    end_indexes = analog_data_swap.index[analog_data_swap['sync_indexes'] == 69]

    # TODO: Ask Josh how we prevent from loosing the occasional sync index when polling a_in

    pass

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

