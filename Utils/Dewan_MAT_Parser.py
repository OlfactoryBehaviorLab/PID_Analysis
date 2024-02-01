import pathlib
import scipy.io as sio
import pandas as pd
# import numpy as np

def parse_mat(path: pathlib.Path) -> pd.DataFrame:
    mat_file = load_mat(path)

    session_data = mat_file['SessionData'][0]
    trial_settings = parse_settings(session_data)

def load_mat(path: pathlib.Path) -> dict:
    mat_file = []
    try:
        mat_file = sio.loadmat(str(path))
    except FileNotFoundError as e:
        print(e.strerror)
        print(f'File at path {path} does not exist!')
        return mat_file
    except TypeError as e2:
        print(e2.strerror)
        return mat_file

    return mat_file


def parse_settings(session_data) -> pd.DataFrame:
    settings = session_data['Settings'][0][0]  # Grab settings list that is deeply nested
    settings = pd.DataFrame(settings)  # Convert to dataframe
    settings = settings.map(lambda x: x[0][0])
    # Each data point is wrapped in a double array [[data]]; lambda function applied to each data point where the
    # double array is removed
    return settings

