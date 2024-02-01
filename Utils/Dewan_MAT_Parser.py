import pathlib
import scipy.io as sio


def parse_mat(path: pathlib.Path) -> dict:
    mat_file = load_mat(path)



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
