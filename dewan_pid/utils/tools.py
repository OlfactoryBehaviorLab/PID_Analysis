from pathlib import Path
from tkinter import filedialog


def get_file(paths=None) -> list[dict]:
    return_paths = []

    if paths is None:
        file_paths = filedialog.askopenfilenames(
            title='Select a File', filetypes=[('MAT files', '*.mat'), ('All Files', '*.*')]
        )
    else:
        file_paths = paths

    if file_paths is None:
        raise FileNotFoundError('No file selected!')

    for each in file_paths:
        path_dict = {}

        path = Path(each)
        file_name = path.stem
        
        if 'Alg' in str(path):
            print(f'Skipping {path}')
            continue

        aIn_file_name = f'{file_name}_Alg'
        aIn_file_path = path.with_stem(aIn_file_name)

        if aIn_file_path.exists():
            path_dict['aIn'] = str(aIn_file_path)
        else:
            path_dict['aIn'] = None

        path_dict['stem'] = str(path.stem)
        path_dict['folder'] = str(path.parent)
        path_dict['path'] = str(path)
        # Get file stem to name output files; and file folder to save output files to

        return_paths.append(path_dict)

    return return_paths


def save_data(file_name_stem, file_folder, data, fig):

    excel_folder = Path(file_folder).joinpath('XLSX')
    figure_folder = Path(file_folder).joinpath('Figures')


    if not excel_folder.exists():
        excel_folder.mkdir(parents=True)
    if not figure_folder.exists():
        figure_folder.mkdir(parents=True)

    file_path = excel_folder.joinpath(f'{file_name_stem}.xlsx')
    fig_path = figure_folder.joinpath(f'{file_name_stem}.png')

    fig.savefig(fig_path, dpi=600)
    data.to_excel(file_path, index=False)


def save_figure(file_name_stem, file_folder, fig):
    fig_path = Path(file_folder, 'Figures', f'{file_name_stem}.pdf')
    fig.savefig(fig_path, transparent=True, dpi=300)
