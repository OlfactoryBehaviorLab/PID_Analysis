from pathlib import Path
from tkinter import filedialog


def get_file(paths=None) -> list[tuple]:
    
    if paths is None:
        file_paths = filedialog.askopenfilenames(
            title='Select a File', filetypes=[('MAT files', '*.mat'), ('All Files', '*.*')]
        )
    else:
        file_paths = paths

    if file_paths is None:
        raise FileNotFoundError('No file selected!')
    
    return_paths = []

    for each in file_paths:
        path = Path(each)

        file_stem = str(path.stem)
        file_folder = str(path.parent)
        file_path = str(path)
        # Get file stem to name output files; and file folder to save output files to

        return_paths.append((file_path, file_stem, file_folder))

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
