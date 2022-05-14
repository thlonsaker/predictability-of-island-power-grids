from help_funcs import *
if __name__ == '__main__':

    # Add path to processed data!
    path_to_data = '/Users/thorbjornlundonsaker/workspace/Master/format_data/Ireland/'
    # Add path to external (downloaded) data!
    path_to_external_data = '/Users/thorbjornlundonsaker/workspace/Master/format_data/Ireland/'

    in_path = path_to_external_data + 'IRL01.csv'
    out_path = path_to_data + 'Ireland.zip'

    correct_indices_grid(in_path, out_path, start='2021-11-04 16:10:01', end='2022-02-22 23:59:59',
                         area='Ireland', tz='GMT')

