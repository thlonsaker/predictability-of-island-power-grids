import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from matplotlib import transforms
import os
import pickle


def true_intervals(bool_arr):
    """ Get intervals where bool_arr is true"""

    mask = np.concatenate([[True], ~bool_arr, [True]])
    interval_bounds = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1, 2)
    interval_sizes = interval_bounds[:, 1] - interval_bounds[:, 0]

    return interval_bounds, interval_sizes


def extreme_points(data, limit=(49, 51)):
    f_too_low = np.argwhere((data < limit[0]).values)[:, 0]
    f_too_high = np.argwhere((data > limit[1]).values)[:, 0]

    print('Number of too high frequency values: ', f_too_high.size,
          'Number of too low frequency values: ', f_too_low.size)

    return f_too_low, f_too_high


def extreme_inc(increments, limit=0.05):
    inc_too_high = np.argwhere((increments.abs() > limit).values)[:, 0]

    print('Number of too large increments: ', inc_too_high.size)

    return inc_too_high


def const_windows(increments, limit=60):
    wind_bounds, wind_sizes = true_intervals(increments.abs() < 1e-9)

    long_windows = [[]]
    long_window_bounds = wind_bounds[wind_sizes > limit]

    if long_window_bounds.size != 0:
        long_windows = np.hstack([np.r_[i:j] for i, j in long_window_bounds])

    print('Number of windows with constant frequency for longer than {}s: '.format(limit),
          long_window_bounds.shape[0])

    return wind_bounds, wind_sizes, long_windows, long_window_bounds


def nan_windows(data):
    wind_bounds, wind_sizes = true_intervals(data.isnull())

    print('Number of Nan-intervals: ', wind_sizes.shape[0])

    return wind_bounds, wind_sizes


def isolated_peaks(increments, limit=0.05):
    high_incs = increments.where(increments.abs() > limit)
    peak_locations = np.argwhere((high_incs * high_incs.shift(-1) < 0).values)[:, 0]

    print('Number of isolated peaks: ', peak_locations.size)

    return peak_locations


def correct_indices_grid(in_path, out_path, start, end, area, tz='CET'):
    """ Convert raw data to pandas Series with complete, (tz-localied time index - not done)."""

    data = pd.Series()
    new_data = pd.read_csv(in_path, sep=';', skiprows=1, usecols=[0, 1],
                           names=['time', 'frequency'], header=None, squeeze=True, parse_dates=[0])
    print('Processing the raw data...\r')
    # full_ind = pd.date_range(start=start, end=end, freq='S', tz=tz)
    # new_data = new_data.reindex(full_ind, fill_value=np.NaN)
    # data = data.append(new_data.values.astype('float64'))
    data = data.append(new_data.frequency)
    data.index = new_data.time
    # Convert timestamp to naive local time (removing tz-information)
    # data.index = data.index.tz_localize(None)
    print('Saving processed data...', end="\r")
    data.to_csv(out_path, float_format='%.6f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                'format_{}.csv'.format(area)}, header=False)


def plot_pdf(clean_data, data):
    nans = clean_data.isnull()
    idxs = clean_data.loc[nans].index
    data = data.drop(index=idxs)
    print('Plotting probability density function...', end="\r")
    ax = data.plot.kde(ind=np.arange(-250, 250, 0.5), logy=True, label='BalearicIslands', ylim=(5e-8, 5e-2), linewidth=2)
    plt.subplots_adjust(bottom=0.15)
    ax.set_xlabel('f - f$^{ref}$ (mHz)', fontsize=15)
    ax.set_ylabel('PDF', fontsize=15)
    ax.tick_params(labelsize=12)
    plt.grid(axis='x')
    plt.title('PDF of BalearicIslands frequency data', fontsize=15)
    print('plotting done.')
    plt.legend()
    plt.show()


def plot_increment_analysis(data, res=None):
    df = data.diff()
    std = df.std()
    print('Plotting increment analysis...', end="\r")
    fig, ax = plt.subplots()
    transform = transforms.offset_copy(ax.transData, fig=fig, x=0.0, y=50, units='points')
    x = np.arange(-15, 15, 0.01)
    plt.plot(x, scipy.stats.norm.pdf(x, 0, 1) / 100, label='gaussian')
    ax = sns.kdeplot(df / std, log_scale=(False, True), label='BalearicIslands, 1s', linestyle='None', marker='o',
                     markersize=3, transform=transform, gridsize=500)
    if res:
        df_r = data.diff(int(res))
        std_r = df_r.std()
        ax = sns.kdeplot(df_r / std_r, log_scale=(False, True), label='BalearicIslands, ' + str(res) + 's',
                         linestyle='None', marker='^', markersize=3)
    print('Plotting done.', end="\r")
    ax.set_xlabel('$\Delta$f/$\sigma$ (mHz)', fontsize=15)
    ax.set_ylabel('PDF $\Delta$f', fontsize=15)
    ax.set_ylim(10e-8, 10e2)
    ax.set_xlim(-15, 15)
    plt.title('Increment analysis', fontsize=15)
    plt.grid(axis='x')
    plt.legend()
    plt.show()


def load_data_to_pickle(path, path_out='', header=None):  # First run through format and clean data!
    data = pd.Series(dtype='float64')
    df = pd.read_csv(path, index_col=0, header=header, squeeze=True, parse_dates=[0])
    data = data.append(df)
    if len(path_out) != 0:
        data.to_pickle(path_out)
    else:
        data.to_pickle(f'{path[:-3]}pkl')
    print('Pickle created.')


def get_a_pickle(path):
    print('Loading data...', end="\r")
    data = pd.read_pickle(path)
    print('Loading of data done.', end="\r")
    return data


def read_data(pfile, files):
    if os.path.exists(pfile):
        with open(pfile, "rb") as fp:
            new_data = pickle.load(fp)
    else:
        data = []
        for file in files:
            print('Reading', file, '...', end='\r')
            data.append(pd.read_csv('Nordic/' + file, sep=',', header=None, skiprows=1, usecols=[0, 1], names=['time', 'frequency']))
        new_data = pd.concat(data, ignore_index=True)
        new_data.to_csv('cleaned_data/nordic_50.zip', float_format='%.6f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                        'nordic_50.csv'}, header=False)
        print('data saved as csv file.')
        # Convert dt_data to datetime and coerce parsing-errors into setting values to NaN
        ind = pd.to_datetime(new_data.time, errors='coerce')

        # If there are errors, try two other datetime formats that can occur in the data due to DST
        if ind.hasnans:
            print('HAS NANs!')
            try:
                mask = ind.isnull()
                ind_a = pd.to_datetime(new_data.time[mask], format='%Y/%m/%d %HA:%M:%S', errors='coerce').dropna()
                ind_b = pd.to_datetime(new_data.time[mask], format='%Y/%m/%d %HB:%M:%S', errors='coerce').dropna()
                ind[mask] = ind_a.append(ind_b).values
            except ValueError:
                print('There are unknown datetime formats in the data!')
        # new_data['frequency'] = (new_data['frequency'] * 1e-3) + 50
        # ind = ind.dt.tz_localize('CET')
        # new_data = new_data.set_index(ind).loc[:, 'frequency']
        # new_data = new_data[~new_data.index.duplicated()]

        # with open(pfile, "wb") as fp:
        #    pickle.dump(new_data, fp)
    return new_data

# how to remove all nan data points
# nans = clean_data.isnull()
# idxs = clean_data.loc[nans].index
# data = data.drop(index=idxs)
