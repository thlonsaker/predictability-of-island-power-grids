import numpy as np
import pandas as pd
from standards import Standards
from help_funcs import get_a_pickle
import pickle
import datetime
import glob
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from pattern_prediction_test import construct_chunks


def new_period_info(sample, week, month, file, test_file=None):
    period_length = define_period(week, month)
    train_starts, train_ends = file['Period start'], file['Period end']
    train_start, train_end = train_starts[sample - 1], train_ends[sample - 1]
    if test_file is None:
        test_start = train_end + DateOffset(seconds=1)
        test_end = train_start + DateOffset(seconds=(period_length - 1))
    else:
        test_start, test_end = test_file['Period start'], test_file['Period end']
        test_start, test_end = pd.to_datetime(test_start), pd.to_datetime(test_end)
    return {'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end}


def show_period(data, hour=False, drop_nans=False, show=False, chunks=False):
    """Get the period length of the different data sets with/without nans."""
    if isinstance(data, str):
        region = underscore(data)
        data = pd.read_pickle(f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{region}/{region}_50.pkl')
    s, e = str(data.index[0]), str(data.index[-1] + pd.DateOffset(seconds=1))
    s_year, s_month, s_day, s_hour = s[:4], s[5:7], s[8:10], s[11:13]
    e_year, e_month, e_day, e_hour = e[:4], e[5:7], e[8:10], e[11:13]
    start = datetime.datetime(int(s_year), int(s_month), int(s_day), int(s_hour))
    end = datetime.datetime(int(e_year), int(e_month), int(e_day), int(e_hour))
    time = end - start
    days = time.days
    quality = 0
    if drop_nans:
        if chunks:
            data = construct_chunks(data, 3600)
        quality = data.dropna().shape[0] / data.shape[0]
        days = int(round(days * quality))
    if show:
        print(f'Days: {days}' + f'\nnans: {round((1 - quality) * 100, 2)}%' * drop_nans)
    if hour:
        hours = (time.seconds // 3600) % days
        return f'{days}d {hours}h'
    return str(days) + ' days'


def define_period(weeks, months, points_to_predict=3600):
    w = points_to_predict * 24 * 7 * weeks
    m = points_to_predict * 24 * 30 * months
    if (w + m) != 0:
        return int(w + m)
    return -1


def chunks_quality(data, limit, q=False):
    quality = 0
    if not isinstance(data, list):
        data = [data]
    for df in data:
        quality = df.dropna().shape[0] / df.shape[0]
        if quality < limit:
            if q:
                return False, quality
            return False
    if q:
        return True, quality
    return True


def space(word):
    return word.replace('_', ' ')


def underscore(word):
    return word.replace(' ', '_')


def period_filename(times):
    week, month = times
    period = f'{month}_months'
    if week > month:
        if week > 1:
            period = f'{week}_weeks'
        else:
            period = f'{week}_week'
    else:
        if month == 1:
            period = f'{month}_month'
    return week, month, period


def frequency_analysis_iceland(region, sec=10, limit=99.5, above=50.2, under=49.8):
    """Check whether Iceland achieve its frequency
    goals for this frequency time series."""

    print(f'analyzing frequency time series from {region.name}...', end='\r')
    data = pd.read_pickle(region.data_path).dropna()
    avg = data.rolling(sec, min_periods=1).mean()
    avg = avg[avg.values < above]
    avg = avg[avg.values > under]
    p = avg.shape[0] / data.shape[0]
    print(round(p * 100, 2))
    print(f'Achieves the set frequency goal: {p * 100 > limit}')


def frequency_analysis(region, standard_range=0.1, s=1):
    """Check whether the a synchronous area with frequency sample rate of 1s achieves its frequency
    goals for this frequency time series."""

    print(f'analyzing frequency time series from {region().name}...', end='\r')
    data = pd.read_pickle(region().data_path).dropna()
    data = data.rolling(s, min_periods=1).mean()
    full_size = data.shape[0]
    above = round((data[data.values > (50 + standard_range)].shape[0] / full_size) * 100, 2)
    under = round((data[data.values < (50 - standard_range)].shape[0] / full_size) * 100, 2)
    print(f'Above: {above}%\nUnder: {under}%\nTotalt in range: {round((100 - above - under), 2)}%')
    print(f'Outside of range: {int(((above + under) / 6000) * full_size)}min')


def plot_k(region, period, sample_start=1, sample_end=25, minutes=60):
    """Plot the adaptive k for a region for a given period"""
    if sample_end > 25 or minutes > 60:
        return 'Invalid input.'
    root = '/Users/thorbjornlundonsaker/workspace/Master/results/optimization/'
    idx = np.arange(60 * minutes)
    _, _, time = period_filename(period)
    for i in range(sample_start, sample_end + 1):
        data = np.load(root + f'{region}/{time}/k_opt_start0_cweightsFalse_win_size3600_test{i}.npz')
        plt.plot(idx, data['k_opt'][:60 * minutes])
    plt.show()


def find_periods_with_frequency(area, low_freq=49.2, high_freq=50.8):
    """Find frequencies above/below own limits and their belonging timestamps."""
    area = underscore(area)
    path = '/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{}/{}_50.pkl'
    df = get_a_pickle(path.format(area, area))
    high_freqs = df.index[df.values >= high_freq]
    low_freqs = df.index[df.values <= low_freq]
    print(f'low frequencies: {len(low_freqs.tolist())} \n high frequencies: {len(high_freqs.tolist())}')
    print(f'HIGH: {high_freqs.tolist()}')
    print(f'LOW: {low_freqs.tolist()}')
    print(low_freqs)


def check_test_set_for_nans(region, period=(1, 0), last=True):
    """Check the amount of nans [%] for a given time period at the end or start of a time series."""
    path = '/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{}/{}_50.pkl'
    region = underscore(region)
    data = pd.read_pickle(path.format(region, region))
    if last:
        data = data.iloc[-define_period(*period):]
        time = 'last'
    else:
        data = data.iloc[:define_period(*period)]
        time = 'first'
    pr = data.dropna().shape[0] / data.shape[0]
    print(f'{(1 - pr) * 100:.4f}% nans in the {time} {time_to_print(*period_filename(period))} of the data.')


def time_to_print(week, month, period):
    if month == 0 and week == 0:
        return 'FULL PERIOD'
    elif month == 0:
        return f'{week} {period[2:]}'
    return f'{month} {period[2:]}'


def statistics(region, period, check_equal=False):
    region = underscore(region)
    _, _, period = period_filename(period)
    path = '/Users/thorbjornlundonsaker/workspace/Master/results/{}/{}/{}/'
    file = pickle.load(open(path.format('optimization', region, period) + "statistics.pkl", "rb"))
    df = pd.DataFrame(file)
    if check_equal:
        file2 = pickle.load(open(path.format('eval_prediction', region, period) + "statistics.pkl", "rb"))
        df2 = pd.DataFrame(file2)
        print(df2.set_index('Sample'))
        check = df['Period start'].isin(df2['Period start']).value_counts()
        print(check)
    else:
        print(df.set_index('Sample'))


def check_train_length(region, period):
    region = underscore(region)
    _, _, t = period_filename(period)
    path_root = '/Users/thorbjornlundonsaker/workspace/Master/'
    data_path = path_root + f'cleaned_data/{region}/{region}_50.pkl'
    stats_path = path_root + f'results/optimization/{region}/{t}/statistics.pkl'
    data = pd.read_pickle(data_path)
    stats = pickle.load(open(stats_path, "rb"))
    for start, end in zip(stats['Period start'], stats['Period end']):
        show_period(data.loc[start:end], hour=True, show=True)


def save_fixed_test_dates(region, start, end):
    """Manually save the fixed test start date and end date as a dictionary for later use."""
    path_root = '/Users/thorbjornlundonsaker/workspace/Master/'
    stats_path = path_root + f'results/optimization/{region}/test_dates.pkl'
    stats = {'Period start': start, 'Period end': end}
    pickle.dump(stats, open(stats_path, 'wb'))
    print(f'Dates for {region} saved.')


def read_new_dim_data(area='Balearic_Islands'):
    """Read new data. Specialized for BalearicIslands data, manually change code for other data types and/or regions."""

    in_path = f'/Users/thorbjornlundonsaker/workspace/Master/format_data/{area}/generation/'
    out_path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{area}/'

    file = glob.glob(in_path + '*.csv')
    files = np.sort(file)

    data = pd.DataFrame()
    print('Processing the data...\n')
    for i, file in enumerate(files):
        print(f'File {i + 1} of {len(files)}', end='\r')
        try:
            # Read external data
            names = ['Time', 'Coal', 'Diesel engines', 'Gas turbine', 'Combined cycle', 'Balearic-Peninsula link', 'Solar PV',
                     'Other special regime', 'Thermal renewable', 'Wind', 'BalearicIslands-Menorca link', 'BalearicIslands-Ibiza link',
                     'Other renewables', 'Wastes', 'Auxiliary generation', 'Cogeneration', 'Ibiza-Formentera link']
            new_data = pd.read_csv(file, header=0, skiprows=2, names=names, index_col=False)

            # Append to other data
            if i == 0:
                data = data.append(new_data.iloc[:162])

            elif file == in_path + 'Custom-Report-2019-10-27-Generation mix (MW).csv':
                for index in new_data['Time']:
                    if index.__contains__("2A"):
                        new_data['Time'].replace(index, f"{index[:11]}02{index[13:]}", inplace=True)
                    if index.__contains__("2B"):
                        new_data['Time'].replace(index, f"{index[:11]}02{index[13:]}", inplace=True)
                data = data.append(new_data.iloc[18:168])

            elif file == in_path + 'Custom-Report-2020-03-29-Generation mix (MW).csv':
                data = data.append(new_data.iloc[18:156])

            elif file == in_path + 'Custom-Report-2020-01-01-Generation mix (MW).csv':
                data = data.append(new_data.iloc[18:])

            elif file == in_path + 'Custom-Report-2020-01-03-Generation mix (MW).csv':
                data = data.append(new_data.iloc[:162])

            elif file == in_path + 'Custom-Report-2021-02-21-Generation mix (MW).csv':
                data = data.append(new_data.iloc[18:])

            elif file == in_path + 'Custom-Report-2021-02-23-Generation mix (MW).csv':
                data = data.append(new_data.iloc[:162])

            elif i == (len(files) - 1):
                data = data.append(new_data.iloc[18:])
            else:
                data = data.append(new_data.iloc[18:162])

        except Exception as e:
            # There are some empty data files which have to be filtered out by this exception
            print(e)
            continue
    if len(data[data.index.duplicated()]) != 0:
        print('Some data points are read twice:')
        print(data[data.index.duplicated()])
        raise ValueError
    data = data.set_index('Time')
    # Localize the datetime index in its timezone
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize('Europe/Madrid', ambiguous='infer')

    # Reindex with full index and identify missing values with NaNs.
    start = data.index[0]
    end = data.index[-1]
    full_ind = pd.date_range(start=start, end=end, freq='10min', tz='Europe/Madrid')
    data = data.reindex(full_ind, fill_value=0.0)

    # Convert timestamp to naive local time (removing tz-information)
    data.index = data.index.tz_convert(None)

    # Save reindexed data
    print('Saving processed data...', end='\r')
    data.to_csv(out_path + 'generation.zip', float_format='%.2f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                'generation.csv'}, header=True)
    print('Data saved.')


def choose_generation_type(region='Balearic_Islands', cols='Balearic-Peninsula link', summation=False, new_filename='', diff=False):
    """Choose what column(s) from generation data to create new csv of for later use."""

    path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{region}/'
    names_from_file = ['Time', 'Coal', 'Diesel engines', 'Gas turbine', 'Combined cycle', 'Balearic-Peninsula link', 'Solar PV',
                       'Other special regime', 'Thermal renewable', 'Wind', 'BalearicIslands-Menorca link', 'BalearicIslands-Ibiza link',
                       'Other renewables', 'Wastes', 'Auxiliary generation', 'Cogeneration', 'Ibiza-Formentera link']
    idxs = [0]
    names = ['Time']
    if isinstance(cols, str):
        if cols == 'all':
            cols = names_from_file
            summation = True
            names = []
            idxs = []
        else:
            cols = [cols]
    filename = ''
    for col in cols:
        filename += underscore(col) + '_'
        idxs.append(names_from_file.index(col))
        names.append(col)
    print('Reading data...', end='\r')
    new_data = pd.read_csv(path + 'generation.csv', header=0, names=names, usecols=idxs, index_col=0, parse_dates=[0])

    # Add zeros where there are data missing in the time series.

    if summation:
        new_data = new_data.sum(axis=1)
    if len(new_filename) != 0:
        filename = new_filename

    if diff:
        new_data = new_data.diff()
        filename = filename + '_diff'

    print('Saving processed data...', end='\r')
    new_data.to_csv(path + f'{filename}.zip', float_format='%.2f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                    f'{filename}.csv'}, header=False)
    print('Data saved.')


def read_additional_data_ireland_entsoe(area='Ireland', feature='Wind Onshore'):
    """Read new data. Specialized for Ireland data, manually change code for other data types and/or regions."""

    in_path = f'/Users/thorbjornlundonsaker/workspace/Master/format_data/{area}/'
    out_path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{area}/'

    file = glob.glob(in_path + '202' + '*.csv')
    files = np.sort(file)
    data = pd.DataFrame()
    print('Processing the data...\n')
    for i, file in enumerate(files):
        print(f'File {i + 1} of {len(files)}', end='\r')
        try:
            # Read external data
            names = ["Area", "MTU", "Biomass", "Fossil Brown coal/Lignite",
                     "Fossil Coal-derived gas", "Fossil Gas",
                     "Fossil Hard coal", "Fossil Oil",
                     "Fossil Oil shale", "Fossil Peat",
                     "Geothermal", "Hydro Pumped Storage agg",
                     "Hydro Pumped Storage con", "Hydro Run-of-river and poundage",
                     "Hydro Water Reservoir", "Marine",
                     "Nuclear", "Other", "Other renewable",
                     "Solar", "Waste", "Wind Offshore",
                     "Wind Onshore"]

            new_data = pd.read_csv(file, header=0, names=names, index_col=False, usecols=['MTU', feature])
            if i == 0:
                new_data = new_data.iloc[14724:]
            if i == 1:
                new_data = new_data.iloc[:2600]
            # Append to other data
            new_data[feature] = new_data[feature].map(lambda x: float(x))
            new_data['MTU'] = new_data['MTU'].map(lambda x: str(x[:16]) + ':00')
            data = data.append(new_data)
        except Exception as e:
            # There are some empty data files which have to be filtered out by this exception
            print(e)
            continue
    # data[data.index.duplicated()]
    data = data.set_index('MTU')
    # Localize the datetime index in its timezone
    data.index = pd.to_datetime(data.index, dayfirst=True)
    data.index = data.index.tz_localize('Europe/Dublin', ambiguous='infer')

    # Reindex with full index and identify missing values with NaNs.
    start = data.index[0]
    end = data.index[-1]
    full_ind = pd.date_range(start=start, end=end, freq='30min', tz='Europe/Dublin')
    data = data.reindex(full_ind)

    # Convert timestamp to naive local time (removing tz-information)
    data.index = data.index.tz_convert(None)

    data = data.interpolate(limit=3)

    # Save reindexed data
    print('Saving processed data...', end='\r')
    data.to_csv(out_path + f'{underscore(feature)}.zip', float_format='%.2f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                f'{underscore(feature)}.csv'}, header=True)
    print('Data saved.')


def read_additional_data_ireland_eirgrid(area='Ireland', feature="Actual Wind Generation"):
    """Read new data. Specialized for Ireland data, manually change code for other data types and/or regions."""

    in_path = f'/Users/thorbjornlundonsaker/workspace/Master/format_data/{area}/wind_eirgrid/'
    out_path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{area}/'

    files = glob.glob(in_path + 'chart' + '*.csv')
    files = sorted(files, key=lambda f: int(f[len(in_path) + 6:-4]))
    data = pd.DataFrame()
    print('Processing the data...\n')
    for i, file in enumerate(files):
        print(f'File {file} of {len(files)}')
        try:
            # Read external data
            names = ["DateTime", "Actual Wind Generation", "Forecast Wind Generation"]

            new_data = pd.read_csv(file, header=0, names=names, index_col=False, usecols=['DateTime', feature])
            # Append to other data
            # new_data[feature] = new_data[feature].map(lambda x: float(x))
            # new_data["DateTime"] = new_data["DateTime"].map(lambda x: str(x[:16]) + ':00')
            data = data.append(new_data)
        except Exception as e:
            # There are some empty data files which have to be filtered out by this exception
            print(e)
            continue
    # data[data.index.duplicated()]
    data = data.set_index("DateTime")
    # Localize the datetime index in its timezone
    data.index = pd.to_datetime(data.index, yearfirst=True)
    data.index = data.index.tz_localize('Europe/Dublin', ambiguous='infer')

    # Reindex with full index and identify missing values with NaNs.
    start = data.index[0]
    end = data.index[-1]
    full_ind = pd.date_range(start=start, end=end, freq='15min', tz='Europe/Dublin')
    data = data.reindex(full_ind)

    # Convert timestamp to naive local time (removing tz-information)
    data.index = data.index.tz_convert(None)

    # Save reindexed data
    print('Saving processed data...', end='\r')
    data.to_csv(out_path + f'{underscore(feature)}.zip', float_format='%.2f', na_rep='NaN', compression={'method': 'zip', 'archive_name':
                f'{underscore(feature)}.csv'}, header=True)
    print('Data saved.')
