from sklearn.neighbors import NearestNeighbors
from extra_functions import *
import random


def calc_mae(s_pred, s_test):
    return np.mean(np.abs(s_pred - s_test), axis=-1)


def calc_mse(s_pred, s_test):
    return np.mean((s_pred - s_test) ** 2, axis=-1)


def calc_rmse(s_pred, s_test):
    return np.sqrt(np.mean((s_pred - s_test) ** 2, axis=-1))


def check_nans(region, period=6):
    """Check the quality of the data for a region for a given number of months.
    Prints the wanted length with the least NaNs."""
    region = underscore(region)
    path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{region}/{region}_50.pkl'
    data = pd.read_pickle(path)
    start = data.index[0].ceil('H')
    data = data.loc[start:]
    chunks = construct_chunks(data, 3600, 0).iloc[:-(24 * 7)]
    time = int(24 * 30 * period * 0.8)
    best_frac = 0
    best_start, best_end, fracs = 0, 0, []
    for i in range(chunks.shape[0] - time):
        new_chunks = chunks.iloc[i:time + i]
        new_chunks_clean = remove_nan_chunks(new_chunks)
        frac = new_chunks_clean.shape[0] / new_chunks.shape[0]
        fracs.append((frac, chunks.index[i]))
        if frac > best_frac:
            best_frac = frac
            best_start = chunks.index[i]
            best_end = chunks.index[time + i]
            print(best_frac, end='\r')

    fracs.sort(key=lambda y: y[0], reverse=True)
    print(f'Best {period} months period with {best_frac} good data: {best_start} - {best_end}')


def daily_profile_prediction(path_to_original_series, init_chunks, points_to_predict, train_start, train_end,
                             skip_init_chunk=True, resampling_rate=1, main_data=None):
    """ Construct a forecast for test chunks at certain initialization start times by
    using the average daily profile of the original time series. """

    # TODO: Create a daily profile for the additional data?
    init_chunks = init_chunks.iloc[:, :3600]

    if main_data is not None:
        original_series = main_data.loc[train_start:train_end]
    else:
        original_series = pd.read_pickle(path_to_original_series).loc[train_start:train_end]

    # Calculate the sum of init points and prediction points
    n_points = points_to_predict + init_chunks.shape[1]

    # Drop the initial pattern if this option is chosen
    skip = 0
    if skip_init_chunk:
        skip = init_chunks.shape[1]

    # Calculate daily average profile
    daily_profile = original_series.groupby(by=original_series.index.time).mean()
    # Collect the set of timestamps for which a prediction should be generated
    init_start_times = init_chunks.index.time.astype(str)
    freq = '{}S'.format(resampling_rate)
    set_of_pred_times = [pd.date_range(str(start),
                                       periods=n_points,
                                       freq=freq).time[skip:] for start in init_start_times]

    # Look for the time in the daily profile that matches the prediction time and use it as forecast
    daily_profile_pred = np.array([daily_profile.loc[times].values for times in set_of_pred_times])

    return daily_profile_pred


def construct_chunks(time_series, chunk_length, start_time_within_hour=0):
    """Cut a time series into a set of non-overlapping chunks of the same length. Start and end
    of the set are chosen in such a way that the set starts and ends at a given time within an hour.
    The chunk length should be a factor of 60 [Minutes]. """

    # Set start and end time of the chunk set
    start = time_series.index[0].floor('H') + DateOffset(minutes=int(start_time_within_hour))
    end = get_end(time_series.index[-1], start_time_within_hour, chunk_length)
    chunks = time_series.loc[start:end]

    return remain(chunks, chunk_length)


def get_end(end, start_time_within_hour, chunk_length):
    remove_sec = int(3600 / chunk_length)
    end = end.ceil('H') + DateOffset(minutes=int(start_time_within_hour)) - DateOffset(seconds=remove_sec)
    return end


def remain(chunks, chunk_length, new_dim_data=None, new_dim_chunk_length=0, scale=False):
    # If chunk_length is larger than 1h there might be a remainder when cutting the time series into chunks...
    remainder = int(chunks.shape[0] % chunk_length)

    if remainder != 0:
        chunks = chunks.iloc[:-remainder]
    # Construct chunks as dataframe with time indices
    chunks = pd.DataFrame(data=chunks.values.reshape((chunks.shape[0] // chunk_length, chunk_length)),
                          index=chunks.index[::chunk_length])
    return chunks


def remove_nan_chunks(chunks, predecessors_to_remove=1, drop_time_index=True):
    # Copy chunks
    chunks_without_nan = chunks.copy()
    # Add integer index (either drop the time index or add integer index as column)
    if drop_time_index:
        chunks_without_nan.reset_index(drop=True, inplace=True)
    else:
        chunks_without_nan['int_index'] = np.arange(chunks.shape[0])

    # Create mask for rows with at least one Nan
    nan_row_mask = np.isnan(chunks).any(axis=1)
    mask_to_remove = nan_row_mask.copy()

    # Remove preceding chunks, too
    for i in range(1, predecessors_to_remove + 1):
        mask_to_remove = mask_to_remove | nan_row_mask.shift(-i)

    # Remove the masked rows
    chunks_without_nan = chunks_without_nan[~mask_to_remove.values]

    return chunks_without_nan


def generate_train_and_test_chunks(data, points_per_chunk, pred_start_minute, points_to_predict, n_tests=-1,
                                   train_start='2019-09-29 00:00:00', train_end='2019-03-31 23:59:59',
                                   test_start='2019-04-01 00:00:00', test_end='2012-06-30 23:59:59',
                                   train_ratio=0.8, size_dependent=False, fixed_test_size=(1, 0),
                                   validation=False, weeks=0, months=0, limit=0.9, sample=0, area=None):

    if size_dependent and validation:
        period_length = define_period(weeks, months)
        if area == 'Ireland':
            _, _, period = period_filename((weeks, months))
            path = '/Users/thorbjornlundonsaker/workspace/Master/results/{}/{}/{}/'
            stats = pickle.load(open(path.format('optimization', area, period) + "statistics.pkl", "rb"))
            train_start = stats['Period start'][sample - 1]
            train_end = stats['Period end'][sample - 1]
            train_data = data.loc[train_start:train_end]
            train_chunks = construct_chunks(train_data, points_per_chunk, pred_start_minute)
            file = pickle.load(open('/Users/thorbjornlundonsaker/workspace/Master/results/optimization/Ireland/test_dates.pkl',
                                    "rb"))
            test_data = data.loc[file['Period start']:file['Period end']]
            test_chunks = construct_chunks(test_data, points_per_chunk, pred_start_minute)
        else:
            train_chunks, test_chunks, train_start, train_end = fix_size_dependent(data, period_length, train_ratio,
                                                                                   points_per_chunk, pred_start_minute, validation,
                                                                                   fixed_test_size, limit, area, sample)
    else:
        # data = pd.read_pickle(data_path).loc[train_start:test_end]  OLD
        # Create train and test chunks. Use 'test_year' to construct test chunks
        data = data.loc[train_start:test_end]
        train_chunks = construct_chunks(data.loc[:train_end], points_per_chunk, pred_start_minute)
        test_chunks = construct_chunks(data.loc[test_start:], points_per_chunk, pred_start_minute)

    # Calculate the start time for the init chunks and the number of chunks to predict
    init_start_minute = (pred_start_minute - points_per_chunk / 60) % 60
    chunks_to_predict = np.int(np.ceil(points_to_predict / points_per_chunk))

    # Remove nan chunks from test chunks and choose chunks with correct start time
    init_chunks = remove_nan_chunks(test_chunks, chunks_to_predict, drop_time_index=False)
    init_chunks = init_chunks.iloc[:-chunks_to_predict]
    init_chunks = init_chunks[init_chunks.index.minute == init_start_minute]

    # Randomly select n_tests init chunks. Otherwise select all chunks
    if (type(n_tests) == int) and (n_tests != -1):
        init_chunks = init_chunks.sample(n_tests, random_state=1)

    # Construct an ensemble of test time series for the prediction time interval
    ensemble_test = np.zeros((init_chunks.shape[0], points_to_predict))
    for i, ind in enumerate(init_chunks.int_index):
        ensemble_test[i] = test_chunks.values[ind + 1:ind + chunks_to_predict + 1].flatten()[:points_to_predict]

    # Drop integer index that was used to construct ensemble_test
    init_chunks.drop(columns=['int_index'], inplace=True)

    return train_chunks, init_chunks, ensemble_test, train_start, train_end


def add_weighted_integer_time_index(series, original_time_distance, time_weight=1000):
    # Construct new time index from total number of seconds
    times = series.index.time
    times = np.array([(t.hour * 3600 + t.minute * 60 + t.second) for t in times])

    # Normalize the distance between timestamps and weight the new index
    times = time_weight * times / original_time_distance

    # Attach it to the series
    series.loc[:, series.shape[1]] = times


def weights_from_distances(dists, n_time_steps, k_nearest_neighbors, constant_weights=False):
    """Construct weights that decrease linearly with distance and normalize their sum to 1.
    The weights depend on the time step, if a time-dependent k_nearest_neighbor is given.
    The value dists.shape[1] should be at least max(k_nearest_neighbors)."""

    weights = np.zeros((dists.shape[0], dists.shape[1], n_time_steps))

    for i in range(n_time_steps):

        # Choose k(t) if k_nearest_neighbors is time-dependent
        if type(k_nearest_neighbors) == np.ndarray:
            k_t = k_nearest_neighbors[i]
        else:
            k_t = k_nearest_neighbors

        # Choose distances for k(t) nearest neighbors
        dists_t = dists[:, :k_t]

        # Construct normalized weights
        if constant_weights or k_t == 1:
            weights_t = np.ones(dists_t.shape).T
        else:
            weights_t = (dists_t.max(axis=1) - dists_t.T) / (dists_t.max(axis=1) - dists_t.min(axis=1))
        weights_t = weights_t / weights_t.sum(axis=0)

        # Assign the non-zero weights. For k>k(t) the weights are zero.
        weights[:, :k_t, i] = weights_t.T

    return weights


def fix_size_dependent(main_data, period_length, train_ratio, points_per_chunk,
                       pred_start_minute, validation, fixed_test_size, limit, area, sample):
    """This function expects that the fixed test set at end of
    data set has been checked manually for too many nans."""

    test_start = define_period(*fixed_test_size)
    test_data = main_data.iloc[-test_start:]
    if test_data.index[0] != test_data.index[0].floor('H'):
        test_start += points_per_chunk
        test_data = main_data.iloc[-test_start:]

    test_chunks = remain(test_data, points_per_chunk)
    new_start = main_data.index[0].ceil('H') + DateOffset(minutes=int(pred_start_minute))  # ! might skip an hour of data
    new_end = main_data.index[-test_start].floor('H') + DateOffset(minutes=int(pred_start_minute)) - DateOffset(seconds=1)  # !

    data = main_data.loc[new_start:new_end]
    size = data.shape[0]
    good_data, count = False, 0

    while good_data is False:
        if area == 'Ireland':
            start = (sample-1)*3600*4
            chosen_data = data.iloc[start:start + period_length]
            train_end_idx = round(period_length * train_ratio)
            train_end = chosen_data.index[train_end_idx].ceil('H') - DateOffset(seconds=1) + DateOffset(minutes=int(pred_start_minute))
            val_start = chosen_data.index[train_end_idx].ceil('H') + DateOffset(minutes=int(pred_start_minute))
            train_chunks = remain(chosen_data.loc[:train_end], points_per_chunk)
            val_chunks = remain(chosen_data.loc[val_start:], points_per_chunk)
        else:
            if size > test_start + points_per_chunk:
                maximum = size - period_length
                random_start = random.randrange(0, maximum, points_per_chunk)
                chosen_data = data.iloc[random_start:random_start + period_length]
                new_size = chosen_data.shape[0]
            else:
                new_size = size
                chosen_data = data

            train_end_idx = round(new_size * train_ratio)
            train_end = chosen_data.index[train_end_idx].ceil('H') - DateOffset(seconds=1) + DateOffset(minutes=int(pred_start_minute))
            val_start = chosen_data.index[train_end_idx].ceil('H') + DateOffset(minutes=int(pred_start_minute))
            train_chunks = remain(chosen_data.loc[:train_end], points_per_chunk)
            val_chunks = remain(chosen_data.loc[val_start:], points_per_chunk)
        val_length = val_chunks.shape[0]
        good_data = chunks_quality([train_chunks, val_chunks], limit)
        if area == 'Ireland':
            k = 1
            while chunks_quality(train_chunks, limit) and not chunks_quality(val_chunks, limit):
                val_chunks = remain(data.loc[val_start + DateOffset(hours=k):val_start + DateOffset(hours=int(val_length*3600 + k))],
                                    points_per_chunk)
                k += 1
                if k % 100 == 0:
                    print(f'k: {k}', end='\r')
            good_data = chunks_quality([train_chunks, val_chunks], limit)

        count += 1
        if count > 100:
            raise ValueError(f'Train and test sets with less than {(1 - limit) * 100}% nans has been \n'
                             f'generated {count} times without success.')
    if validation:
        return train_chunks, val_chunks, chosen_data.index[0], train_end
    return train_chunks, test_chunks, chosen_data.index[0], train_end


def nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk, points_to_predict,
                                k_nearest_neighbors, time_between_points=1, time_sensitive=True, return_error=False,
                                constant_weights=True, n_jobs=1):
    """This function predicts the succesors of test chunks by searching for nearest neighbors
    in the set of training chunks. For each time t, the prediction is a weighted average
    of k(t) nearest neighbors. Thus, k_nearest_neighbors should be an array with length
    'points_to_predict', but it can also be a time-independent scalar value.
    NaN values are only allowed in the training data. """

    # Calculate the number of chunks to predict and the time between chunks
    chunks_to_predict = np.int(np.ceil(points_to_predict / points_per_chunk))
    time_between_chunks = time_between_points * points_per_chunk

    # Convert the timestamps of the chunks to integers with distance "time_weight"
    if time_sensitive:
        add_weighted_integer_time_index(train_chunks, time_between_chunks, time_weight=1000)
        add_weighted_integer_time_index(init_chunks, time_between_chunks, time_weight=1000)

    # Construct a training set without NaNs
    train_chunks_without_nan = remove_nan_chunks(train_chunks, chunks_to_predict)
    # Construct nearest neighbor finder (for max(k(t)) if k is time-dependent)
    neighbor_finder = NearestNeighbors(n_neighbors=np.max(k_nearest_neighbors), n_jobs=n_jobs)
    neighbor_finder.fit(train_chunks_without_nan.iloc[:-chunks_to_predict])

    # Find nearest neighbors and their distances using the euclidean norm
    dists, nns = neighbor_finder.kneighbors(init_chunks)
    nns = np.array(train_chunks_without_nan.index)[nns]

    # Drop integer time index again
    if time_sensitive:
        train_chunks.drop(columns=[train_chunks.shape[1] - 1], inplace=True)
        init_chunks.drop(columns=[init_chunks.shape[1] - 1], inplace=True)

    # Construct prediction by concatenating subsequent chunks and nearest neighbors
    pred_ind = range(1, chunks_to_predict + 1)
    predictions = np.concatenate([train_chunks.values[nns + i] for i in pred_ind],
                                 axis=-1)
    predictions = predictions[:, :, :points_to_predict]

    # Construct weights from distances for each time step...
    weights = weights_from_distances(dists, points_to_predict,
                                     k_nearest_neighbors, constant_weights=constant_weights)

    # Average over nearest neighbors
    predictions_mean = np.average(predictions, axis=1, weights=weights)
    if return_error:
        return predictions_mean, predictions.std(axis=1)
    return predictions_mean


def optimal_k_from_errors(k_nearest_neighbors, errors, k_opt_resolution=60, window_size=60, method='mean', new_dim_chunk_length=0):
    k_opt = 0

    if method == 'mean':
        errors_mean = errors.reshape((k_nearest_neighbors.shape[0],
                                      errors.shape[-1] // k_opt_resolution,
                                      k_opt_resolution)).mean(axis=-1)
        k_opt = k_nearest_neighbors[np.argmin(errors_mean, axis=0)].astype('int')
        k_opt = (np.ones((errors.shape[-1] // k_opt_resolution,
                          k_opt_resolution)) * k_opt).T.flatten().astype(int)

    if method == 'smooth':
        k_opt = k_nearest_neighbors[np.argmin(errors, axis=0)]
        k_opt = pd.Series(k_opt).rolling(window_size, center=True,
                                         min_periods=window_size // 2).mean().values
        k_opt = np.ceil(k_opt).astype('int')

    if method == 'original':
        k_opt = k_nearest_neighbors[np.argmin(errors, axis=0)]

    if method == 'fixed':
        k_opt = k_nearest_neighbors[np.argmin(errors.sum(axis=-1), axis=0)]
        k_opt = int(k_opt)

    if new_dim_chunk_length > 0:
        return k_opt, np.min(errors, axis=0).mean()
    return k_opt


def optimize_nearest_neighbor_predictor(train_chunks, init_chunks, points_per_chunk, points_to_predict,
                                        constant_weights, ensemble_test, n_jobs,
                                        k_nearest_neighbors, time_sensitive=True, size_dependent=True):

    train_chunks_size = remove_nan_chunks(train_chunks, np.int(np.ceil(points_to_predict / points_per_chunk)),
                                          drop_time_index=False)
    train_size = train_chunks_size.shape[0]
    new_max_k = get_max_number_of_days(train_chunks_size)
    if size_dependent:
        if train_size < max(k_nearest_neighbors) or time_sensitive:
            if time_sensitive:
                k_nearest_neighbors = np.arange(1, new_max_k + 1)
            else:
                k_nearest_neighbors = np.arange(1, train_size)
    elif time_sensitive and new_max_k < max(k_nearest_neighbors):
        k_nearest_neighbors = np.arange(1, new_max_k + 1)
    elif not time_sensitive and train_size < max(k_nearest_neighbors):
        print(f'New maximum k: {train_size}')
        k_nearest_neighbors = np.arange(1, train_size)

    mse_t = np.zeros((k_nearest_neighbors.shape[0], points_to_predict))

    for i, k_nn in enumerate(k_nearest_neighbors):
        print('current k: ', k_nn, end='\r')
        ensemble_pred = nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk,
                                                    points_to_predict, k_nn, n_jobs=n_jobs,
                                                    constant_weights=constant_weights,
                                                    time_sensitive=time_sensitive)

        mse_t[i] = calc_mse(ensemble_pred.T, ensemble_test.T)

    k_opt = optimal_k_from_errors(k_nearest_neighbors, mse_t, method='smooth')
    k_opt_fixed = optimal_k_from_errors(k_nearest_neighbors, mse_t, method='fixed')

    return mse_t, k_opt, k_opt_fixed, k_nearest_neighbors, new_max_k


def get_max_number_of_days(train_chunks):
    """Returns the maximum number of days used to calculate the daily profile,
    and thus an upper bound for k if time sensitive."""
    size = []
    for i in range(24):
        size.append(train_chunks.groupby(train_chunks.index.hour == i).count().loc[True, 1])
    print(size)
    return int(max(size))
