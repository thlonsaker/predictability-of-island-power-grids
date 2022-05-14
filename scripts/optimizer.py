from predictor import *
import os


class Optimizer(Predictor):
    validation = True

    def __init__(self, k_nearest_neighbors=np.arange(1, 300), size_dependent=False):
        super().__init__()
        self.k_nearest_neighbors = k_nearest_neighbors
        self.size_dependent = size_dependent
        self.weights = None

    def optimize(self, regions, periods=None):
        self.set_regions(regions)
        # Prediction, error evaluation and minimization for different parameters
        for region in self.areas:
            samples, times, period_info = region(self.validation, self.size_dependent)
            print(f'Running optimization for {region}...')
            sample_list = np.arange(1, samples + 1)
            data_path = self.data_path_root.format(underscore(region.name), underscore(region.name))
            main_data = pd.read_pickle(data_path)
            for time in times:
                if periods is not None:
                    if time not in periods:
                        break
                random.seed(12345)
                for points_per_chunk in self.points_per_chunk_s:
                    for start_min in self.pred_start_minute_s:
                        dp_starts, dp_ends, min_k, max_k = [], [], [], []
                        week, month, period = period_filename(time)
                        if not self.size_dependent:
                            period = '2_dim'
                        for sample in sample_list:
                            print(f'Period: {time_to_print(week, month, period)}, sample: {sample}', end='\n')
                            self.inner_function(underscore(region.name), points_per_chunk, start_min, sample, main_data,
                                                region.time_sensitive, dp_starts, dp_ends, min_k, max_k, week, month, period, period_info)

                        if self.size_dependent:
                            stats = {'Sample': sample_list, 'Period start': dp_starts,
                                     'Period end': dp_ends, 'Min k': min_k, 'Max k': max_k}
                            pickle.dump(stats,
                                        open(self.result_path_root + f'optimization/{underscore(region.name)}/{period}/statistics.pkl',
                                             'wb'))
            print(f'{underscore(region.name)} data is optimized for all periods.')
        print('Optimization is done.')

    def inner_function(self, region_name, points_per_chunk, start_min, sample, main_data, time_sensitive,
                       dp_starts, dp_ends, min_k, max_k, week, month, period, period_info):

        train_chunks, init_chunks, ensemble_test, dp_s, dp_e = generate_train_and_test_chunks(main_data,
                                                                                              points_per_chunk, start_min,
                                                                                              self.points_to_predict,
                                                                                              validation=self.validation,
                                                                                              size_dependent=self.size_dependent,
                                                                                              sample=sample, weeks=week, months=month,
                                                                                              **period_info, area=region_name)

        for c_weights in self.const_weights_s:
            mse_t, k_opt, k_opt_fixed, new_k_list, new_max_k = optimize_nearest_neighbor_predictor(train_chunks, init_chunks,
                                                                                                   points_per_chunk,
                                                                                                   self.points_to_predict,
                                                                                                   c_weights, ensemble_test, 1,
                                                                                                   self.k_nearest_neighbors,
                                                                                                   time_sensitive=time_sensitive,
                                                                                                   size_dependent=self.size_dependent)

            if self.size_dependent:
                path = f'{self.result_path_root}optimization/{region_name}/{period}/'
                np.savez(path + f'k_opt_start{start_min}_cweights{c_weights}_win_size{points_per_chunk}_test{sample}.npz',
                         k_opt=k_opt, k_opt_fixed=k_opt_fixed, k_nearest_neighbors=new_k_list)

                np.savez(path + f'mse_start{start_min}_cweights{c_weights}_win_size{points_per_chunk}_test{sample}.npz', mse_t=mse_t)

                dp_starts.append(dp_s), dp_ends.append(dp_e), min_k.append(min(k_opt)), max_k.append(max(k_opt))
            else:
                np.savez(self.result_path_root + 'optimization/' +
                         f'{region_name}/k_opt_start{start_min}_cweights{c_weights}_win_size{points_per_chunk}_test{sample}.npz',
                         k_opt=k_opt, k_opt_fixed=k_opt_fixed, k_nearest_neighbors=new_k_list,
                         mse_t=mse_t, size=new_max_k)

    def optimize_2dim(self, regions, feature, feature_chunk_length, weights=0, periods='all', sample_start=1):
        self.set_regions(regions)
        self.set_periods(periods)
        if isinstance(weights, int) or isinstance(weights, float):
            weights = [weights]
        self.weights = weights
        self.new_dim_chunk_length = feature_chunk_length

        for region in self.areas:
            print(f'Running optimization for {region.name}, {feature}...')
            samples, times, period_info = region(self.validation, self.size_dependent)
            reg_name = underscore(region.name)
            data_path = self.data_path_root.format(reg_name, reg_name)
            sample_list = np.arange(sample_start, samples + 1)
            main_data = pd.read_pickle(data_path)
            self.new_dim_data = pd.read_pickle(self.data_path_root.format(reg_name, feature)[:-7] + '.pkl')
            start_min, file = 0, 0
            points_per_chunk = 3600
            for time in times:
                random.seed(12345)
                week, month, period = period_filename(time)
                print(space(period))
                if self.size_dependent:
                    path = self.result_path_root + f'/optimization/{reg_name}/{period}/statistics.pkl'
                    file = pickle.load(open(path, "rb"))
                else:
                    period = '2_dim'
                for sample in sample_list:
                    print(f'sample: {sample}')
                    if self.size_dependent:
                        period_info = new_period_info(sample, week, month, file)
                        print(period_info)
                        data_tuple = self.generate_train_and_test_chunks_2dim(main_data, points_per_chunk,
                                                                              start_min, **period_info)
                    else:
                        data_tuple = self.generate_train_and_test_chunks_2dim(main_data, points_per_chunk,
                                                                              start_min, **period_info)

                    train_chunks, init_chunks, ensemble_test, dp_s, dp_e = data_tuple

                    result_tuple = self.nearest_neighbor_prediction_2dim_test(train_chunks, init_chunks, ensemble_test,
                                                                              points_per_chunk, n_jobs=1,
                                                                              constant_weights=False,
                                                                              time_sensitive=region.time_sensitive)

                    best_weight, best_k_opt, best_k_opt_fixed, best_mse, k_nn = result_tuple

                    path = f'{self.result_path_root}optimization/{reg_name}/2_dim/{feature}/adaptive_k_weight.npz'
                    if self.check_current_best(path, best_mse):
                        np.savez(path, k_opt_fixed=best_k_opt_fixed, weight=best_weight, best_mse=best_mse,
                                 k_nearest_neighbors=k_nn, k_opt=best_k_opt)
        print('Optimization is done.')

    @staticmethod
    def check_current_best(path, best_value):
        if os.path.exists(path):
            data = np.load(path)
            old_value = data['best_value']
            if best_value > old_value:
                return False
            print(f'New best: {best_value} - old best: {old_value}')
        return True

    def nearest_neighbor_prediction_2dim(self, train_chunks_full, init_chunks_full, ensemble_test,
                                         points_per_chunk, time_between_points=1, time_sensitive=True,
                                         constant_weights=True, n_jobs=1):
        """This function predicts the succesors of test chunks by searching for nearest neighbors
        in the set of training chunks. For each time t, the prediction is a weighted average
        of k(t) nearest neighbors. Thus, k_nearest_neighbors should be an array with length
        'points_to_predict', but it can also be a time-independent scalar value.
        NaN values are only allowed in the training data. """

        # Calculate the number of chunks to predict and the time between chunks
        chunks_to_predict = np.int(np.ceil(self.points_to_predict / points_per_chunk))
        time_between_chunks = time_between_points * points_per_chunk
        train_chunks = train_chunks_full.iloc[:, :self.points_to_predict]
        init_chunks_null = init_chunks_full.iloc[:, :self.points_to_predict]
        test_train_chunks = train_chunks_full.iloc[:, :self.points_to_predict]

        mse_weight = []
        optimal_k = []
        mses = []
        if time_sensitive:
            add_weighted_integer_time_index(test_train_chunks, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(train_chunks_full, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(init_chunks_full, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(init_chunks_null, time_between_chunks, time_weight=1000)

        train_chunks_without_nan = remove_nan_chunks(train_chunks_full, chunks_to_predict)
        train_chunks_without_nan_null = remove_nan_chunks(test_train_chunks, chunks_to_predict)

        pred_ind = range(1, chunks_to_predict + 1)
        # set new k:
        size = train_chunks_without_nan.shape[0] // 24
        if size < np.max(self.k_nearest_neighbors):
            self.k_nearest_neighbors = np.arange(1, size + 1)
            print(f'new k: {np.max(self.k_nearest_neighbors)}')
        for weight in self.weights:
            print(f'weight: {weight}')

            self.weight = weight

            if weight == 0:
                neighbor_finder = NearestNeighbors(n_neighbors=np.max(self.k_nearest_neighbors),
                                                   n_jobs=n_jobs)
                neighbor_finder.fit(train_chunks_without_nan_null.iloc[:-chunks_to_predict])
                print('Searching for neighbors...', end='\r')
                dists, nns = neighbor_finder.kneighbors(init_chunks_null)
                nns = np.array(train_chunks_without_nan_null.index)[nns]
            else:
                neighbor_finder = NearestNeighbors(n_neighbors=np.max(self.k_nearest_neighbors),
                                                   n_jobs=n_jobs, metric=self.special_weights)
                neighbor_finder.fit(train_chunks_without_nan.iloc[:-chunks_to_predict])
                print('Searching for neighbors...', end='\r')
                dists, nns = neighbor_finder.kneighbors(init_chunks_full)
                nns = np.array(train_chunks_without_nan.index)[nns]

            # Construct prediction by concatenating subsequent chunks and nearest neighbors
            predictions = np.concatenate([train_chunks.values[nns + i] for i in pred_ind], axis=-1)

            mse_t = np.zeros((self.k_nearest_neighbors.shape[0], self.points_to_predict))
            for i, k_nn in enumerate(self.k_nearest_neighbors):
                print('current k: ', k_nn, end='\r')

                weights = weights_from_distances(dists[:, :k_nn], self.points_to_predict, k_nn, constant_weights=constant_weights)

                predictions_mean = np.average(predictions[:, :k_nn, :], axis=1, weights=weights)
                # predictions_std = predictions[:, :k_nn, :].std(axis=1)

                mse_t[i] = calc_mse(predictions_mean.T, ensemble_test.T)
            sum_mse = mse_t.sum(axis=-1)
            print(f'sum mse: {sum_mse}')
            k_opt = self.k_nearest_neighbors[int(np.argmin(sum_mse, axis=0))]
            mse_weight.append(np.min(sum_mse, axis=0))
            optimal_k.append(int(k_opt))
            mses.append(mse_t)
        idx = int(np.argmin(np.array(mse_weight)))
        best_weight = self.weights[idx]
        best_k = optimal_k[idx]
        best_mse = mses[idx]
        best_value = np.min(np.array(mse_weight))
        print(f'Best k: {best_k}, best weight: {best_weight}')

        return best_weight, best_k, best_mse, best_value

    def nearest_neighbor_prediction_2dim_test(self, train_chunks_full, init_chunks_full, ensemble_test,
                                              points_per_chunk, time_between_points=1, time_sensitive=True,
                                              constant_weights=True, n_jobs=1):
        """This function predicts the succesors of test chunks by searching for nearest neighbors
        in the set of training chunks. For each time t, the prediction is a weighted average
        of k(t) nearest neighbors. Thus, k_nearest_neighbors should be an array with length
        'points_to_predict', but it can also be a time-independent scalar value.
        NaN values are only allowed in the training data. """

        # Calculate the number of chunks to predict and the time between chunks
        chunks_to_predict = np.int(np.ceil(self.points_to_predict / points_per_chunk))
        time_between_chunks = time_between_points * points_per_chunk
        train_chunks = train_chunks_full.iloc[:, :self.points_to_predict]
        init_chunks_null = init_chunks_full.iloc[:, :self.points_to_predict]
        test_train_chunks = train_chunks_full.iloc[:, :self.points_to_predict]

        mse_weight = []
        k_opt_s = []
        k_opt_fixed_s = []
        if time_sensitive:
            add_weighted_integer_time_index(test_train_chunks, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(train_chunks_full, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(init_chunks_full, time_between_chunks, time_weight=1000)
            add_weighted_integer_time_index(init_chunks_null, time_between_chunks, time_weight=1000)

        train_chunks_without_nan = remove_nan_chunks(train_chunks_full, chunks_to_predict)
        train_chunks_without_nan_null = remove_nan_chunks(test_train_chunks, chunks_to_predict)

        pred_ind = range(1, chunks_to_predict + 1)
        # set new k:
        k_nearest_neighbors = self.k_nearest_neighbors
        size = train_chunks_without_nan.shape[0] // 24
        if size < np.max(self.k_nearest_neighbors):
            k_nearest_neighbors = np.arange(1, size + 1)
        for weight in self.weights:
            print(f'weight: {weight}')

            self.weight = weight

            if weight == 0:
                neighbor_finder = NearestNeighbors(n_neighbors=np.max(k_nearest_neighbors),
                                                   n_jobs=n_jobs)
                neighbor_finder.fit(train_chunks_without_nan_null.iloc[:-chunks_to_predict])
                print('Searching for neighbors...', end='\r')
                dists, nns = neighbor_finder.kneighbors(init_chunks_null)
                nns = np.array(train_chunks_without_nan_null.index)[nns]
            else:
                neighbor_finder = NearestNeighbors(n_neighbors=np.max(k_nearest_neighbors),
                                                   n_jobs=n_jobs, metric=self.special_weights)
                neighbor_finder.fit(train_chunks_without_nan.iloc[:-chunks_to_predict])
                print('Searching for neighbors...', end='\r')
                dists, nns = neighbor_finder.kneighbors(init_chunks_full)
                nns = np.array(train_chunks_without_nan.index)[nns]

            # Construct prediction by concatenating subsequent chunks and nearest neighbors
            predictions = np.concatenate([train_chunks.values[nns + i] for i in pred_ind], axis=-1)

            mse_t = np.zeros((k_nearest_neighbors.shape[0], self.points_to_predict))
            for i, k_nn in enumerate(k_nearest_neighbors):
                print('current k: ', k_nn, end='\r')

                weights = weights_from_distances(dists[:, :k_nn], self.points_to_predict, k_nn, constant_weights=constant_weights)

                predictions_mean = np.average(predictions[:, :k_nn, :], axis=1, weights=weights)
                # predictions_std = predictions[:, :k_nn, :].std(axis=1)

                mse_t[i] = calc_mse(predictions_mean.T, ensemble_test.T)

            k_opt = optimal_k_from_errors(k_nearest_neighbors, mse_t, method='smooth')
            k_opt_fixed = optimal_k_from_errors(k_nearest_neighbors, mse_t, method='fixed')
            best_error = np.min(mse_t, axis=0).sum()
            print(f'sum mse: {best_error}, max k: {max(k_opt)}')
            mse_weight.append(best_error)
            k_opt_s.append(k_opt)
            k_opt_fixed_s.append(k_opt_fixed)

        idx = int(np.argmin(np.array(mse_weight)))
        best_weight = self.weights[idx]
        best_k_opt = k_opt_s[idx]
        best_k_opt_fixed = k_opt_fixed_s[idx]
        best_value = np.min(np.array(mse_weight))
        print(f'Best weight: {best_weight}, best fixed-k: {best_k_opt_fixed}, best adaptive-k: {best_k_opt}')

        return best_weight, best_k_opt, best_k_opt_fixed, best_value, k_nearest_neighbors

    @staticmethod
    def add_additional_feature_column(a, series, add_feature):
        # Construct new time index from total number of seconds
        column = add_feature.iloc[:, a]
        series.loc[:, series.shape[1]] = column
