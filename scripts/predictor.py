import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pattern_prediction_test import *
from regions import *
from extra_functions import *


class Predictor:
    validation = False
    plt.rc('text', usetex=True)
    areas = [Nordic(), Ireland(), BalearicIslands(), Iceland(), FaroeIslands()]
    areas_grid = [[Nordic(), None], [Ireland(), BalearicIslands()], [Iceland(), FaroeIslands()]]
    weight = 0
    new_dim_data = None
    scale = None
    new_dim_chunk_length = None
    base = Standards()
    data_path_root = '/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/{}/{}_50.pkl'
    result_path_root = '/Users/thorbjornlundonsaker/workspace/Master/results/'
    k_opt_archive = 'k_opt_start{}_cweightsFalse_win_size{}_test{}.npz'
    out_file = 'performance_start{}_cweightsFalse_win_size{}_test{}.npz'

    def __init__(self, points_per_chunk_s=np.array([3600]), pred_start_minute_s=np.array([0]),
                 points_to_predict=3600, const_weights_s=np.array([False])):

        self.points_per_chunk_s = points_per_chunk_s
        self.pred_start_minute_s = pred_start_minute_s
        self.points_to_predict = points_to_predict
        self.const_weights_s = const_weights_s
        self.one_pred_index = np.arange(points_to_predict) / 60

    @classmethod
    def set_regions(cls, regions):
        if regions != 'all':
            if isinstance(regions, str):
                regions = [regions]
            new_regions = []
            for r in regions:
                for reg in cls.areas:
                    if underscore(r) == underscore(reg.name):
                        new_regions.append(reg)
            cls.areas = new_regions

    @staticmethod
    def plot_region_names(regions):
        return [region.name for region in regions]

    @staticmethod
    def plot_periods(periods):
        return [time_to_print(*period_filename(time)) for time in periods].append('full\ntime series')

    @staticmethod
    def generation_plot(savefig=False):
        test = pd.read_csv('/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/Balearic_Islands/generation.csv',
                           parse_dates=[0])
        test = test.set_index('Unnamed: 0')
        start = '2020-06-13 00:00:00'
        end = '2020-06-19 23:59:59'
        start = '2021-09-29 00:00:00'
        end = '2021-03-10 23:59:59'
        data = test.loc[start:end]
        d = []
        ax = plt.subplot()
        for feature in data.columns:
            d.append((data[feature].to_numpy(), feature))
        d = sorted(d, key=lambda x: x[0].sum(), reverse=True)
        size = len(d[0][0])
        old_data, i = np.zeros(size), 0
        d = d[:6]
        while i < len(d):
            data, label = d[i]
            ax.plot(np.arange(size), data + old_data, label=label, lw=0.7)
            ax.fill_between(np.arange(size), old_data + data, old_data, alpha=0.7)
            old_data += data
            i += 1
        ax.tick_params(axis='both', which='both', labelsize=14)
        ax.set_ylabel('MW', fontsize=15)
        ax.set_xlabel(f'{start[:4]}', fontsize=15)
        ax.locator_params(axis='x', nbins=11)
        s = int(start[8:10])
        m = int(start[5:7])
        plt.subplots_adjust(-0.1)
        ax.set_xticks(np.linspace(0, size + 1, 8), [f'{s + i}.0{m}' for i in range(8)])
        leg = ax.legend(ncol=2, fontsize=15, bbox_to_anchor=(0.43, -0.65, 0.5, 0.5))
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        plt.tight_layout()
        if savefig:
            plt.savefig('plots/generation.pdf')
        plt.show()

    def plot_features(self):
        region = BalearicIslands()
        d = []
        for feature in region.additional_features[1:]:
            data = pd.read_pickle(f'/cleaned_data/Balearic_Islands/{feature}.pkl')
            d.append((data.loc['2019-12-12 00:00:00':'2019-12-18 23:59:59'].to_numpy(), feature))
        d = sorted(d, key=lambda x: x[0].sum(), reverse=True)
        size = len(d[0][0])
        old_data, i = np.zeros(size), 0
        colors = dict(zip([x[1] for x in d], ['brown', 'lightblue', 'k', 'lime']))
        while i < len(d):
            data, label = d[i]
            plt.plot(np.arange(size), data + old_data, label=label, lw=0.7, c=colors[label])
            plt.fill_between(np.arange(size), old_data + data, old_data, alpha=0.7, facecolor=colors[label])
            old_data += data
            i += 1
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_relative_new_feature(self, minutes=60):
        """"""
        region = Ireland()
        # self.set_periods(region.periods[-1])
        ax = plt.subplot()
        color = self.base.prediction()
        filename = 'average_result_performance.npz'
        error = 'pred_error_mean'
        x = np.arange(len(region.periods))
        avg, avg_2dim = [], []
        week, month, period, k = 0, 0, 0, 0
        for feature in ['wind_generation']:  # region.additional_features:
            for k, times in enumerate(region.periods[:-1]):
                week, month, period = period_filename(times)
                a = np.load(self.result_path_root + f'eval_prediction/{underscore(region.name)}/{period}/' + filename)
                avg.append(a[error][:60 * minutes].mean())
                a_2dim = np.load(self.result_path_root + f'eval_prediction/{underscore(region.name)}/2_dim/{feature}/{period}/' +
                                 filename)
                avg_2dim.append(a_2dim[error][:60 * minutes].mean())
            avg_2dim.append(1), avg.append(1)
            ax.plot(x, np.array(avg_2dim) / np.array(avg), color=color[k],
                    label=time_to_print(week, month, period))
        ax.set_ylabel('Relative RMSE', fontsize=self.base.ylabel)
        ax.set_xlabel('Regions', fontsize=self.base.xlabel + 1)
        ax.set_xticks(x, self.plot_periods(region.periods), fontsize=self.base.xlabel)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = plt.legend(by_label.values(), by_label.keys(), handlelength=0)
        for j, text in enumerate(leg.get_texts()):
            text.set_color(color[j])
        for item in leg.legendHandles:
            item.set_visible(False)
        # c = Patterns().colors
        # ax.plot(10, 0.045, visible=True, label='Prediction error', marker='o', c=c, linestyle='None')
        # ax.plot(10, 0.045, visible=True, label='Daily profile error', marker='^', c=c, linestyle='None')
        # ax.plot(10, 0.045, visible=True, label='50 Hz error', marker='s', c=c, linestyle='None')
        # plt.legend()
        plt.show()

    def dot_plot(self, minutes=60, savefig=False):
        """"""
        self.areas = self.areas[:-1]
        ax = plt.subplot()
        color = self.base.prediction()
        filename = 'average_result_performance.npz'
        errors = ['', 'pred_error_mean', 'daily_profile_error_mean', 'fiftyhz_error_mean', '']
        marker = {'pred_error_mean': 'o', 'daily_profile_error_mean': '^', 'fiftyhz_error_mean': 's'}
        n_areas = len(self.areas)
        steps = (n_areas + 2) * n_areas
        x = np.arange(3, steps - 1, 5)
        i = 0
        for area in self.areas:
            for error in errors:
                i += 1
                if len(error) != 0:
                    for k, times in enumerate(area.periods):
                        week, month, period = period_filename(times)
                        avg = np.load(self.result_path_root + f'eval_prediction/{underscore(area.name)}/{period}/' + filename)
                        avg = avg[error][:60 * minutes].mean()
                        ax.plot(i, avg, color=color[k], marker=marker[error], label=time_to_print(week, month, period),
                                markersize=8, linestyle='None')
        ax.set_ylabel('Average RMSE$(\hat f)$ [Hz]', fontsize=14)
        ax.set_xticks(x, self.plot_region_names(self.areas), fontsize=self.base.xlabel - 1, minor=False)
        ax.set_xticks([5.5, 10.5, 15.5], minor=True)
        if minutes != 60:
            ax.set_title(f'--- First {minutes} min\n--- of the hour', pad=-36, y=1, fontsize=18)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = plt.legend(by_label.values(), by_label.keys(), handlelength=0)
        for j, text in enumerate(leg.get_texts()):
            text.set_color(color[j])
        for item in leg.legendHandles:
            item.set_visible(False)
        # c = self.base.patterns
        # ax.plot(10, 0.045, visible=True, label='WNN prediction error', marker='o', c=c, linestyle='None')
        # ax.plot(10, 0.045, visible=True, label='Daily profile error', marker='^', c=c, linestyle='None')
        # ax.plot(10, 0.045, visible=True, label='50 Hz error', marker='s', c=c, linestyle='None')
        # plt.legend()
        # plt.savefig('plots/legend.pdf')
        plt.grid(axis='x', which='minor')
        if savefig:
            if minutes != 60:
                plt.savefig(f'plots/dot_plot{minutes}.pdf')
            else:
                plt.savefig('plots/dot_plot.pdf')
        plt.show()

    def plot_rmse_full_series(self, savefig=False):

        fig, axs = plt.subplots(3, 2, figsize=(14, 14))
        filename = 'performance_start0_cweightsFalse_win_size3600_test1.npz'
        areas = self.areas_grid
        # areas[-1][-1], areas[0][0] = Nordic(), FaroeIslands()
        for i in range(3):
            for j in range(2):
                region = areas[i][j]
                if region is None:
                    axs[i, j].set_visible(False)
                    continue
                data = np.load(self.result_path_root + f'eval_prediction/{underscore(region.name)}/' + filename)
                axs[i, j].plot(self.one_pred_index, data['fiftyHz_error'], color=self.base.fiftyhz, label='50 Hz', lw=1.2)
                axs[i, j].plot(self.one_pred_index, data['daily_profile_error'], color=self.base.daily_profile,
                               label='Daily profile', lw=1)
                axs[i, j].plot(self.one_pred_index, data['pred_error_adaptive_k'], color=self.base.one_pred,
                               label='WNN prediction', lw=0.8)
                axs[-1, j].set_xlabel('Time $\Delta t$ [min]', fontsize=17)
                axs[i, 0].set_ylabel('RMSE$(\hat f)$ [Hz]', fontsize=17)
                axs[i, j].set_xticks([0, 15, 30, 45, 60])
                axs[i, j].grid(axis='x')
                axs[i, j].set_title(region, fontsize=19)
                axs[i, j].tick_params(axis='both', which='both', labelsize=15)

        leg = axs[0, 0].legend(ncol=3, bbox_to_anchor=(1.3, -3.08, 0.5, 0.5), fontsize=18)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        if savefig:
            plt.savefig('plots/rmse.pdf', bbox_inches='tight')
        plt.show()

    def plot_hour_chosen_periods(self, savefig=False, start=0, end=60):
        self.set_regions(['Ireland', 'Balearic Islands'])
        fig, axs = plt.subplots(2, 1, figsize=(10, 11), sharex='col')
        color = self.base.prediction()
        filename = 'average_result_performance.npz'
        legend_cols = 0
        times_to_plot = [(1, 0), (0, 2), (0, 6)]
        for i, region in enumerate(self.areas):
            info = []
            for k, times in enumerate(region.periods):
                lw = 1.5
                if times in times_to_plot:
                    if region.name == 'Balearic Islands' and times == (0, 2):
                        lw = 0.0
                    week, month, period = period_filename(times)
                    data = np.load(self.result_path_root +
                                   f'eval_prediction/{underscore(region.name)}/{period}/' + filename)['pred_error_mean']
                    info.append(data)
                    axs[i].plot(self.one_pred_index, data, color=color[k], label=time_to_print(week, month, period), lw=lw)
            old = np.mean(info[0][start * 60:end * 60])
            new = np.mean(info[-1][start * 60:end * 60])
            improvement = (old - new) / new
            print(f'{region}, {start}-{end}min: {round(improvement * 100, 1)}%')

            axs[i].set_xticks([0, 15, 30, 45, 60])
            axs[i].set_ylabel('RMSE$(\hat f)$ [Hz]', fontsize=18)
            axs[1].set_xlabel('Time $\Delta t$ [min]', fontsize=18)
            axs[i].grid(axis='x')
            axs[i].tick_params(axis='both', labelsize=15)
            axs[i].set_title(f'{region.name}', fontsize=21)

        plt.subplots_adjust(bottom=0.1)
        leg = axs[1].legend(ncol=3, bbox_to_anchor=(0.32, -0.62, 0.5, 0.5), fontsize=19)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        plt.tight_layout()
        if savefig:
            plt.savefig('plots/rmse_periods.pdf')
        plt.show()

    def plot_hour_chosen_periods_relative(self, savefig=False):
        self.set_regions(['Nordic', 'Ireland', 'BalearicIslands', 'Iceland'])
        filename = 'average_result_performance.npz'
        for region in self.areas:
            d = []
            for k, times in enumerate(region.periods):
                if k == 0 or k == len(region.periods) - 1:
                    week, month, period = period_filename(times)
                    data = np.load(self.result_path_root +
                                   f'eval_prediction/{underscore(region.name)}/{period}/' + filename)['pred_error_mean']
                    d.append(data)
            plt.plot(self.one_pred_index, d[1] / d[0], color=region.color, label=region.name, lw=1.5)
        plt.xticks([0, 15, 30, 45, 60])
        plt.ylabel('RMSE$(\hat f)$ [Hz]', fontsize=16)
        plt.xlabel('Time $\Delta t$ [min]', fontsize=16)
        plt.grid(axis='x')
        plt.tick_params(axis='both', labelsize=14)
        plt.title('test', fontsize=19)

        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig('plots/rmse_periods.pdf')
        plt.show()

    def plot_hour_all_periods_one_region(self, region):
        self.set_regions(region)
        plt.figure()
        axs = plt.subplot()
        color = self.base.prediction()
        filename = 'average_result_performance.npz'
        legend_cols = 0
        x = np.arange(self.points_to_predict) / 60
        for region in self.areas:
            for k, times in enumerate(region.periods):
                week, month, period = period_filename(times)
                data = np.load(self.result_path_root + f'eval_prediction/{underscore(region.name)}/{period}/' + filename)['pred_error_mean']
                axs.plot(x, data, color=color[k], label=time_to_print(week, month, period))
                legend_cols = len(region.periods) // 2

        axs.set_xticks([0, 15, 30, 45, 60])
        axs.set_xlabel('Time $\Delta t$ [min]', fontsize=14)
        axs.grid(axis='x')

        axs.set_title(f'{region.name}', fontsize=14)
        leg = axs.legend(ncol=legend_cols, bbox_to_anchor=(0.18, -0.35, 0.5, 0.5), fontsize=14)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        plt.tight_layout()
        plt.show()

    def set_periods(self, periods):
        if periods != 'all':
            if isinstance(periods, tuple):
                periods = [periods]
            for region in self.areas:
                region.set_periods(periods)

    def compute_average_rmse(self, region='all', periods='all', feature=None):
        """Compute average of samples from different periods."""

        filename = 'performance_start0_cweightsFalse_win_size3600_test{}.npz'
        out_name = 'average_result_performance.npz'
        self.set_regions(region)
        self.set_periods(periods)
        for region in self.areas:
            print(f'Computing average and standard deviation of {region.name}...', end='\r')
            for times in region.periods:
                reg_name = underscore(region.name)
                week, month, period = period_filename(times)
                path = f'eval_prediction/{reg_name}/{period}/'
                fiftyhz_error = []
                daily_profile_error = []
                pred_error_adaptive_k = []
                for sample in range(1, region.samples + 1):
                    if feature is not None:
                        path = f'eval_prediction/{reg_name}/2_dim/{feature}/{period}/'
                        data = np.load(self.result_path_root + path + filename.format(sample))
                    else:
                        data = np.load(self.result_path_root + path + filename.format(sample))
                    fifty, dp, adapt_k = data['fiftyHz_error'], data['daily_profile_error'], data['pred_error_adaptive_k']
                    fiftyhz_error.append(fifty)
                    daily_profile_error.append(dp)
                    pred_error_adaptive_k.append(adapt_k)

                pred_error_adaptive_k = np.array(pred_error_adaptive_k)
                pred_std, pred_mean = np.std(pred_error_adaptive_k, axis=0), np.mean(pred_error_adaptive_k, axis=0)
                dp_mean, fiftyhz_mean = np.mean(np.array(daily_profile_error), axis=0), np.mean(np.array(fiftyhz_error), axis=0)
                np.savez(self.result_path_root + path + out_name,
                         pred_error_std=pred_std, pred_error_mean=pred_mean,
                         daily_profile_error_mean=dp_mean, fiftyhz_error_mean=fiftyhz_mean)
        print('Average and standard deviation calculated.')

    def special_weights(self, a, b):
        freq_a, time_idx_a = a[:3600], a[-1]
        freq_b, time_idx_b = b[:3600], b[-1]
        freq_a = np.concatenate((freq_a, [time_idx_a]))
        freq_b = np.concatenate((freq_b, [time_idx_b]))
        dist = np.linalg.norm(freq_a - freq_b) + (self.weight * np.linalg.norm(a[3600:-1] - b[3600:-1]))
        return dist

    def generate_train_and_test_chunks_2dim(self, data, points_per_chunk, pred_start_minute, n_tests=-1,
                                            train_start='2019-09-29 00:00:00', train_end='2020-03-31 23:59:59',
                                            test_start='2020-04-01 00:00:00', test_end='2020-06-30 23:59:59'):

        data = data.loc[train_start:test_end]
        train_chunks, mm = self.construct_chunks_2dim(data.loc[:train_end], points_per_chunk, pred_start_minute)
        test_chunks, _ = self.construct_chunks_2dim(data.loc[test_start:], points_per_chunk, pred_start_minute, test=True, mm=mm)

        # Calculate the start time for the init chunks and the number of chunks to predict
        init_start_minute = int((pred_start_minute - points_per_chunk / 60) % 60)
        chunks_to_predict = np.int(np.ceil(self.points_to_predict / points_per_chunk))

        # Remove nan chunks from test chunks and choose chunks with correct start time
        init_chunks = remove_nan_chunks(test_chunks, chunks_to_predict, drop_time_index=False)
        init_chunks = init_chunks.iloc[:-chunks_to_predict]
        init_chunks = init_chunks[init_chunks.index.minute == init_start_minute]

        # Randomly select n_tests init chunks. Otherwise select all chunks
        if (type(n_tests) == int) and (n_tests != -1):
            init_chunks = init_chunks.sample(n_tests, random_state=1)

        # Construct an ensemble of test time series for the prediction time interval
        ensemble_test = np.zeros((init_chunks.shape[0], self.points_to_predict))
        for i, ind in enumerate(init_chunks.int_index):
            ensemble_test[i] = test_chunks.values[ind + 1:ind + chunks_to_predict + 1].flatten()[:self.points_to_predict]

        # Drop integer index that was used to construct ensemble_test
        init_chunks.drop(columns=['int_index'], inplace=True)

        return train_chunks, init_chunks, ensemble_test, train_start, train_end

    def construct_chunks_2dim(self, time_series, chunk_length, start_time_within_hour=0, test=False, mm=None):
        """Cut a time series into a set of non-overlapping chunks of the same length. Start and end
        of the set are chosen in such a way that the set starts and ends at a given time within an hour.
        The chunk length should be a factor of 60 [Minutes]. """

        # Set start and end time of the chunk set
        start = time_series.index[0].floor('H') + DateOffset(minutes=int(start_time_within_hour))
        end = get_end(time_series.index[-1], start_time_within_hour, chunk_length)
        chunks = time_series.loc[start:end]
        new_dim_data = self.new_dim_data
        if new_dim_data is not None:
            new_dim_end = get_end(time_series.index[-1], start_time_within_hour, self.new_dim_chunk_length)
            new_dim_data = self.new_dim_data.loc[start:new_dim_end]

        return self.remain_2dim(chunks, chunk_length, new_dim_data, test, mm)

    @staticmethod
    def get_end(end, start_time_within_hour, new_dim_chunk_length):
        remove_sec = int(3600 / new_dim_chunk_length)
        end = end.ceil('H') + DateOffset(minutes=int(start_time_within_hour)) - DateOffset(seconds=remove_sec)
        return end

    def remain_2dim(self, chunks, chunk_length, new_dim_data, test, mm):
        # If chunk_length is larger than 1h there might be a remainder when cutting the time series into chunks...
        remainder = int(chunks.shape[0] % chunk_length)

        if remainder != 0:
            chunks = chunks.iloc[:-remainder]
        # Construct chunks as dataframe with time indices
        chunks = pd.DataFrame(data=chunks.values.reshape((chunks.shape[0] // chunk_length, chunk_length)),
                              index=chunks.index[::chunk_length])
        if self.new_dim_data is not None:
            chunks, mm = self.add_feature(chunks, new_dim_data, test, mm)
        return chunks, mm

    def add_feature(self, chunks, new_dim_data, test, mm):
        new_dim_data = new_dim_data.astype(np.float64)
        if not test:
            mm = MinMaxScaler()
            new_dim_data = pd.DataFrame(mm.fit_transform(new_dim_data.values.reshape(-1, 1)),
                                        index=new_dim_data.index)
        if test:
            new_dim_data = pd.DataFrame(mm.transform(new_dim_data.values.reshape(-1, 1)),
                                        index=new_dim_data.index)

        a = pd.DataFrame(
            data=new_dim_data.values.reshape((new_dim_data.shape[0] // self.new_dim_chunk_length, self.new_dim_chunk_length)),
            index=new_dim_data.index[::self.new_dim_chunk_length])
        chunks = pd.concat([chunks, a], axis=1)
        chunks.columns = list(range(chunks.shape[1]))
        return chunks, mm

    def nearest_neighbor_prediction_performance(self, train_chunks_full, init_chunks_full, points_per_chunk, points_to_predict,
                                                k_nearest_neighbors_f, k_nearest_neighbors_a, time_between_points=1,
                                                time_sensitive=True, return_error=False, constant_weights=False, n_jobs=1):
        """This function predicts the succesors of test chunks by searching for nearest neighbors
        in the set of training chunks. For each time t, the prediction is a weighted average
        of k(t) nearest neighbors. Thus, k_nearest_neighbors should be an array with length
        'points_to_predict', but it can also be a time-independent scalar value.
        NaN values are only allowed in the training data. """

        # Calculate the number of chunks to predict and the time between chunks
        chunks_to_predict = np.int(np.ceil(points_to_predict / points_per_chunk))
        time_between_chunks = time_between_points * points_per_chunk
        train_chunks = train_chunks_full.iloc[:, :self.points_to_predict]
        init_chunks = init_chunks_full.iloc[:, :self.points_to_predict]
        max_k = max(k_nearest_neighbors_f, np.max(k_nearest_neighbors_a))

        # Construct a training set without NaNs
        # Construct nearest neighbor finder (for max(k(t)) if k is time-dependent)

        if self.weight == 0:
            if time_sensitive:
                add_weighted_integer_time_index(train_chunks, time_between_chunks, time_weight=1000)
                add_weighted_integer_time_index(init_chunks, time_between_chunks, time_weight=1000)
            train_chunks_without_nan_test = remove_nan_chunks(train_chunks, chunks_to_predict)
            if time_sensitive:
                train_chunks.drop(columns=[train_chunks.shape[1] - 1], inplace=True)
            neighbor_finder = NearestNeighbors(n_neighbors=max_k, n_jobs=n_jobs)
            neighbor_finder.fit(train_chunks_without_nan_test.iloc[:-chunks_to_predict])
            print('Searching for neighbors...', end='\r')
            dists, nns = neighbor_finder.kneighbors(init_chunks)
        else:
            # Convert the timestamps of the chunks to integers with distance "time_weight"
            if time_sensitive:
                add_weighted_integer_time_index(train_chunks_full, time_between_chunks, time_weight=1000)
                add_weighted_integer_time_index(init_chunks_full, time_between_chunks, time_weight=1000)
            train_chunks_without_nan_test = remove_nan_chunks(train_chunks_full, chunks_to_predict)
            neighbor_finder = NearestNeighbors(n_neighbors=max_k, n_jobs=n_jobs, metric=self.special_weights)
            neighbor_finder.fit(train_chunks_without_nan_test.iloc[:-chunks_to_predict])
            # Find nearest neighbors and their distances using the euclidean norm
            print('Searching for neighbors...', end='\r')
            dists, nns = neighbor_finder.kneighbors(init_chunks_full)
        print('Making predictions...', end='\r')
        nns = np.array(train_chunks_without_nan_test.index)[nns]

        # Drop integer time index and additional feature again MUST BE CHANGED TO PREDICT ADDITIONAL FEATURE
        # if time_sensitive:
        #     cols = [i for i in range(points_to_predict, train_chunks.shape[1])]
        #     train_chunks.drop(columns=cols, inplace=True)
        #     init_chunks.drop(columns=cols, inplace=True)

        # Construct prediction by concatenating subsequent chunks and nearest neighbors
        pred_ind = range(1, chunks_to_predict + 1)
        predictions = np.concatenate([train_chunks.values[nns + i][:points_to_predict] for i in pred_ind],
                                     axis=-1)
        predictions = predictions[:, :, :points_to_predict]

        # Construct weights from distances for each time step...
        weights = weights_from_distances(dists, points_to_predict,
                                         k_nearest_neighbors_f, constant_weights=constant_weights)
        # Average over nearest neighbors
        predictions_mean_f = np.average(predictions, axis=1, weights=weights)
        weights = weights_from_distances(dists, points_to_predict,
                                         k_nearest_neighbors_a, constant_weights=constant_weights)
        predictions_mean_a = np.average(predictions, axis=1, weights=weights)
        if return_error:
            return predictions_mean_f, predictions_mean_a, predictions.std(axis=1)
        return predictions_mean_f, predictions_mean_a
