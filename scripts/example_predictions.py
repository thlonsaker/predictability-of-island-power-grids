import matplotlib.pyplot as plt

from pattern_prediction import *
from predictor import Predictor
from regions import *
from extra_functions import space


class ExamplePrediction(Predictor):

    def __init__(self, add_new_data=False, size_dependent=False, region=None):
        super().__init__()
        # TODO Move/save performance to examples instead of optimization
        self.size_dependent = size_dependent
        self.filename_test_dates = 'test_dates.pkl'
        self.filename = 'optimization/{}/{}/example_predictions_data.pkl'
        self.filename_full_size = 'optimization/{}/example_predictions_data.pkl'
        self.color = self.base.prediction()
        self.best, self.worst = None, None
        if add_new_data:
            self.load_and_create(region)

    @staticmethod
    def dates(path):
        stats = pickle.load(open(path, "rb"))
        start, end = stats['Period start'], stats['Period end']
        return start, end

    def load_and_create(self, region):
        if region is not None:
            self.set_regions(regions=region)
        for region in self.areas:
            print(f'Extracting information from {region.name}...')
            reg_name = underscore(region.name)
            data_path = self.data_path_root.format(reg_name, reg_name)
            main_data = pd.read_pickle(data_path)
            samples, times, period_info = region(self.validation, self.size_dependent)
            if self.size_dependent:
                self.load_size_dependent(times, reg_name, samples, main_data, 3600, 0)
            else:
                self.best = {'idx': 0, 'value': 1000, 'ensemble_pred': list(),
                             'ensemble_pred_std': list(), 'dp': list(), 'ensemble_test': list()}
                self.worst = {'idx': 0, 'value': 0, 'ensemble_pred': list(),
                              'ensemble_pred_std': list(), 'dp': list(), 'ensemble_test': list()}
                self.load_full_size_data(reg_name, main_data, 3600, 0, period_info)
            print('New plots added.')

    def load_full_size_data(self, reg_name, main_data, points_per_chunk, pred_start_minute, period_info):

        knn_npz = np.load(self.result_path_root + f'optimization/{reg_name}/k_opt_star'
                                               f't0_cweightsFalse_win_size'
                                               f'3600_test1.npz')
        k_opt = knn_npz['k_opt']
        print(f'Loading... full size {space(reg_name)}:', end='\r')
        data_path = self.data_path_root.format(reg_name, reg_name)

        data_tuple = generate_train_and_test_chunks(main_data, points_per_chunk,
                                                    pred_start_minute, self.points_to_predict,
                                                    size_dependent=self.size_dependent,
                                                    validation=self.validation, **period_info)
        train_chunks, init_chunks, ensemble_test, train_start, train_end = data_tuple

        print('Running predictions...', end='\r')
        ensemble_pred, ensemble_pred_std = nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk,
                                                                       self.points_to_predict,
                                                                       k_opt, return_error=True, constant_weights=False)
        print('Computing daily profile...', end='\r')
        daily_profile_pred = daily_profile_prediction(data_path, init_chunks, self.points_to_predict,
                                                      train_start, train_end, main_data=main_data)
        print('calculating RMSE...', end='\r')
        rmse = calc_rmse(ensemble_pred, ensemble_test) / calc_rmse(50, ensemble_test)
        i_min_idx = np.argmin(rmse, axis=0)
        i_min_val = np.min(rmse, axis=0)
        i_max_idx = np.argmax(rmse, axis=0)
        i_max_val = np.max(rmse, axis=0)
        if i_min_val < self.best['value']:
            self.update_dictionary(self.best, i_min_val, i_min_idx, daily_profile_pred[i_min_idx],
                                   ensemble_pred[i_min_idx], ensemble_pred_std[i_min_idx], ensemble_test[i_min_idx])
        if i_max_val > self.worst['value']:
            self.update_dictionary(self.worst, i_max_val, i_max_idx, daily_profile_pred[i_max_idx],
                                   ensemble_pred[i_max_idx], ensemble_pred_std[i_max_idx], ensemble_test[i_max_idx])
        pickle.dump((self.best, self.worst),
                    open(self.result_path_root + self.filename_full_size.format(reg_name), 'wb'))

    def load_size_dependent(self, region, reg_name, samples, main_data,
                            points_per_chunk, pred_start_minute):
        for times in region.periods:
            self.best = {'idx': 0, 'value': 1000, 'ensemble_pred': list(),
                         'ensemble_pred_std': list(), 'dp': list(), 'ensemble_test': list()}
            self.worst = {'idx': 0, 'value': 0, 'ensemble_pred': list(),
                          'ensemble_pred_std': list(), 'dp': list(), 'ensemble_test': list()}
            week, month, period = period_filename(times)
            test_start, test_end = self.dates(self.result_path_root + f'optimization/{reg_name}/test_dates.pkl')
            train_starts, train_ends = self.dates(self.result_path_root + f'optimization/{reg_name}/{period}/statistics.pkl')
            for sample in range(1, samples + 1):
                knn_npz = np.load(self.result_path_root + f'optimization/{reg_name}/{period}/k_opt_star'
                                                       f't0_cweightsFalse_win_size'
                                                       f'3600_test{sample}.npz')
                k_opt = knn_npz['k_opt']
                train_start, train_end = train_starts[sample - 1], train_ends[sample - 1]
                print(f'Loading... period: {time_to_print(week, month, period)}, sample: {sample}', end='\r')
                data_path = self.data_path_root.format(reg_name, reg_name)
                train_chunks, init_chunks, ensemble_test, _, _ = generate_train_and_test_chunks(main_data,
                                                                                                points_per_chunk,
                                                                                                pred_start_minute,
                                                                                                self.points_to_predict,
                                                                                                size_dependent=self.size_dependent,
                                                                                                validation=self.validation,
                                                                                                train_start=train_start,
                                                                                                train_end=train_end,
                                                                                                test_start=test_start,
                                                                                                test_end=test_end)

                ensemble_pred, ensemble_pred_std = nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk,
                                                                               self.points_to_predict,
                                                                               k_opt, return_error=True, constant_weights=False)

                daily_profile_pred = daily_profile_prediction(data_path, init_chunks, self.points_to_predict,
                                                              train_start, train_end, main_data=main_data)

                rmse = calc_rmse(ensemble_pred, ensemble_test) / calc_rmse(50, ensemble_test)
                i_min_idx = np.argmin(rmse, axis=0)
                i_min_val = np.min(rmse, axis=0)
                i_max_idx = np.argmax(rmse, axis=0)
                i_max_val = np.max(rmse, axis=0)
                if i_min_val < self.best['value']:
                    self.update_dictionary(self.best, i_min_val, i_min_idx, daily_profile_pred[i_min_idx],
                                           ensemble_pred[i_min_idx], ensemble_pred_std[i_min_idx], ensemble_test[i_min_idx])
                if i_max_val > self.worst['value']:
                    self.update_dictionary(self.worst, i_max_val, i_max_idx, daily_profile_pred[i_max_idx],
                                           ensemble_pred[i_max_idx], ensemble_pred_std[i_max_idx], ensemble_test[i_max_idx])
            pickle.dump((self.best, self.worst),
                        open(self.result_path_root + self.filename.format(reg_name, period), 'wb'))

    @staticmethod
    def update_dictionary(dic, val, idx, daily_profile_pred, ensemble_pred, ensemble_pred_std, ensemble_test):
        dic['value'] = val
        dic['idx'] = idx
        dic['dp'] = daily_profile_pred
        dic['ensemble_pred'] = ensemble_pred
        dic['ensemble_pred_std'] = ensemble_pred_std
        dic['ensemble_test'] = ensemble_test

    def best_prediction(self, region='all', period=None, savefig=False):
        if region == 'all':
            self.plot_all(0, period)
        else:
            self.plot(0, region, period)
        if savefig:
            plt.savefig('plots/best_prediction.pdf', bbox_inches='tight')
        plt.show()

    def plot(self, idx, region, time):
        reg_name = underscore(region)
        title = 'Best'
        week, month, period = 0, 0, 0
        if idx == 1:
            title = 'Worst'
        if time is not None:
            week, month, period = period_filename(time)
            data = pickle.load(open(self.result_path_root + self.filename.format(reg_name, period), "rb"))[idx]
        else:
            data = pickle.load(open(self.result_path_root + self.filename_full_size.format(reg_name), "rb"))[idx]
        ensemble_test = data['ensemble_test']
        daily_profile_pred = data['dp']
        ensemble_pred_std = data['ensemble_pred_std']
        ensemble_pred = data['ensemble_pred']
        plt.fill_between(self.one_pred_index, ensemble_pred + ensemble_pred_std,
                         ensemble_pred - ensemble_pred_std, alpha=0.2, facecolor=self.base.one_pred, label='Prediction error')
        plt.plot(self.one_pred_index, ensemble_test, lw=1, label='Test series', c=self.base.test_series)
        plt.plot(self.one_pred_index, daily_profile_pred, lw=1, label='Daily profile', c=self.base.daily_profile)
        plt.plot(self.one_pred_index, ensemble_pred, label='WNN prediction', lw=1.5, c=self.base.one_pred)
        plt.tick_params(axis='both', which='major', labelsize=14)
        if time is not None:
            plt.title(f'{title} prediction {space(region)}, {time_to_print(week, month, period)}', fontsize=self.base.title)
        else:
            plt.title(f'{title} prediction {space(region)}', fontsize=self.base.title)
        plt.legend()
        plt.xlabel('Time $\Delta t$ [min]', fontsize=self.base.xlabel)
        plt.ylabel('Frequency [Hz]', fontsize=self.base.ylabel)
        plt.xticks([0, 15, 30, 45, 60])
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

    def worst_prediction(self, region='all', period=None, savefig=False):
        if region == 'all':
            self.plot_all(1, period)
        else:
            self.plot(1, region, period)
        if savefig:
            plt.savefig('plots/worst_prediction.pdf', bbox_inches='tight')
        plt.show()

    def standard_deviations(self, best=True, period='full'):
        title, idx = 'Best', 0
        if not best:
            title, idx = 'Worst', 1
        print(f'{title} predictions standard deviations')
        for region in Regions.__subclasses__():
            reg_name = underscore(region.name)
            if period != 'full' and isinstance(period, tuple):
                week, month, period = period_filename(period)
                data = pickle.load(open(self.result_path_root + self.filename.format(reg_name, period), "rb"))[idx]
            else:
                data = pickle.load(open(self.result_path_root + self.filename_full_size.format(reg_name), "rb"))[idx]
            ensemble_pred_std = data['ensemble_pred_std']
            print(f'Average STD {region.name}: {round(ensemble_pred_std.mean() * 1000, 0)} mHz')

    def plot_all(self, idx, time):
        title = 'Best'
        fig, axs = plt.subplots(3, 2, figsize=(14, 11))
        for i in range(3):
            for j in range(2):
                region = self.areas_grid[i][j]
                if region is None:
                    axs[i][j].set_visible(False)
                    continue
                reg_name = underscore(region.name)
                if idx == 1:
                    title = 'Worst'
                if time is not None:
                    week, month, period = period_filename(time)
                    data = pickle.load(open(self.result_path_root + self.filename.format(reg_name, period), "rb"))[idx]
                else:
                    data = pickle.load(open(self.result_path_root + self.filename_full_size.format(reg_name), "rb"))[idx]
                ensemble_test = data['ensemble_test']
                daily_profile_pred = data['dp']
                ensemble_pred_std = data['ensemble_pred_std']
                ensemble_pred = data['ensemble_pred']
                axs[i, j].fill_between(self.one_pred_index, ensemble_pred + ensemble_pred_std,
                                       ensemble_pred - ensemble_pred_std, alpha=0.35,
                                       facecolor=self.base.one_pred, label='Prediction error')
                axs[i, j].plot(self.one_pred_index, ensemble_test, lw=0.9, label='Test series', c=self.base.test_series)
                axs[i, j].plot(self.one_pred_index, daily_profile_pred, lw=1, label='Daily profile', c=self.base.daily_profile)
                axs[i, j].plot(self.one_pred_index, ensemble_pred, label='WNN prediction', lw=1.2, c=self.base.one_pred)
                if time is not None:
                    axs[i, j].set_title(f'{region.name}, {time_to_print(week, month, period)}',
                                        fontsize=self.base.title)
                else:
                    axs[i, j].set_title(region, fontsize=self.base.title)
                axs[-1, j].set_xlabel('Time $\Delta t$ [min]', fontsize=self.base.title)
                axs[i, 0].set_ylabel('Frequency [Hz]', fontsize=self.base.ylabel)
                axs[i, j].set_xticks([0, 15, 30, 45, 60])
                axs[i, j].tick_params(axis='both', which='major', labelsize=self.base.xlabel)
                axs[i, j].grid(axis='x')
        plt.suptitle(f'{title} prediction', fontsize=22, y=0.93)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        leg = axs[0, 0].legend(ncol=2, bbox_to_anchor=(1.1, -3.35, 0.5, 0.5), fontsize=17)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0 * legobj.get_linewidth())
