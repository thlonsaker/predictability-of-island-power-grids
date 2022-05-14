import matplotlib.pyplot as plt
import numpy as np

from predictor import *
import os


class Performance(Predictor):

    def __init__(self, size_dependent=False):
        super().__init__()
        self.size_dependent = size_dependent
        self.weights = None

    def plot_k(self, region='all', relative=False, savefig=False):
        # TODO fix title and y-label for relative=False
        """Method to plot the adaptive-k for all wanted regions, full time series.
        If relative is set to True, the adaptive-k is plotted relative to the
        maximum k, which reflects the daily profile."""
        self.set_regions(region)
        plt.figure()
        k = 1
        for region in self.areas:
            path = self.result_path_root + f'optimization/{underscore(region.name)}/' + self.k_opt_archive.format(0, 3600, 1)
            data = np.load(path)
            k_opt = data['k_opt']
            visible = True
            if relative:
                if region.time_sensitive:
                    k = data['size']
                else:
                    visible = False
            plt.plot(self.one_pred_index, k_opt / k, label=f'{region.name}'*visible,
                     c=region.color, visible=visible, linewidth=2)
        plt.xticks([0, 15, 30, 45, 60], fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='x')
        # plt.title('Optimal number of neighbors $k_{opt}$ relative to\nmaximum possible neighbors $K$', fontsize=14)
        plt.xlabel('Time $\Delta t$ [min]', fontsize=14)
        plt.ylabel('$Adaptive$-$k$ / $K$', fontsize=14)
        plt.subplots_adjust(bottom=0.05)
        leg = plt.legend(ncol=2, bbox_to_anchor=(0.3, -0.68, 0.5, 0.5), fontsize=14)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        plt.tight_layout()
        if savefig:
            plt.savefig(f'plots/adaptive_k_relative{relative}.pdf')
        plt.show()

    def plot_rmse_relative_Ireland(self, region=Ireland(), feature='all', k='adaptive', specific_weight=1,
                                   start=0, end=60, best=False, scale=False, savefig=False):
        region.set_additional_features(feature)
        s = 'not_scaled'
        if scale:
            s = 'scaled'
        if k != 'adaptive' and k != 'fixed':
            return f'ERROR: --{k}-- not a type of k.'
        null_file = self.result_path_root + f'eval_prediction/{region}/' + self.out_file.format(0, 3600, 1)
        best_rmse, best_pred, best_feature, = 10000, None, None
        if len(feature) != 0:
            null_data = np.load(null_file)
            pred_zero = null_data[f'pred_error_{k}_k'][60 * start:60 * end]
        else:
            return
        for feature in region.additional_features:
            file = self.result_path_root + f'eval_prediction/{region}/2_dim/{feature}/{s}/' + \
                   self.out_file.format(0, 3600, specific_weight)
            data = np.load(file)
            pred = data[f'pred_error_{k}_k'][60 * start:60 * end]
            if not best:
                plt.plot(self.one_pred_index[60 * start:60 * end], pred / pred_zero,
                         label=f'Extended WNN prediction, {space(feature)}', lw=1, c=self.base.one_pred)
            rmse_sum = pred.sum()
            if rmse_sum < best_rmse:
                best_rmse = rmse_sum
                best_feature = feature
                best_pred = pred
        zero_rmse_sum = pred_zero.sum()
        if zero_rmse_sum < best_rmse:
            best_rmse = zero_rmse_sum
            best_feature = 'NO FEATURE'
        if best:
            plt.plot(self.one_pred_index[60 * start:60 * end], best_pred / pred_zero,
                     label=f'WNN prediction with {space(best_feature)}', lw=0.5)
        plt.plot(self.one_pred_index[60 * start:60 * end], pred_zero / pred_zero, lw=1.5, c='k', label=f'Original WNN prediction')
        plt.xticks([int(i) for i in range(start, end + 1, (end - start) // 4)], fontsize=14)
        plt.grid(axis='x')
        plt.yticks(fontsize=14)
        plt.title(region, fontsize=14)
        print(f'Best feature: {space(best_feature)} with rmse {best_rmse} \n relative rmse: {best_rmse / zero_rmse_sum}')
        plt.xlabel('Time $\Delta t$ [min]', fontsize=14)
        plt.ylabel('RMSE$(\hat f)$ / RMSE$(f_{wnn})$', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if savefig:
            plt.savefig(f'plots/{k}_relative_rmse.pdf')
        plt.show()

    def plot_rmse_relative(self, k='adaptive', region=BalearicIslands(), features='0',
                           weights='0', start=0, end=60, savefig=False):

        if isinstance(features, str) or isinstance(weights, str):
            return 'Enter features and weights as lists.'
        if k != 'adaptive' and k != 'fixed':
            return f'ERROR: --{k}-- not a type of k.'
        null_file = self.result_path_root + f'eval_prediction/{underscore(region.name)}/' + self.out_file.format(0, 3600, 1)
        err = f'pred_error_{k}_k'
        original = np.load(null_file)[err][60 * start:60 * end]
        colors = ['g', 'b', 'y']
        for i, (feature, w) in enumerate(zip(features, weights)):
            p = np.load(self.result_path_root + f'eval_prediction/{underscore(region.name)}/2_dim/'
                        f'{feature}/not_scaled/performance_start0_cweightsFalse_win_size3600_test{w}.npz')
            error = p[err][60 * start:60 * end]
            print(f'Weight: {w} - relative error: {error.sum() / original.sum()}')
            plt.plot(self.one_pred_index[60 * start:60 * end], error / original,
                     label=f'{space(feature)}, $\\beta$ = {w}', lw=1, c=colors[i])
        # plt.ylim((0.955, 1.07))
        plt.plot(self.one_pred_index[60 * start:60 * end], original / original,
                 lw=1.5, c='k', label=f'Original WNN prediction')
        plt.xticks([int(i) for i in range(start, end + 1, (end - start) // 4)], fontsize=14)
        plt.grid(axis='x')
        plt.title(region, fontsize=14)
        plt.xlabel('Time $\Delta t$ [min]', fontsize=14)
        plt.ylabel('RMSE$(\hat f)$ / RMSE$(f_{wnn})$', fontsize=14)
        plt.yticks(fontsize=14)
        leg = plt.legend(fontsize=12)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)
        plt.tight_layout()
        if savefig:
            plt.savefig(f'plots/{region}_relative.pdf')
        plt.show()

    def plot_rmse_all_weights(self, region, feature='', scaled=True, start=0, end=60):
        s = 'not_scaled'
        if scaled:
            s = 'scaled'
        files = glob.glob(self.result_path_root + f'eval_prediction/{region}/2_dim/{feature}/{s}/' + '*.npz')
        files = np.sort(files)
        best_rmse = 1000
        best_weight = 0
        for i, file in enumerate(files):
            data = np.load(file)
            pred_f = data['pred_error_fixed_k'][60 * start:60 * end]
            label = file[128 + len(region) + len(feature) + len(s):-4]
            plt.plot(self.one_pred_index[60 * start:60 * end], pred_f, label=f'fixed-k WNN {label}', lw=0.5)
            rmse_sum = pred_f.sum()
            if rmse_sum < best_rmse:
                best_rmse = rmse_sum
                best_weight = label
        plt.xticks([int(i) for i in range(start, end + 1, (end - start) // 4)])
        plt.grid(axis='x')
        plt.title(f'RMSE for {region} \n New feature: {space(feature)}', fontsize=14)
        print(f'Best weight: {best_weight} with rmse {best_rmse}')
        plt.xlabel('Time $\Delta t$ [min]', fontsize=14)
        plt.ylabel('RMSE$(\hat f)$ [Hz]', fontsize=14)
        # axs[1,0].set_ylabel(r'RMSE$(f_p)$ / RMSE$(f_d)$', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def performance(self, regions, periods=None, test=False):
        self.set_regions(regions)
        self.set_periods(periods)
        for region in self.areas:
            print(f'Running performance for {region.name}...')
            reg_name = underscore(region.name)
            data_path = self.data_path_root.format(reg_name, reg_name)
            main_data = pd.read_pickle(data_path)
            samples, times, period_info = region(self.validation, self.size_dependent)
            for time in times:
                if periods is not None:
                    if time not in region.periods:
                        break
                week, month, period = period_filename(time)
                print(space(period))
                file, test_file = '', ''
                if self.size_dependent:
                    path = self.result_path_root + f'optimization/{reg_name}/{period}/statistics.pkl'
                    file = pickle.load(open(path, "rb"))
                    test_file = pickle.load(open(self.result_path_root + f'optimization/{reg_name}/test_dates.pkl', "rb"))
                for points_per_chunk in self.points_per_chunk_s:
                    for pred_start_minute in self.pred_start_minute_s:
                        dp_starts, dp_ends = [], []
                        week, month, period = period_filename(time)
                        for sample in range(1, samples + 1):
                            print(f'Period: {time_to_print(week, month, period)}, sample: {sample}', end='\n')
                            data_path = self.data_path_root.format(reg_name, reg_name)
                            if self.size_dependent:
                                period_info = new_period_info(sample, week, month, file, test_file)
                            print(period_info)
                            data_tuple = generate_train_and_test_chunks(main_data, points_per_chunk,
                                                                        pred_start_minute,
                                                                        self.points_to_predict,
                                                                        size_dependent=self.size_dependent,
                                                                        validation=self.validation,
                                                                        weeks=week, months=month, sample=sample,
                                                                        **period_info, limit=0.9, area=reg_name)

                            train_chunks, init_chunks, ensemble_test, dp_s, dp_e = data_tuple
                            if self.size_dependent:
                                in_path = self.result_path_root + f'optimization/{reg_name}/{period}/'
                                if test:
                                    knn_npz = np.load(in_path + 'average_k.npz')
                                else:
                                    knn_npz = np.load(in_path + self.k_opt_archive.format(pred_start_minute, points_per_chunk, sample))
                            else:
                                if test:
                                    knn_npz = np.load(self.result_path_root + f'optimization/{reg_name}/2_dim/adaptive_k_weight.npz')
                                else:
                                    knn_npz = np.load(self.result_path_root + f'optimization/{reg_name}/' + self.k_opt_archive.format(0,
                                                                                                                                      3600,
                                                                                                                                      1))
                            k_opt = knn_npz['k_opt']
                            k_opt_fixed = int(knn_npz['k_opt_fixed'])
                            daily_profile_pred = daily_profile_prediction(data_path, init_chunks, self.points_to_predict, dp_s, dp_e)

                            ensemble_pred_a = nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk,
                                                                          self.points_to_predict, k_opt, constant_weights=False,
                                                                          time_sensitive=region.time_sensitive)
                            ensemble_pred_f = nearest_neighbor_prediction(train_chunks, init_chunks, points_per_chunk,
                                                                          self.points_to_predict, k_opt_fixed, constant_weights=False,
                                                                          time_sensitive=region.time_sensitive)

                            fiftyhz_error = calc_rmse(50., ensemble_test.T)
                            daily_profile_error = calc_rmse(daily_profile_pred.T, ensemble_test.T)
                            pred_error_adaptive_k = calc_rmse(ensemble_pred_a.T, ensemble_test.T)
                            pred_error_fixed_k = calc_rmse(ensemble_pred_f.T, ensemble_test.T)
                            if test:
                                np.savez(self.result_path_root + f'eval_prediction/{reg_name}/test/{period}' +
                                         self.out_file.format(pred_start_minute, points_per_chunk, sample),
                                         fiftyHz_error=fiftyhz_error, daily_profile_error=daily_profile_error,
                                         pred_error_adaptive_k=pred_error_adaptive_k)

                            elif self.size_dependent:
                                np.savez(self.result_path_root + f'eval_prediction/{reg_name}/{period}/' +
                                         self.out_file.format(pred_start_minute, points_per_chunk, sample),
                                         fiftyHz_error=fiftyhz_error, daily_profile_error=daily_profile_error,
                                         pred_error_fixed_k=pred_error_fixed_k, pred_error_adaptive_k=pred_error_adaptive_k)
                                dp_starts.append(dp_s), dp_ends.append(dp_e)
                            else:
                                np.savez(self.result_path_root + f'eval_prediction/{reg_name}/' +
                                         self.out_file.format(pred_start_minute, points_per_chunk, sample),
                                         fiftyHz_error=fiftyhz_error, daily_profile_error=daily_profile_error,
                                         pred_error_fixed_k=pred_error_fixed_k, pred_error_adaptive_k=pred_error_adaptive_k)
                        p = self.result_path_root + f'eval_prediction/{reg_name}/{period}/statistics.pkl'
                        if self.size_dependent:  # and not os.path.exists(p):
                            stats = {'Sample': np.arange(1, samples + 1), 'Period start': dp_starts, 'Period end': dp_ends}
                            pickle.dump(stats, open(p, 'wb'))
            print(f'Performance is done for {region.name} for all periods.')
        print('Performance is done.')

    def performance_2dim(self, regions, feature, new_dim_chunk_length, weight=0, scale=False):
        self.set_regions(regions)
        self.scale = scale
        self.new_dim_chunk_length = new_dim_chunk_length
        self.weight = weight

        for region in self.areas:
            print(f'Running performance for {region.name}, {space(feature)}...')
            reg_name = underscore(region.name)
            data_path = self.data_path_root.format(reg_name, reg_name)
            main_data = pd.read_pickle(data_path)
            samples, times, period_info = region(self.validation, self.size_dependent)
            self.new_dim_data = pd.read_pickle(self.data_path_root.format(reg_name, feature)[:-7] + '.pkl')
            start_min = 0
            points_per_chunk = 3600
            path = self.result_path_root + f'optimization/{reg_name}/2_dim/{feature}/'

            weight = ''
            sample_list = np.arange(1, samples + 1)
            if self.weight != 0:
                weight = self.weight
            s = 'not_scaled'
            if self.scale:
                s = 'scaled'
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
                    self.inner_2dim(main_data, points_per_chunk, start_min, period_info, weight,
                                    data_path, region.time_sensitive, reg_name, feature, s, sample, period)

        print('Performance is done.')

    def inner_2dim(self, main_data, points_per_chunk, start_min, period_info, weight, data_path,
                   time_sensitive, area_name, feature, s, sample, period):

        if self.size_dependent:
            knn_npz = np.load(self.result_path_root + f'optimization/{area_name}/2_dim/{feature}/{period}/adaptive_k_weight{sample}.npz')
        else:
            knn_npz = np.load(self.result_path_root + f'optimization/{area_name}/2_dim/{feature}/{s}/adaptive_k_weight{weight}.npz')
        k_opt = knn_npz['k_opt']
        k_opt_fixed = int(knn_npz['k_opt_fixed'])
        w = self.weight
        if weight == 0:
            w = 1
            self.weight = knn_npz['weight']
        print(f'weight: {self.weight}, fixed k: {k_opt_fixed}, adaptive_k: {k_opt}')
        data_tuple = self.generate_train_and_test_chunks_2dim(main_data, points_per_chunk, start_min, **period_info)
        train_chunks, init_chunks, ensemble_test, dp_s, dp_e = data_tuple

        ensemble_pred_f, ensemble_pred_a = self.nearest_neighbor_prediction_performance(train_chunks, init_chunks, points_per_chunk,
                                                                                        self.points_to_predict, k_opt_fixed, k_opt,
                                                                                        time_sensitive=time_sensitive)

        daily_profile_pred = daily_profile_prediction(data_path, init_chunks, self.points_to_predict, dp_s, dp_e)
        fiftyhz_error = calc_rmse(50., ensemble_test.T)
        daily_profile_error = calc_rmse(daily_profile_pred.T, ensemble_test.T)
        pred_error_fixed_k = calc_rmse(ensemble_pred_f.T, ensemble_test.T)
        pred_error_adaptive_k = calc_rmse(ensemble_pred_a.T, ensemble_test.T)
        if self.size_dependent:
            np.savez(self.result_path_root + f'eval_prediction/{area_name}/2_dim/{feature}/{period}/' +
                     self.out_file.format(start_min, points_per_chunk, sample),
                     fiftyHz_error=fiftyhz_error, daily_profile_error=daily_profile_error,
                     pred_error_fixed_k=pred_error_fixed_k, pred_error_adaptive_k=pred_error_adaptive_k)
        else:
            np.savez(self.result_path_root + f'eval_prediction/{area_name}/2_dim/{feature}/{s}/new' +
                     self.out_file.format(start_min, points_per_chunk, w),
                     fiftyHz_error=fiftyhz_error, daily_profile_error=daily_profile_error,
                     pred_error_fixed_k=pred_error_fixed_k, pred_error_adaptive_k=pred_error_adaptive_k)

    def compute_average_k(self, plot=False, region='all', feature=None):
        """Compute average k value for prediction period of the samples."""

        filename = 'k_opt_start0_cweightsFalse_win_size3600_test{}.npz'
        out_name = 'average_k.npz'
        self.set_regions(region)
        for region in self.areas:
            print(f'Computing average k for {region.name}...', end='\r')
            samples, times, period_info = region(self.validation, self.size_dependent)
            reg_name = underscore(region.name)
            for times in region.periods:
                if times == (0, 6):
                    break
                week, month, period = period_filename(times)
                adaptive_k, weight = [], []
                path = f'optimization/{reg_name}/{period}/'
                for sample in range(1, samples + 1):
                    if feature is None:
                        data = np.load(self.result_path_root + path + filename.format(sample))
                        w = np.array(0)
                    else:
                        filename = 'adaptive_k_weight{}.npz'
                        path = f'optimization/{reg_name}/2_dim/{feature}/{period}/'
                        data = np.load(self.result_path_root + path + filename.format(sample))
                        w = data['weight']
                    adapt_k = data['k_opt']
                    adaptive_k.append(adapt_k), weight.append(w.flatten()[0])

                adaptive_k, weight = np.array(adaptive_k), np.array(weight)
                adaptive_k_mean = np.rint(np.mean(adaptive_k, axis=0)).astype(int)
                np.savez(self.result_path_root + path + out_name, k_opt=adaptive_k_mean, weight=round(weight.mean(), 1))
            if plot:
                plt.figure()
                for times in region.periods:
                    week, month, period = period_filename(times)
                    data = np.load(self.result_path_root + f'optimization/{reg_name}/{period}/' + out_name)
                    plt.plot(np.arange(3600), data['k_opt'], label=f'{time_to_print(week, month, period)}')
                plt.title(space(reg_name))
                plt.legend()
                plt.show()
        print('Average adaptive k calculated.')