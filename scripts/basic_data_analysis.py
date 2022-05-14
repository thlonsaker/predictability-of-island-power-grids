import numpy as np
from statsmodels.tsa.stattools import acf
from extra_functions import *
from standards import Standards
from regions import *


class DataAnalysis:
    plt.rc('text', usetex=True)
    out_path = '/Users/thorbjornlundonsaker/workspace/Master/results/time_scales/'
    in_path = '/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/'
    areas = [[FaroeIslands(), None], [Ireland(), BalearicIslands()], [Iceland(), Nordic()]]
    color = Standards().patterns
    figsize = (15, 12)
    fontsize = 28
    ticks = 22
    label = 24

    def __init__(self, info):
        """info of type: {'area1': (start_time, end_time), 'area2': ...}.
        Values can be of type: (str, str), (int, int), (), int"""
        self.info = info
        print('Loading data...', end='\r')
        self.data = []
        # self.areas[0][0] = FaroeIslands()
        # self.areas[-1][-1] = Nordic()
        for i in range(3):
            lst = []
            for j in range(2):
                region = self.areas[i][j]
                if region is None:
                    lst.append(None)
                    continue
                lst.append(self.specific_data(region.name))
            self.data.append(lst)

    def specific_data(self, area):
        file_area = underscore(area)
        file = self.in_path + f'{file_area}/{file_area}_50.pkl'
        if self.info[area] == ():
            return pd.read_pickle(file)
        if type(self.info[area]) is int:
            return pd.read_pickle(file).iloc[:self.info[area] * 86400]
        start, end = self.info[area]
        if type(start) and type(end) is str:
            return pd.read_pickle(file).loc[start:end]
        return pd.read_pickle(file).iloc[start:end]

    def plot_daily_profile(self, save_fig=False, save_text=False):
        cols = len(self.areas[0])
        rows = len(self.areas)
        fig, axs = plt.subplots(rows, cols, figsize=self.figsize, sharey='row')
        for i in range(rows):
            for j in range(cols):
                area = self.areas[i][j]
                data = self.data[i][j]
                if area is None:
                    axs[i][j].set_visible(False)
                    continue
                print('Plotting daily profile of {}...'.format(area.name), end='\r')
                data_day_profile = data.groupby(by=data.index.time).mean()
                axs[i, j].plot(np.arange(24 * 60 * 60) / 3600., data_day_profile.values, c=area.color)
                if i == 2:
                    axs[i, j].set_xlabel('Time [h]', fontsize=self.label)
                if j == 0:
                    axs[i, j].set_ylabel('Average frequency [Hz]', fontsize=self.label)
                axs[i, j].set_title(self.two_line_title(area, data))
                # axs[i, j].set_title(area.name + '\n Time period: ' + show_period(data, drop_nans=True), fontsize=22)
                axs[i, j].set_xticks(np.arange(24), minor=True)
                axs[i, j].set_xticks(np.arange(0, 25, 4), minor=False)
                axs[i, j].tick_params(axis='both', which='both', labelsize=self.ticks)
                axs[i, j].grid(axis='x', which='both')
                if save_text is True:
                    np.savetxt(self.out_path + '{}/{}_profile.txt'.format(area, area), data_day_profile)
        print('Plot of daily profile done.')
        plt.suptitle('Daily profile', fontsize=self.fontsize)
        plt.tight_layout()
        if save_fig is True:
            plt.savefig('plots/daily_profile.pdf')
        plt.show()

    @staticmethod
    def two_line_title(region, data):
        time = show_period(data, drop_nans=True)
        return r'{\fontsize{28pt}{3em}\selectfont{}' + region.name + '}' + '\n' + r'{\fontsize{24pt}{3em}\selectfont{}' + time + '}'

    def plot_daily_std(self, save_fig=False):
        cols = len(self.areas[0])
        rows = len(self.areas)
        fig, axs = plt.subplots(rows, cols, figsize=self.figsize, sharey='row')
        for i in range(rows):
            for j in range(cols):
                area = self.areas[i][j]
                data = self.data[i][j]
                if area is None:
                    axs[i][j].set_visible(False)
                    continue
                print(f'Plotting daily standard deviation of {area.name}...', end='\r')
                time_index_daily = data.index.time
                data_day_std = data.groupby(by=time_index_daily).std()

                axs[i, j].plot(np.arange(24 * 60 * 60) / 3600., data_day_std.values, c=area.color)
                if i == 2:
                    axs[i, j].set_xlabel('Time [h]', fontsize=self.label)
                if j == 0:
                    axs[i, j].set_ylabel('Standard deviation [Hz]', fontsize=self.label)
                axs[i, j].set_title(self.two_line_title(area, data))
                axs[i, j].set_xticks(np.arange(24), minor=True)
                axs[i, j].set_xticks(np.arange(0, 25, 4), minor=False)
                axs[i, j].tick_params(axis='both', which='both', labelsize=self.ticks)
                axs[i, j].grid(axis='x', which='both')

        print('Plot of daily standard deviation done.')
        plt.suptitle('Daily standard deviation', fontsize=self.fontsize)
        plt.tight_layout()
        if save_fig is True:
            plt.savefig('plots/daily_std_deviation.pdf')
        plt.show()

    def plot_hourly_std(self, save_fig=False):
        cols = len(self.areas[0])
        rows = len(self.areas)
        fig, axs = plt.subplots(rows, cols, figsize=self.figsize, sharey='row')
        for i in range(rows):
            for j in range(cols):
                area = self.areas[i][j]
                data = self.data[i][j]
                if area is None:
                    axs[i][j].set_visible(False)
                    continue
                print(f'Plotting hourly standard deviation of {area.name}...', end='\r')
                time_index_hourly = [time.replace(hour=0) for time in data.index.time]
                data_hour_std = data.groupby(by=time_index_hourly).std()
                axs[i, j].plot(np.arange(3600) / 60., data_hour_std.values, c=area.color)
                if i == 2:
                    axs[i, j].set_xlabel('Time [Min]', fontsize=self.label)
                if j == 0:
                    axs[i, j].set_ylabel('Standard deviation [Hz]', fontsize=self.label)
                axs[i, j].grid(axis='x')
                axs[i, j].set_title(self.two_line_title(area, data))
                axs[i, j].tick_params(axis='both', which='both', labelsize=self.ticks)
        print('Plot of hourly standard deviation done.')
        plt.suptitle('Hourly standard deviation', fontsize=self.fontsize)
        plt.tight_layout()
        if save_fig is True:
            plt.savefig('plots/hourly_std_deviation.pdf')
        plt.show()

    def plot_acf(self, save_fig=False, day=True, minute=False, period=20):
        cols, rows, axis, lw = len(self.areas[0]), len(self.areas), 'both', 3
        fig, axs = plt.subplots(rows, cols, figsize=self.figsize, sharey='row')
        areas = self.areas
        data_s = self.data
        if day and period > 6:
            # areas[-1][-1] = areas[0][0]
            # data_s[-1][-1] = data_s[0][0]
            data_s.pop(0)
            areas.pop(0)
            rows, cols, lw = 2, 2, 1
            fig, axs = plt.subplots(rows, cols, figsize=(14, 8), sharey='row')
        for i in range(rows):
            for j in range(cols):
                if areas[i][j] is None:
                    axs[i][j].set_visible(False)
                    continue
                area = areas[i][j]
                data = data_s[i][j]
                size = data.shape[0]
                if minute and not day:
                    day = False
                    idxs = 60
                    nlags = idxs * period
                    time, points = 'min', 5
                elif day and not minute:
                    idxs = 3600 * 24
                    nlags = idxs * period
                    time, points = 'd', 4
                else:
                    return None
                print(f'Plotting acf of {area.name}...', end='\r')
                if size < nlags:
                    nlags = size
                acf_result = acf(data.values, nlags=nlags, fft='True', missing='drop')
                axs[i, j].plot(np.arange(acf_result.shape[0]) / idxs, acf_result, lw=lw, c=area.color)
                axs[i, j].set_ylim([-0.1, 0.35])
                if time == 'min':
                    axis = 'major'
                    axs[i, j].set_ylim([-0.1, 0.9])
                axs[-1, j].set_xlabel(f'Lag [{time}]', fontsize=self.label)
                if j == 0:
                    axs[i, j].set_ylabel('Autocorrelation', fontsize=self.label)
                axs[i, j].set_title(self.two_line_title(area, data))
                axs[i, j].set_xticks(np.arange(period + 1), minor=True)
                axs[i, j].set_xticks(np.arange(0, period + 1, period // points), minor=False)
                axs[i, j].tick_params(axis='both', which='both', labelsize=self.ticks)
                axs[i, j].grid(axis='x', which=axis)
                axs[i, j].locator_params(axis='y', nbins=4)
        print('Plot of acf done.')
        plt.suptitle('Autocorrelation function', fontsize=self.fontsize)
        plt.tight_layout()
        if save_fig is True:
            plt.savefig(f'plots/long_autocorr_func_day{day}_minute{minute}.pdf')
        plt.show()
