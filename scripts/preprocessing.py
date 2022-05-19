import matplotlib.pyplot as plt
import pandas as pd

from help_funcs import *
from regions import *
from predictor import Predictor
from standards import Standards


class PreProcessing(Predictor):
    color = Standards().regions

    def __init__(self):
        super().__init__()

    def plot_pdf(self, xrange=500, savefig=False):
        fig, ax = plt.subplots(3, 2, figsize=(11, 13), sharey='row')
        for i in range(3):
            for j in range(2):
                region = self.areas_grid[i][j]
                if region is None:
                    ax[i, j].set_visible(False)
                    continue
                print(f'Plotting probability distribution function - {region}...', end="\r")
                clean_data = pd.read_pickle(region.data_path)
                nans = clean_data.isnull()
                idxs = clean_data.loc[nans].index
                data = pd.read_pickle(region.format_data)
                data = data.drop(index=idxs)
                data = data[~data.index.duplicated()]
                data.plot.kde(ax=ax[i, j], ind=np.arange(-xrange, xrange, 0.5), logy=True, label='frequency\ndistribution',
                                         c=self.color[region.name], linewidth=3, ylim=(1e-7, 2e-2))
                ax[-1, j].set_xlabel('f - f$^{ref}$ (mHz)', fontsize=17)
                ax[i, j].set_ylabel('PDF', fontsize=17)
                ax[i, j].tick_params(labelsize=15)
                ax[i, j].grid(axis='x')
                ax[i, j].set_title(region, fontsize=18)
                ax[i, j].legend(fontsize=14)
        plt.suptitle('Probability distribution function', fontsize=20)
        plt.subplots_adjust(hspace=0.2)
        plt.tight_layout()
        if savefig:
            plt.savefig('plots/new_pdf.pdf')
        print('plotting done.')
        plt.show()

    def plot_increment_analysis(self, res=10, savefig=False):
        fig, ax = plt.subplots(3, 2, figsize=(11, 13), sharey='row')
        for i in range(3):
            for j in range(2):
                region = self.areas_grid[i][j]
                if region is None:
                    ax[i, j].set_visible(False)
                    continue
                clean_data = pd.read_pickle(region.data_path)
                nans = clean_data.isnull()
                idxs = clean_data.loc[nans].index
                data = pd.read_pickle(region.format_data)
                data = data.drop(index=idxs)
                data = data[~data.index.duplicated()]
                df = data.diff()
                std = df.std()
                print(f'Plotting increment analysis - {region}...', end="\r")
                transform = transforms.offset_copy(ax[i, j].transData, fig=fig, x=0.0, y=50, units='points')
                x = np.arange(-15, 15, 0.01)
                ax[i, j].plot(x, scipy.stats.norm.pdf(x, 0, 1) / 100, label='gaussian')
                sns.kdeplot(df / std, ax=ax[i, j], log_scale=(False, True), label='1s', linestyle='None', marker='o',
                            markersize=3, transform=transform, gridsize=250, c=self.color[region.name])
                if res:
                    df_r = data.diff(int(res))
                    std_r = df_r.std()
                    sns.kdeplot(df_r / std_r, ax=ax[i, j], log_scale=(False, True), label=f'{res}s',
                                linestyle='None', marker='^', markersize=4, c='orange', gridsize=250)
                print(f'Plotting for {region} done.', end="\r")
                ax[-1, j].set_xlabel('$\Delta$f/$\sigma$ (mHz)', fontsize=17)
                if j == 0:
                    ax[i, j].set_ylabel('PDF $\Delta$f', fontsize=17)
                else:
                    ax[i, j].set_ylabel('', fontsize=1)
                ax[i, j].set_ylim(10e-8, 10e2)
                ax[i, j].set_xlim(-20, 20)
                ax[i, j].tick_params(axis='both', labelsize=15)
                ax[i, j].set_title(region, fontsize=17)
                ax[i, j].grid(axis='x')
                ax[i, j].legend(fontsize=15, markerscale=3)
        plt.suptitle('Increment analysis', fontsize=20, y=0.99)
        # plt.subplots_adjust(bottom=0.09, hspace=0.2)
        # leg = ax[0, 0].legend(ncol=3, bbox_to_anchor=(0.43, -2.8, 0.5, 0.5), fontsize=19)
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(4.0 * legobj.get_linewidth())
        #     legobj.set_markersize(5.0 * legobj.get_markersize())
        plt.tight_layout()
        if savefig:
            plt.savefig('plots/increment_analysis.pdf')
        plt.show()

    def plot_increment_analysis_test(self, res=None):
        # TODO use clean data instead of only formated data(?)
        fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharey='row')
        x = np.arange(-15, 15, 0.01)
        for i in range(3):
            for j in range(2):
                region = self.areas_grid[i][j]
                if region is None:
                    axs[i, j].set_visible(False)
                    continue
                print(f'Plotting increment analysis for {region.name}...', end="\r")
                data = pd.read_pickle(region.format_data)
                df = data.diff()
                std = df.std()
                transform = transforms.offset_copy(axs[i, j].transData, fig=fig, x=0.0, y=50, units='points')
                axs[i, j] = sns.kdeplot(df / std, log_scale=(False, True), label='1s', linestyle='None', marker='o',
                                        markersize=3, transform=transform, gridsize=100)
                axs[i, j].plot(x, scipy.stats.norm.pdf(x, 0, 1) / 100, label='gaussian')
                if res:
                    df_r = data.diff(int(res))
                    std_r = df_r.std()
                    axs[i, j] = sns.kdeplot(df_r / std_r, log_scale=(False, True), label=str(res) + 's',
                                            linestyle='None', marker='^', markersize=3, gridsize=100)
                print('Plotting done.', end="\r")
                axs[i, j].set_xlabel('$\Delta$f/$\sigma$ (mHz)', fontsize=15)
                axs[i, j].set_ylabel('PDF $\Delta$f', fontsize=15)
                axs[i, j].set_ylim(10e-8, 10e2)
                axs[i, j].set_xlim(-15, 15)
                plt.title(region.name)
                plt.grid(axis='x')
        plt.suptitle('Increment analysis', fontsize=18)
        plt.legend()
        plt.show()

    def plot_pdf_test(self, xrange=500):
        fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharey='all')
        for i in range(3):
            for j in range(2):
                region = self.areas_grid[i][j]
                if region is None:
                    axs[i, j].set_visible(False)
                    continue
                print(f'Plotting probability density function - {region.name}...', end="\r")
                clean_data = pd.read_pickle(region.data_path)
                nans = clean_data.isnull()
                idxs = clean_data.loc[nans].index
                data = pd.read_pickle(region.format_data).drop(index=idxs)
                axs[i, j] = data.plot.kde(ind=np.arange(-xrange, xrange, 0.5), logy=True, label=region.name,
                                          linewidth=2, ylim=(10 / data.shape[0], 2e-2))
                plt.subplots_adjust(bottom=0.15)  # TODO remove this?
                axs[i, j].set_xlabel('f - f$^{ref}$ (mHz)', fontsize=15)
                axs[i, j].set_ylabel('PDF', fontsize=15)
                axs[i, j].tick_params(labelsize=12)
                plt.grid(axis='x')
                plt.title(region.name, fontsize=15)
        plt.suptitle('PDF')
        print('plotting done.')
        plt.legend()
        plt.show()
