import pandas as pd
from extra_functions import underscore, plt
import matplotlib as mpl


class Regions:
    name = 'Regions'
    periods = [(1, 0), (2, 0), (0, 1), (0, 2), (0, 4), (0, 6)]
    samples = 25
    train_start = None
    train_end = None
    val_start = None
    val_end = None
    test_start = None
    test_end = None
    time_sensitive = True
    feature_chunk_length = None
    color = None
    additional_features = list()

    def __init__(self):
        self.data_path = f'/Users/thorbjornlundonsaker/workspace/Master/cleaned_data/' \
                         f'{underscore(self.name)}/{underscore(self.name)}_50.pkl'
        self.format_data = f'/Users/thorbjornlundonsaker/workspace/Master/format_data/' \
                           f'{underscore(self.name)}/{underscore(self.name)}.pkl'

    @classmethod
    def set_periods(cls, new_periods):
        """
        Overrides region parameters.
        :param new_periods: new input parameters
        """
        if new_periods is None:
            cls.periods = [(0, 0)]
        else:
            for p in new_periods:
                if p not in cls.periods:
                    raise ValueError(f'{p} not a valid period for {cls.name}')
            cls.periods = new_periods

    @classmethod
    def regions(cls):
        print(f'all subclasses in {cls.name}:')
        for region in cls.__subclasses__():
            print(region.name)

    @classmethod
    def set_additional_features(cls, feature):
        if feature != 'all':
            if not isinstance(feature, list):
                feature = [feature]
            for f in feature:
                if f not in cls.additional_features:
                    raise ValueError(f'{f} not a valid additional feature for {cls.name}')
            cls.additional_features = feature

    def __call__(self, validation=True, size_dependent=True):
        if validation:
            dictionary = {'train_start': self.train_start, 'train_end': self.train_end, 'test_start':
                          self.val_start, 'test_end': self.val_end}
        else:
            dictionary = {'train_start': self.train_start, 'train_end': self.train_end, 'test_start':
                          self.test_start, 'test_end': self.test_end}
        if size_dependent:
            return self.samples, self.periods, dictionary
        return 1, [(0, 0)], dictionary

    def __repr__(self):
        return self.name

    def get_info(self):
        data = pd.read_pickle(self.data_path).loc[self.train_start:self.test_end]
        full_size = data.shape[0]
        train = self.info(data, full_size, self.train_start, self.train_end, 'Train')
        val = self.info(data, full_size, self.val_start, self.val_end, 'Validation')
        test = self.info(data, full_size, self.test_start, self.test_end, 'Test')
        print(f'--------------- {self} ---------------\n{train}\n{val}\n{test}')

    @staticmethod
    def colorbar():
        fig, ax = plt.subplots(figsize=(2.7, 0.85))
        fig.subplots_adjust(bottom=0.5)
        colors = ["darkorange", "gold", "forestgreen"]
        nodes = [0.0, 0.5, 1.0]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label('Low-carbon generation [\%]', fontsize=12, fontweight='bold')
        ticks = [0, 20, 40, 60, 80, 100]
        cb1.set_ticks(ticks)
        cb1.set_ticklabels(ticks, fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/colorbar.pdf')
        fig.show()

    @staticmethod
    def info(data, full_size, start, end, string):
        size = data.loc[start:end].shape[0] / full_size
        return f'{string}: \n{start} - {end}   {round(size * 100, 1)}%'


class Nordic(Regions):
    name = 'Nordic'
    train_start = '2021-01-01 00:00:00'
    train_end = '2021-08-31 23:59:59'
    val_start = '2021-09-01 00:00:00'
    val_end = '2021-10-31 23:59:59'
    test_start = '2021-11-01 00:00:00'
    test_end = '2021-12-31 23:59:59'
    color = '#367e7f'


class Ireland(Regions):
    name = 'Ireland'
    train_start = '2021-11-04 17:00:00'
    train_end = '2022-01-18 23:59:59'
    val_start = '2022-01-19 00:00:00'
    val_end = '2022-02-05 23:59:59'
    test_start = '2022-02-06 00:00:00'
    test_end = '2022-02-22 23:59:59'
    periods = [(1, 0), (2, 0), (0, 1), (0, 2)]
    feature_chunk_length = [2, 4]
    additional_features = ['onshore_wind', 'wind_generation']
    color = '#c75b23'


class BalearicIslands(Regions):
    name = 'Balearic Islands'
    train_start = '2019-09-29 00:00:00'
    train_end = '2020-09-30 23:59:59'
    val_start = '2020-10-01 00:00:00'
    val_end = '2020-12-31 23:59:59'
    test_start = '2021-01-01 00:00:00'
    test_end = '2021-03-10 23:59:59'
    feature_chunk_length = 6
    additional_features = ['all_generation', 'HVDC_cable', 'Coal', 'Renewables', 'Combined_cycle', 'Gas_turbine']
    color = 'red'


class Iceland(Regions):
    name = 'Iceland'
    train_start = '2021-11-05 09:00:00'
    train_end = '2022-01-02 23:59:59'
    val_start = '2022-01-03 00:00:00'
    val_end = '2022-01-16 23:59:59'
    test_start = '2022-01-17 00:00:00'
    test_end = '2022-01-30 23:59:59'
    periods = [(1, 0), (2, 0), (0, 1), (0, 2)]
    color = 'mediumblue'


class FaroeIslands(Regions):
    name = 'Faroe Islands'
    train_start = '2019-11-03 22:00:00'
    train_end = '2019-11-08 08:59:59'
    val_start = '2019-11-08 09:00:00'
    val_end = '2019-11-09 08:59:59'
    test_start = '2019-11-09 09:00:00'
    test_end = '2019-11-10 08:59:59'
    time_sensitive = False
    samples = 1
    periods = []
    color = '#c9ddf0'
