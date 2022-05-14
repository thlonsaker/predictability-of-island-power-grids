import matplotlib.pyplot as plt


class Standards:
    viridis = plt.get_cmap("viridis", 12)
    cividis = plt.get_cmap('cividis', 12)
    p = plt.get_cmap('plasma', 12)
    xlabel = 17
    ylabel = 17
    xticks = 14
    yticks = 14
    title = 18
    suptitle = 18
    regions = {'Balearic Islands': 'red', 'Nordic': '#367e7f', 'Iceland': 'mediumblue',
               'Faroe Islands': '#c9ddf0', 'Ireland': '#c75b23'}
    daily_profile = 'mediumblue'
    fiftyhz = viridis(12)
    patterns = viridis(5)
    one_pred = viridis(10) #  '#c75b23'
    test_series = '#367e7f'

    @staticmethod
    def period_colors(cmap):
        one_week = cmap(10)
        two_weeks = 'y'
        one_month = cmap(8)
        two_months = cmap(5)
        four_months = cmap(1)
        six_months = 'k'
        return one_week, two_weeks, one_month, two_months, four_months, six_months

    def prediction(self):
        return self.period_colors(self.viridis)
