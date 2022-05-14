import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from help_funcs import *
from basic_data_analysis import DataAnalysis
from predictor import *
from performance import Performance
from optimizer import Optimizer
from example_predictions import ExamplePrediction
from preprocessing import PreProcessing
from regions import *
from extra_functions import show_period


def plot_profiles():
    # Plot daily profile, hourly and daily standard deviation and acf.
    info = {'Ireland': (), 'Iceland': (), 'Balearic Islands': (), 'Nordic': (), 'Faroe Islands': ()}
    d = DataAnalysis(info)
    # d.plot_daily_profile(save_fig=True)
    # d.plot_daily_std(save_fig=True)
    # d.plot_hourly_std(save_fig=True)
    # DataAnalysis(info).plot_acf(save_fig=True)
    DataAnalysis(info).plot_acf(save_fig=True, minute=True, day=False, period=100)


if __name__ == '__main__':

    p = Performance()
    p.plot_hour_chosen_periods(start=15, end=60)
    p.plot_hour_chosen_periods(start=5, end=60)
    p.plot_hour_chosen_periods(start=0, end=5)
    p.plot_hour_chosen_periods(start=0, end=15)
    pass
