import numpy as np
from tqdm import tqdm
from importlib import reload
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(): 
    
    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        # code to profile
        import class_datareading
        reload(class_datareading)
        from class_datareading import datareading

        datareader = datareading()
        datareader.read_general()
        datareader.read_ssps()
        datareader.read_undata()
        datareader.read_historicalemis()
        datareader.read_ar6()
        datareader.relation_budget_nonco2()
        datareader.determine_global_nonco2_trajectories()
        datareader.determine_global_budgets()
        datareader.determine_global_co2_trajectories()
        datareader.read_baseline()
        datareader.read_ndc() 
        datareader.merge_xr()
        datareader.add_country_groups()
        datareader.save() 

    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats('profiling_results.prof') # save to file
    
if __name__ == '__main__':
    main()