# ======================================== #
# Some additional functions for effort sharing class
# ======================================== #

# Preambule
import numpy as np
import pandas as pd

# Functions
def rho(self, t):
    try:
        rho = 1-(1/self.timescale_of_convergence)*(t-2019)
        rho[rho<0] = 0
    except:
        rho = 1-(1/self.timescale_of_convergence)*(t-2019)
        if rho < 0:
            rho = 0
    return rho

def determine_coefficient(budget, currentemissions):
    return -0.5*currentemissions**2 / budget

def cumfuturepop_func(self):
    pop = 0
    for y in range(2020, 2101):
        p = float(self.xr_total.sel(ISO="WORLD", Time=y).Population)
        pop += p
    return pop

def cumpopshare_func(self, c):
    teller = float(self.xr_total.sel(ISO=c, Time=range(2020, 2101)).Population.sum(dim="Time"))
    noemer = float(self.xr_total.sel(ISO="WORLD", Time=range(2020, 2101)).Population.sum(dim="Time"))
    # for y in range(2020, 2101):
    #     try:
    #         sh1 = float(self.xr_total.sel(ISO=c, Time=y).Population)
    #         teller += sh1
    #         sh2 = float(self.xr_total.sel(ISO="WORLD", Time=y).Population)
    #         noemer += sh2
    #     except:
    #         continue
    try:
        return teller / noemer
    except:
        return np.nan

# def popshare_array(self, y, c):
#     try:
#         if y <= 2021:
#             sh = float(self.xr_unp.sel(Time=y, ISO=c).Population)/float(self.xr_unp.sel(Time=y, ISO='WORLD').Population)
#         else:
#             sh = float(self.xr_unp_f.sel(Time=y, ISO=c).Population)/float(self.xr_unp_f.sel(Time=y, ISO='WORLD').Population)
#     except:
#         sh = 0
#     return sh

def gdp_future_reread(self, y, c, mode):
    try:
        if mode == "fraction":
            return float(self.xr_gdp.sel(ISO=c, Time=y).GDP) / float(self.xr_gdp.sel(ISO="WORLD", Time=y).GDP)
        if mode == "abs":
            return float(self.xr_gdp.sel(ISO=c, Time=y).GDP)
    except:
        return np.nan

def popshare_func(self, y, c):
    try:
        sh = float(self.xr_total.sel(ISO=c, Time=y).Population) / float(self.xr_total.sel(ISO='WORLD', Time=y).Population)
    except:
        sh = 0
    return sh

def gdp_future(self, y, c, mode):
    if c in ["CYM", "FRO", "GIB", "IMN", "JEY", "LIE", "MAF", "SXM", "TCA",
             "VGB", "SSD", "PRK", "KIR", "FSM", "MHL", "TUV", "DMA", "NRU",
             "ASM", "GRD", "PLW", "SOM", "CUW", "ATG", "KNA", "SYC", "MNP",
             "GUM", "VIR", "BMU", "SMR", "AND", "GRL"]:
        return np.nan
    else:
        try:
            if mode == "fraction":
                return float(self.xr_total.sel(ISO=c, Time=y).GDP) / float(self.xr_total.sel(ISO="WORLD", Time=y).GDP)
            if mode == "abs":
                return float(self.xr_total.sel(ISO=c, Time=y).GDP)
        except:
            return np.nan

# def gdpshare_func(self, y, c):
#     if c != "EU":
#         try:
#             sh = float(np.array(self.xr_weo.sel(Region=c, Time = y).Value).astype(str)[0].replace(',', ''))/float(np.array(self.xr_weo_gr.sel(Region="World", Time = y).Value).astype(str)[0].replace(',', ''))
#         except:
#             sh = 0
#     if c == "EU":
#         try:
#             sh = float(np.array(self.xr_weo_gr.sel(Region="European Union", Time = y).Value).astype(str)[0].replace(',', ''))/float(np.array(self.xr_weo_gr.sel(Region="World", Time = y).Value).astype(str)[0].replace(',', ''))
#         except:
#             sh = 0
#     return sh

# def gdp_func(self, y, c):
#     if c != "EU":
#         try:
#             sh = float(np.array(self.xr_weo.sel(Region=c, Time = y).Value).astype(str)[0].replace(',', ''))
#         except:
#             sh = 0
#     if c == "EU" :
#         try:
#             sh = float(np.array(self.xr_weo_gr.sel(Region="European Union", Time = y).Value).astype(str)[0].replace(',', ''))
#         except:
#             sh = 0
#     return sh

def pop_func(self, y, c):
    try:
        sh = float(self.xr_total.sel(ISO=c, Time=y).Population)
    except:
        sh = 0
    return sh

def emis_func(self, y, c):
    try:
        sh = float(self.xr_total.sel(ISO=c, Time=y).GHG_p)
    except:
        sh = 0
    return sh

def emis_f_func(self, y, ccat):
    try:
        sh = float(self.xr_total.sel(ISO="WORLD", Time=y, Category=ccat).GHG_f)
    except:
        sh = 0
    return sh

# def emis_fpos_func(self, y, ccat_i):
#     try:
#         sh = float(self.xr_ar6.sel(Time=y, Variable=["Emissions|Kyoto Gases|w/o LULUCF", "Carbon Sequestration|CCS", "Carbon Sequestration|Direct Air Capture"], ModelScenario=self.modscens_cats[ccat_i]).mean(dim="ModelScenario").sum(dim="Variable").Value)/1e3
#     except:
#         sh = 0
#     return sh

def emis_total_func(self, y):
    try:
        sh = float(self.xr_total.sel(ISO="WORLD", Time=y).GHG_p)
    except:
        sh = 0
    return sh

def emisshare_func(self, y, c):
    try:
        sh = float(self.xr_total.sel(ISO=c, Time=y).GHG_p) / float(self.xr_total.sel(ISO="WORLD", Time=y).GHG_p)
        #float(self.xr_edgar.sel(year=y,ISO=c).sum(dim="Source").GHG)/float(self.xr_edgar.sel(year=y, ISO=self.all_countries_edgar).sum(dim=["ISO", "Source"]).GHG)
    except:
        sh = 0
    return sh

# def baseline_emis_func(self, y, c, ms):
#     try:
#         sh = float(self.xr_ar6_iso3.sel(Region=c, Variable="Emissions|Kyoto Gases|w/o LULUCF", Time=y, ModelScenario=ms).mean(dim="ModelScenario").Value)
#     except:
#         sh = 0
#     return sh

# def baseline_pop_func(self, y, c):
#     try:
#         sh = float(self.xr_ar6_iso3.sel(Region=c, Variable="Population", Time=y).mean(dim="ModelScenario").Value)
#     except:
#         sh = 0
#     return sh

def create_groups(self, df, col, operation, time='yes'):
    if operation == 'sum':
        operation = np.nansum
    elif operation == 'mean':
        operation = np.nanmean
    if time == 'yes':
        iso_tot = []
        vals_tot = []
        time_tot = []
        time = np.array(df.Time)
        for y in np.arange(1950, 2101):
            isos = np.array(df.ISO)[time == y]
            vals = np.array(df[col])[time == y]
            isos_new = self.groups_iso
            vals_new = np.zeros(len(self.groups_ctys))
            for g_i in range(len(self.groups_iso)):
                ctys = self.groups_ctys[g_i]
                val = []
                for c in ctys:
                    try:
                        val.append(vals[isos == c][0])
                    except:
                        val.append(np.nan)
                        continue
                vals_new[g_i] = operation(val)
            iso_tot = iso_tot + list(isos_new)
            vals_tot = vals_tot + list(vals_new)
            time_tot = time_tot + [y]*len(vals_new)
        df = {}
        df['ISO'] = iso_tot
        df['Time'] = time_tot
        vals_tot = np.array(vals_tot)
        vals_tot[vals_tot == 0] = np.nan
        df[col] = vals_tot
        return pd.DataFrame(df)
    else:
        isos = np.array(df.ISO)
        vals = np.array(df[col])
        isos_new = self.groups_iso
        vals_new = np.zeros(len(self.groups_ctys))
        for g_i in range(len(self.groups_iso)):
            ctys = self.groups_ctys[g_i]
            val = []
            for c in ctys:
                val.append(vals[isos == c][0])
            vals_new[g_i] = operation(val)
        df = {}
        df['ISO'] = isos_new
        df[col] = vals_new
        return pd.DataFrame(df)