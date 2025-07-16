import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as copy
from model import Data, ode_objective, ode_rk4
#from optuna_optimiser import run_optuna

import sqlite3

with open("paramteres.yml", "r", encoding="utf8") as file:
    p = yaml.safe_load(file)


def output(string: str):
    with open("output.txt", "a") as file:
        file.write(string + "\n")


db_filename = "tuberculosis.db"

connection = sqlite3.connect(db_filename)
cursor = connection.cursor()
cursor.execute("SELECT * from 'Reg'")
all_regions = cursor.fetchall()
region = all_regions[7]

for region in all_regions[1:2]:
    cursor.execute(
        "SELECT Year, ParameterID, value from Data WHERE RegionID = " + str(region[0])
    )
    reg_data = cursor.fetchall()

    cursor.execute("SELECT * from Par")
    all_pars = dict(cursor.fetchall())
    try:
        df_region = pd.DataFrame(columns=list(all_pars.values()))
        for date, parameterID, value in reg_data:
            # df_region.loc[date, all_pars[parameterID]] = value
            df_region.loc[date, parameterID] = value

        params_from_db = list(p["db_keys"].values())
        df_region = df_region[params_from_db]
        df_region.columns = list(p["db_keys"].keys())

        df_region["I"] = df_region["I+J3"] - df_region["J3"]
        df_region["S"] = df_region["N"] * 1000 - df_region["I+J3"] - df_region["J1"]
        points = []
        values = []
        # make here keys to be passed to inverse problem, but not all the data in db
        for key in p["passed_keys"]:  # df_region.columns:
            temp_df = df_region[key].dropna()
            points.append(list(temp_df.index))
            values.append(list(temp_df))
        inverse_problem_data = Data(keys=p["passed_keys"], points=points, data=values)
    except:
        continue
    print(inverse_problem_data["I"])

    _temp_keys = p["model_kwargs"]["equation_strings"].keys()
    initial_state = dict.fromkeys(_temp_keys, np.nan)
    for key in _temp_keys:
        initial_state[key] = df_region.iloc[0][key]

    params = p["all_parameters"]

    estimation_bounds = p["estim_and_bounds"]
    # for key in estimation_bounds.keys():
    #    if (estimation_bounds[key] == "default":
    #        estimation_bounds[key] = np.array([0, 1])

    t_start = 2009
    modelling_time = 10
    t_end = t_start + modelling_time

    params["N"] = df_region["N"].iloc[0] * 1000

    model_kwargs = {
        "init_x": initial_state,
        "t_start": t_start,
        "t_end": t_end,
        "data_ex": inverse_problem_data,
        "default_params": params,
        "estimation_bounds": estimation_bounds,
    } | p["model_kwargs"]

    results = ode_rk4(params=params, **model_kwargs)
    plt.plot(results['I'].points, results['I'].data)
    a = inverse_problem_data['I']
    #plt.scatter(a.points, a.data)
    #plt.show()
    
    model_kwargs = {
        "init_x": initial_state,
        "t_start": t_end,
        "t_end": t_start,
        "step": -1/1200,
        "default_params": params,
    } | p["model_kwargs"]
    results = ode_rk4(params=params, **model_kwargs)
    a = results['I']
    plt.plot(a.points,a.data)
    plt.show()
    print(results)
    print('DONE')
    
connection.close()
