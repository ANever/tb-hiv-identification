import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as copy
from model import Data, ode_objective, ode_rk4
from optuna_optimiser import run_optuna

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

for region in all_regions[25:]:
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

    t0 = 2009
    modelling_time = 10
    T = t0 + modelling_time

    params["N"] = df_region["N"].iloc[0] * 1000

    model_kwargs = {
        "init_x": initial_state,
        "t0": 2009,
        "T": T,
        "data_ex": inverse_problem_data,
        "default_params": params,
        "objective": ode_objective,
        "estimation_bounds": estimation_bounds,
        # "equation_strings": p["equation_strings"],
        # "custom_vars": p["custom_vars"],
        "n_workers": 8,
        "n_trials": 40,
    } | p["model_kwargs"]

    # for region in ["Свердловская обл.", "Московская обл."]:
    try:
        print("Optuna run")
        optuna_dict, best_val = run_optuna(**model_kwargs)

        optuna_P = params.copy()
        for key in estimation_bounds.keys():
            optuna_P[key] = optuna_dict[key]

        print("Start plotting")

        optuna_Pb = copy(optuna_P)
        optuna_Pw = copy(optuna_P)

        optuna_Pb["r2"] *= 1.1
        optuna_Pw["r2"] *= 0.9

        optuna = ode_rk4(params=optuna_P, **model_kwargs)
        optuna_better_treat = ode_rk4(params=optuna_Pb, **model_kwargs)
        optuna_worse_treat = ode_rk4(params=optuna_P, **model_kwargs)

        finish_state = optuna.data[:, -1].T[0]

        model_kwargs["init_x"] = finish_state
        Y0 = finish_state

        titles = [
            "Number of TB infectious (without HIV) individuals (I)",
            "Number of infectious with HIV (J1)",
            "Number of infectious with both TB and HIV individuals (J3)",
        ]

        short_titles = ["(I)", "(J1)", "(J3)"]

        eq_ind = ["I", "J1", "J3"]

        syn_data = inverse_problem_data.data
        year_step = 1
        start_point = 0
        st_year = 2009 + start_point
        end_year = st_year + int(np.floor(modelling_time) + 1)
        for g in range(len(eq_ind)):
            plt.figure(figsize=(10, 6))
            i, j = optuna.data.shape[:2]

            def plot_scenario(
                scenario: Data, ind, plot_func=plt.plot, linewidth=3, **kwargs
            ):
                pNd = scenario[ind]
                plot_func(
                    pNd.points,
                    pNd.data,
                    linewidth=linewidth,
                    linestyle="dashed",
                    **kwargs,
                )

            # print(optuna.points)
            plot_scenario(optuna, eq_ind[g], label="Modelling_result", color="#335df5")
            plot_scenario(
                optuna_better_treat,
                eq_ind[g],
                label="Better_treatment",
                color="#5cd424",
            )
            plot_scenario(
                optuna_worse_treat, eq_ind[g], label="Worse_treatment", color="#f56d33"
            )

            new_p = np.array(points[0]) * int(j / modelling_time)
            l = int(j / modelling_time)
            if eq_ind[g] in inverse_problem_data.keys:
                """
            plot_scenario(
                inverse_problem_data,
                eq_ind[g],
                plot_func=plt.scatter,
                label="Train data",
                color="black",
                linewidth=4,
            )
                """
                plt.scatter(
                    # new_p,
                    inverse_problem_data[eq_ind[g]].points[
                        start_point : start_point + l
                    ],
                    inverse_problem_data[eq_ind[g]].data[start_point : start_point + l],
                    label="Train data",
                    linewidth=4,
                    color="black",
                )

            plt.legend(fontsize=12)

            # plt.xticks(
            #    np.arange(0, j + 1, int(np.floor(year_step * j / modelling_time))),
            #    np.arange(st_year, end_year, year_step, dtype=int),
            #    color="black",
            #    fontsize=14,
            # )
            plt.yticks(fontsize=18)
            plt.title(titles[g], fontsize=18)
            plt.savefig(
                short_titles[g]
                + str(region[0])
                + str(st_year)
                + "-"
                + str(end_year - 1)
                + ".png",
                dpi=300,
            )
        # plt.show()
        output(str(region[1]))
        output(str(st_year) + "-" + str(end_year - 1))
        # output('best result '+str(best_val))
        output(str(optuna_dict) + "\n")

        print("Plotted")
        mu = params["mu"]
        d = params["d"]
        # o1 = mu + optuna_dict["4"] + optuna_dict["9"]
        # o2 = mu + d + optuna_dict["10"]
        print("R1")
        # print(optuna_dict["1"] * optuna_dict["4"] / o1 / o2)

        print("R2")
    # print(optuna_dict["2"] / (mu + 0.1))

    # for i in range(len(optuna_P)):
    #    print(list(params.keys)[i], " : ", optuna_P[i])
    # except ValueError:
    #    output("region_failed\n")
    except:
        pass

connection.close()
