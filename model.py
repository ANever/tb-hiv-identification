import numpy as np
import copy as copy
import sciris as sc
from dataclasses import dataclass


@dataclass
class Data:
    keys: np.ndarray  # list
    points: np.ndarray  # list  # np.ndarray
    data: np.ndarray

    def __getitem__(self, key):
        if isinstance(key,str):
            ind = sc.findinds(self.keys, key)[0]
            return Data(keys=key, points=self.points[ind], data=self.data[ind])
        else:
            return Data(keys=self.keys[key], points=self.points[key], data=self.data[key])

default_step = 1 / 1200


def rungekutta4(func, init_x: dict, t_end: float, step: float = default_step, t_start=0, **kwargs):
    ## input :
    # func(parameters, x_values) - function discribes the model
    # param - system parameters

    # init_x - initial system values
    # T - time-end
    # step - grid step

    ## output :
    # result - system vaue on grid of time points
    T = t_end - t_start
    Nt = int(T / step)
    if Nt<0:
        raise Error('Wrong step direction')
    # init_val_temp = np.fromiter(init_x.values(), np.float32)
    init_val_temp = init_x
    keys = init_x.keys()
    result = np.zeros((len(init_x), Nt + 1))
    result[:, 0] = np.fromiter(init_val_temp.values(), np.float32)
    for j in range(1, Nt + 1):
        a1 = step * func(system_state=init_val_temp, t=j * step, **kwargs)
        val_temp = dict(zip(keys, result[:, j - 1] + a1 * 0.5))
        a2 = step * func(system_state=val_temp, t=(j + 0.5) * step, **kwargs)
        val_temp = dict(zip(keys, result[:, j - 1] + a2 * 0.5))
        a3 = step * func(system_state=val_temp, t=(j + 0.5) * step, **kwargs)
        val_temp = dict(zip(keys, result[:, j - 1] + a3))
        a4 = step * func(system_state=val_temp, t=(j + 1.0) * step, **kwargs)
        result[:, j] = result[:, j - 1] + (a1 + 2 * a2 + 2 * a3 + a4) / 6
    return result


def ode_model(
    equation_strings: dict,
    system_state: dict,
    params: dict = {},
    custom_vars: dict = {},
    t: float = 0,
    **kwargs,
):
    ## input :
    # param - system parameters
    # sys_x - system values at indicated time point

    ## output :
    # result - system values at indicated time point

    ### TB+HIV model ####
    _custom_vars = copy.copy(custom_vars)
    aliases_dict = params | system_state | {"t": t}
    for key in _custom_vars.keys():
        _custom_vars[key] = eval(_custom_vars[key], aliases_dict)
    result = np.zeros(len(equation_strings))
    for i, val in enumerate(equation_strings.values()):
        result[i] = eval(val, aliases_dict | _custom_vars)
    return np.array(result)


def ode_rk4(params, init_x, t_end, t_start=0, **kwargs):
    data = np.array(
        rungekutta4(func=ode_model, params=params, init_x=init_x, t_end=t_end, t_start=t_start, **kwargs)
    )
    points = np.array([
        np.linspace(t_start, t_end, num=len(data[0]), endpoint=True) for _ in range(len(data))
    ])
    keys = np.array(list(init_x.keys()))
    return Data(keys, points, data)


def loss_func(data_pred: Data, data_ex: Data):
    ## input :
    # data_pred - predicted data
    # data_ex - exact data

    ## output :
    # quadratic loss
    target_sum = 0
    inds = list(map(lambda x: x in data_ex.keys, data_pred.keys))
    data_pred.data = data_pred.data[inds]
    data_pred.points = data_pred.points[inds]
    data_pred.keys = data_pred.keys[inds]

    if np.all(data_pred.keys != data_ex.keys):
        raise ValueError("Given exact data keys do not match model keys")

    for i in range(len(data_pred.keys)):
        inds = list(map(lambda x: (x in data_ex.points[i]), data_pred.points[i]))
        inds2 = list(map(lambda x: (x in data_pred.points[i]), data_ex.points[i]))
        print(data_ex.points)

        dp = np.array(data_ex.data[i])[inds2]
        # dp_n = np.sqrt(np.sum(dp*dp))
        dp_n = np.sum(np.abs(dp) ** 2)
        target_sum += np.sum((data_pred.data[i, inds].T - dp).T ** 2) / dp_n
    return target_sum
    # return np.sum((data_pred.data - data_ex.data)**2, (0,1))


def objective(data_ex: Data, model, default_params, trial_params, **kwargs):
    ## input :
    # data_ex - exact data

    ## output :
    # quadratic loss of model with indicated parameters and exact data

    params = copy.copy(default_params)
    params.update(trial_params)
    prediction = model(params=params, **kwargs)
    normed_pred = prediction  # prediction/np.asarray([np.max(prediction, prediction.shape[-1]).T]).T
    normed_ex = data_ex  # data_ex/np.asarray([np.max(data_ex, data_ex.shape[-1]).T]).T
    return loss_func(normed_pred, normed_ex)


def ode_objective(**kwargs):
    return objective(model=ode_rk4, **kwargs)
