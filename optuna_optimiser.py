import os
import sciris as sc
import optuna as op
import numpy as np

from model import full_func

def objective(params, fill_param, **kwargs):
    return full_func(params = fill_param(params.T), **kwargs)

def run_trial(trial, param_ind, min_bounds, max_bounds, **kwargs):#, param_ind=param_ind, min_bounds=min_bounds, max_bounds=max_bounds):
    pkeys = [str(param_ind[i]) for i in range(len(param_ind))]
    trial_params = np.zeros(len(pkeys))
    for k, key in enumerate(pkeys):
        trial_params[k] = trial.suggest_float(key, min_bounds[k], max_bounds[k])
    mismatch = objective(trial_params, **kwargs)
    return mismatch


def worker(db, **kwargs):
    ''' Run a single worker '''
    study = op.load_study(storage=db.storage, study_name=db.name)
    output = study.optimize(lambda trial: run_trial(trial, **kwargs), n_trials=db.n_trials)
    return output


def run_workers(db, **kwargs):
    ''' Run multiple workers in parallel '''
    output = sc.parallelize(lambda : worker(db, **kwargs), db.n_workers, die=False)
    return output


def make_study(db):
    ''' Make a study, deleting one if it already exists '''
    if os.path.exists(db.db_name):
        os.remove(db.db_name)
        print(f'Removed existing calibration {db.db_name}')
    output = op.create_study(storage=db.storage, study_name=db.name)
    return output

def run_optuna(n_workers = 6, n_trials = 200, **kwargs):
    # Create a (mutable) dictionary for global settings
    g = sc.objdict()
    g.name      = 'my-example-calibration3'
    g.db_name   = f'{g.name}.db'
    g.storage   = f'sqlite:///{g.db_name}'
    g.n_workers = n_workers # Define how many workers to run in parallel
    g.n_trials = n_trials # Define the number of trials, i.e. sim runs, per worker
    
    # Run the optimization
    t0 = sc.tic()
    make_study(db = g)
    run_workers(db = g, **kwargs)
    study = op.load_study(storage=g.storage, study_name=g.name)
    params_optuna = study.best_params
    T = sc.toc(t0, output=True)
    print(f'\n\nOutput: {params_optuna}, time: {T:0.1f} s')
    return params_optuna, study.best_value
