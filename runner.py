import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as copy
from model import TB_HIV_model, rungekutta4, TB_rk4, Data, fill_param, full_func
from optuna_optimiser import run_optuna, objective
#from sklearn import preprocessing
#import scipy
import time
#from numpy import linalg as LA

with open('paramteres.yml', 'r', encoding='utf8') as file:
    p = yaml.safe_load(file)
#Y0 = np.array(list(p['initial_state'].values()))
P_names = list(p['all_parameters'].keys())
P_exact = np.array(list(p['all_parameters'].values()))
param_ind = np.array(list(p['indeces2estim'].values()), dtype=int)
Nb_estim_param = len(param_ind)

print(Nb_estim_param)

## Boundaries of the parameter definition domains
min_bounds = np.zeros(Nb_estim_param)
max_bounds = np.ones(Nb_estim_param)*4.01 ### 9, 10, 11 may be from 0 to inf
max_bounds[:7] = np.array([100,5,2,2,2,2,2])
min_bounds[2] = 0.05
min_bounds[5:7] = np.array([0.5,0.5])

max_bounds[-2]=100. #J2 init
min_bounds[-2]=0.

full_df = pd.read_csv('epid.csv')

def output(string):
    with open('output.txt', 'a') as file:
        file.write(string+'\n')

def region_df(reg_name = '', df = full_df, df_e = None):
    _df = df[df['регион']==reg_name]
    _df.index = _df['год']
    _df = _df.drop(['год','регион'], axis = 1)
    _df = _df[['I+J3','J3','J1','N']]
    
    return _df



regions = np.unique(full_df['регион'])

#cycle through "regions" to estimate parameters for all regions or place list of needed region names
for region in ['Свердловская обл.', 'Московская обл.']: # for region in ['regions']:
    try:
        output(region)
        df_region = region_df(region)
        
        k=12
        n=(12)*k
        T = k-1 + 3
        
        print(df_region)
        pop = df_region['N'].iloc[0]
        P_exact[15] = pop
        print(pop)
        model_kwargs = {
            'model': TB_rk4,
            'init_x': None,
            'T': None,
            'data_ex': None,

            'param_ind': param_ind,
            'min_bounds':min_bounds, 
            'max_bounds':max_bounds,
            'fill_param':lambda params: fill_param(params.T, param_ind_ = param_ind, exact_param = P_exact),
        }
        
        I = df_region['I+J3']/1000 
        J1 = df_region['J1']/1000
        J3 = df_region['J1']/1000
        
        I = I-J3
        
        #S L I T J1 J2 J3 A
        Y0 = np.array([0.95,0.05,0,0,0,0,0,0]) * pop
        Y0[2] = I.iloc[0]
        Y0[3] = J1.iloc[0]
        Y0[4] = J1.iloc[0]
        Y0[5] = J1.iloc[0]*0.5
        Y0[6] = J3.iloc[0]
        
        print(Y0)
        model_kwargs['init_x'] = Y0
        
        for i in range(int(12/k)):
            start_point = 0
            end_point = k+start_point
            
            
            keys = ['I', 'J1']#, 'J3'
            real_data = np.array([I, J1]) #, J3
            eq_ind = np.array([2,4], #,6
                                dtype=int)

            points = ([list(range(k))]*len(real_data))
            
            rest_points = ([list(range(k, len(I)))] * len(real_data))
            
            syn_data = Data(keys=keys, points = points, data = real_data[:,start_point:end_point])
            
            model_kwargs['T'] = T
            model_kwargs['data_ex'] = syn_data
            
            print(model_kwargs)
            print('Optuna run')
            optuna_dict, best_val = run_optuna(**model_kwargs)
            
            _optuna_P = np.array(list(optuna_dict.values()))
            optuna_P = P_exact.copy()
            optuna_P[param_ind] = _optuna_P
            
            print('Start plotting')

            optuna = rungekutta4(TB_HIV_model, optuna_P, Y0, T,)
            
            optuna_Pb = copy(optuna_P)
            optuna_Pw = copy(optuna_P)
            
            optuna_Pb[10] *= 1.1
            optuna_Pw[10] *= 0.9
            
            optuna_better_treat = rungekutta4(TB_HIV_model, optuna_Pb, Y0, T,)
            
            optuna_worse_treat = rungekutta4(TB_HIV_model, optuna_Pw, Y0, T,)

            finish_state = optuna[:,-1].T[0]
            
            model_kwargs['init_x'] = finish_state
            Y0 = finish_state

            titles = [ 'Number of TB infectious (without HIV) individuals (I)',
                      'Number of infectious with HIV (J1)',
                      'Number of infectious with both TB and HIV individuals (J3)',
                      ]
            
            
            short_titles = [
                      '(I)',
                      '(J1)',
                      '(J3)']
                      
            syn_data = syn_data.data
            year_step = 1
            st_year = 2009+start_point
            end_year = st_year+int(np.floor(T)+1)
            for g in range(len(eq_ind)):
                plt.figure(figsize=(10,6))
                i,j,_ = optuna.shape
                
                plt.plot(np.arange(j), optuna[eq_ind[g]],label = 'Modelling result', linewidth = 3, linestyle = 'dashed', color = '#335df5')
                plt.plot(np.arange(j), optuna_better_treat[eq_ind[g]],label = 'Better treatment', linewidth = 3, linestyle = 'dashed', color = '#5cd424')
                plt.plot(np.arange(j), optuna_worse_treat[eq_ind[g]],label = 'Worse treatment', linewidth = 3, linestyle = 'dashed', color = '#f56d33')
                
                try:
                    new_p = np.array(points[0])* int(j/T)
                    plt.scatter(new_p , real_data[g][start_point:start_point+len(new_p)], label = 'Train data', linewidth =4, color='black')
                    new_p2 = np.array(rest_points[0])* int(j/T)
                    plt.scatter(new_p2 , real_data[g][start_point+len(new_p):start_point+len(new_p)+len(new_p2)], label = 'Test data', linewidth =4, color='red')
                except:
                    pass
                plt.legend(fontsize=12)
                
                plt.xticks(np.arange(0, j+1, int(np.floor(year_step*j/T))),
                           np.arange(st_year, end_year, year_step, dtype=int), 
                           color='black',fontsize=14)
                plt.yticks(fontsize=18)
                plt.title(titles[g],fontsize=18)
                plt.savefig(short_titles[g]+region+str(st_year)+'-'+str(end_year-1)+'.png', dpi=300)
                # plt.show()
            output(str(st_year)+'-'+str(end_year-1))
            #output('best result '+str(best_val))
            output(str(optuna_dict)+'\n')
    
    
    
        print('Plotted')
        mu = P_exact[3]
        d = P_exact[6]
        o1 = mu + optuna_dict['4'] + optuna_dict['9']
        o2 = mu + d + optuna_dict['10']
        print('R1')
        print(optuna_dict['1'] * optuna_dict['4']/o1/o2)
        
        print('R2')
        print(optuna_dict['2'] / (mu + 0.1))
        
        for i in range(len(optuna_P)):
            print(P_names[i], ' : ', optuna_P[i])
    except:
        output('region_failed\n')
    
