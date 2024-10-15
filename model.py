import numpy as np
from dataclasses import dataclass


@dataclass
class Data:
    keys: set
    points: list #np.ndarray
    data: np.ndarray

# default_step = 0.01/12

default_step = 0.01/12

def rungekutta4(func, param, init_x, T, step = default_step, econ_data=None, **kwargs):

    ## input :
    # func(parameters, x_values) - function discribes the model
    # param - system parameters
    
    # init_x - initial system values
    # T - time-end
    # step - grid step
    
    ## output :
    # result - system vaue on grid of time points
    init_x[5] = param[20]
    Nt = int(T/step) 
    ad_shape = int(param.size/param.shape[0])
    init_val_temp = np.asarray([init_x.copy()]*ad_shape).T
    
    result = np.zeros((len(init_x), Nt+1, ad_shape))
    result[:,0] = init_val_temp
        
    for j in range(1,Nt+1):
    
        if econ_data:
            econ = econ_data.iloc[int(np.floor(j*step))]
            try:
                param[1] = np.array(param[-len(econ):,0]).dot(np.array(econ))
            except:
                param[1] = np.array(param[-len(econ):]).dot(np.array(econ))
        a1 = step * func(param, init_val_temp, j*step)
        val_temp = init_val_temp + a1*0.5

        a2 = step * func(param, val_temp, (j+0.5)*step)
        val_temp = init_val_temp + a2*0.5

        a3 = step * func(param, val_temp, (j+0.5)*step)
        val_temp = init_val_temp + a3

        a4 = step * func(param, val_temp, (j+1.0)*step)

        init_val_temp += (a1 + 2.0 * a2 + 2.0 * a3 + a4)*1.0 / 6.0

        result[:,j] = init_val_temp

    return result


model_keys = np.array(['S', 'L', 'I', 'T', 'J1', 'J2', 'J3', 'A',
                       'S+', 'L+', 'I+', 'T+', 'J1+', 'J2+', 'J3+', 'A+'])

def TB_HIV_model(param, sys_x, time=0):
    ## input :
    # param - system parameters
    # sys_x - system values at indicated time point
    
    ## output :
    # result - system values at indicated time point
    
    ### TB+HIV model ####
    result = [0]*len(sys_x)
    R = sys_x[0]+sys_x[1]+sys_x[3]+sys_x[4]+sys_x[5]
    J_ast = sys_x[5]+sys_x[6]+sys_x[7] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    JR = J_ast*1.0/R

    result[0] = param[0] - param[1]*sys_x[0]*(sys_x[2]+sys_x[6])/param[15] - param[2]*sys_x[0]*JR - param[3]*sys_x[0]
    result[1] = param[1]*(sys_x[0]+sys_x[3])*(sys_x[2]+sys_x[6])/param[15] - param[2]*sys_x[1]*JR - (param[3]+param[4]+param[9])*sys_x[1]
    result[2] = param[4]*sys_x[1] -(param[3]+param[6]+param[10])*sys_x[2]
    result[3] = param[9]*sys_x[1] + param[10]*sys_x[2] - param[1]*sys_x[3]*(sys_x[2]+sys_x[6])/param[15] - param[2]*sys_x[3]*JR - param[3]*sys_x[3]
    result[4] = param[2]*(sys_x[0]+sys_x[3])*JR - param[1]*sys_x[4]*(sys_x[2]+sys_x[6])/param[15] - (param[12]+param[3])*sys_x[4] + param[21]*sys_x[6]
    result[5] = param[2]*sys_x[1]*JR + param[1]*sys_x[4]*(sys_x[2]+sys_x[6])/param[15] + param[11]*sys_x[6] - (param[13]+param[3]+param[5])*sys_x[5]
    result[6] = param[5]*sys_x[5] - (param[14]+param[3]+param[7]+param[11])*sys_x[6] - param[21]*sys_x[6]
    result[7] = param[12]*sys_x[4] + param[13]*sys_x[5] + param[14]*sys_x[6] - (param[3]+param[8])*sys_x[7]

    return np.array(result)

def TB_HIV_plus(param, sys_x, time=0):
    ## input :
    # param - system parameters
    # sys_x - system values at indicated time point
    
    ## output :
    # result - system values at indicated time point
    
    ### TB+HIV model ####
    result = [0]*len(sys_x)
    # result = np.empty((len(sys_x)))
    R = sys_x[0]+sys_x[1]+sys_x[3]+sys_x[4]+sys_x[5]
    J_ast = sys_x[5]+sys_x[6]+sys_x[7] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    JR = J_ast*1.0/R
    # param = np.array(param)
    # sys_x = np.array(sys_x)


    result[0] = param[0] + sys_x[0]-sys_x[0] #- param[1]*sys_x[0]*(sys_x[2]+sys_x[6])/param[15] - param[2]*sys_x[0]*JR - param[3]*sys_x[0]
    result[1] = param[1]*(sys_x[0]+sys_x[3])*(sys_x[2]+sys_x[6])/param[15]
    result[2] = param[4]*sys_x[1]
    result[3] = param[9]*sys_x[1] + param[10]*sys_x[2] 
    result[4] = param[2]*(sys_x[0]+sys_x[3])*JR
    result[5] = param[2]*sys_x[1]*JR + param[1]*sys_x[4]*(sys_x[2]+sys_x[6])/param[15] + param[11]*sys_x[6] 
    result[6] = param[5]*sys_x[5]
    result[7] = param[12]*sys_x[4] + param[13]*sys_x[5] + param[14]*sys_x[6]
    return np.array(result)


def calculate_plus(params, data, points, T, n_equations, steps = 10):
    try:
        data_plus = np.zeros((data.shape[0], data.shape[1], params.shape[1]))
    except:
        data_plus = np.zeros((data.shape[0], data.shape[1],1))
    #print(params.shape)
    points_plus = points
    # print(data.shape)
    for i in range(data.shape[1]):
        for j in range(steps):
            # print(params.shape)
            # print(TB_HIV_plus(params, data[:,i-j]).shape)
            # print(data_plus[:,i].shape)

            data_plus[:,i] += TB_HIV_plus(params, data[:,i-j])
        # if i>0:
        # points_plus[i] = points_plus[i-1]+step
    data_plus *= default_step
    return data_plus, points_plus

def TB_rk4(params, init_x, T, **kwargs):
    data = np.array(rungekutta4(TB_HIV_model, params, init_x, T, **kwargs))
    points = np.array([np.linspace(0,T,num = len(data[0]), endpoint=True) for i in range(len(data))])

    data_plus, points_plus = calculate_plus(params, data, points, n_equations=data.shape[0] , T=T)

    data = np.concatenate((data, data_plus))
    points = np.concatenate((points, points_plus))
    keys = model_keys

    return Data(keys, points, data)
    

def data_prediction(nb_mes, mes_eq_ind, **kwargs): # model,param, init_x, T, ):
    ## input :
    # nb_mes - number of measurements
    # mes_eq_ind - indexes of measured equations

    ## output :
    # result - predicted data (for parameters) at indicated time points, except 0-point
    
    result = TB_rk4(**kwargs).data #(model, param, init_x, T)
    nt = result.shape[1]
    
    ### to return only the mesurements at indecated time points, without 0-point
    mesurements = np.arange(0, nt, int((nt-1)*1.0/nb_mes))[1:]
    
    return result[mes_eq_ind][:,mesurements]

def syntetic_data(noise = [[0.2], [0], [1]], **kwargs):
    ## input :
    # noise = [%s, alfas, sigmas] - percentages of given gaussian noises for each measured equations
    
    ## output :
    # result - synthetic data with given noise at indicated time points, except 0-point
    
    result = data_prediction(**kwargs)
    nt = result.shape
    
    if (noise != None):
        result += noise[0]*np.random.normal(noise[1],noise[2],nt)
    
    ### to remove some mistakes with negative numbers
    condition = result < 0.0
    result[condition] = 0.0
     
    return result

def loss_func(data_pred: Data, data_ex: Data):
    ## input :
    # data_pred - predicted data
    # data_ex - exact data
    
    ## output :
    # quadratic loss
    target_sum = 0

    _f = lambda x: x in data_ex.keys
    inds = list(map(_f, data_pred.keys))
    data_pred.data = data_pred.data[inds]
    data_pred.points = data_pred.points[inds]
    data_pred.keys = data_pred.keys[inds]

    if np.all(data_pred.keys != data_ex.keys):
        raise ValueError('Given exact data keys do not match model keys')
    # temp = [0]*(len(data_pred.keys))

    for i in range(len(data_pred.keys)):
        _f = lambda x: (x in data_ex.points[i])
        _f2 = lambda x: (x in data_pred.points[i])
        inds = list(map(_f, data_pred.points[i]))
        inds2 = list(map(_f2, data_ex.points[i]))
        
        # temp[i] = data_pred.points[i, inds]
        # print(data_pred.data[i, inds].T.shape, '    ', np.array(data_ex.data[i]).shape)
        # print(data_pred.data[i, inds].T, '   ', np.array(data_ex.data[i])[inds2])
        dp = np.array(data_ex.data[i])[inds2]
        #dp_n = np.sqrt(np.sum(dp*dp))
        dp_n = np.sum(np.abs(dp)**2)
        target_sum += np.sum((data_pred.data[i, inds].T - dp).T**2, (0))/dp_n   

    return target_sum
    #return np.sum((data_pred.data - data_ex.data)**2, (0,1))


def full_func(data_ex: Data, model, **kwargs): # init_x, T, nb_mes, mes_eq_ind):
    ## input :
    # data_ex - exact data

    ## output :
    # quadratic loss of model with indicated parameters and exact data
    
    prediction = model(**kwargs)#data_prediction(**kwargs) #param, model, init_x, T, nb_mes, mes_eq_ind)
    normed_pred = prediction # prediction/np.asarray([np.max(prediction, prediction.shape[-1]).T]).T
    normed_ex = data_ex # data_ex/np.asarray([np.max(data_ex, data_ex.shape[-1]).T]).T
    
    return loss_func(normed_pred, normed_ex)


def fill_param(x, param_ind_, exact_param):
    added_shape = int(x.size/x.shape[0])
    param = np.asarray([exact_param.copy()]*added_shape).T
    
    if added_shape == 1 :
        param[param_ind_] = x.reshape((-1, 1))
      
    else: 
        param[param_ind_] = x

    return param
