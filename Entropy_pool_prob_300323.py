#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:46:33 2023

@author: lucadaquanno
"""

import pandas as pd
import json
import requests
import numpy as np
import scipy.stats as sts
from scipy.stats import norm,chi2,t,lognorm
import matplotlib.pyplot as plt
import random
import math
import statistics
import time
import plotly as plty
import scipy.optimize as spopt
import datetime
import warnings
from operator import itemgetter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy.stats import gaussian_kde
#%%
user=1
if user ==1:
    path = "/Users/lucadaquanno/Desktop/Documents/CIOS.Analyse/Return_forecasting/Entropy_pooling_python/"
scaling_factor='d'
#%%
def time_series_flex(list_of_ISIN, start_date, end_date, **kwargs):
    list_of_dcts=[]
    for e in list_of_ISIN:
        d={"code": e, "code_type": "isin"}
        list_of_dcts.append(d)
    dct_body={
        "start_date": start_date,
        "end_date": end_date,
        "instruments": list_of_dcts,
        "convert_prices": False,
        "use_live_data": True,
        "extend_timeseries_in_db": False,
        "extend_investment_universe": False,
        "source": "merged"
    }
    dct_body.update(kwargs)
    body = json.dumps(dct_body)
    r = requests.post("https://data.acp-cios.fincite.net/api/v1/timeseries/", data=body,
                         headers = {
                             'content-type':'application/json',
                             'authorization':'Bearer L0hxZj2udrAgY1QxqW1rG5HkshYR0EY8AU9QMtDM'})
    return json.loads(r.text)

def Exp_Decay_prob(X,T_date,Tau_date,data_freq=scaling_factor):
    '''X is the timeseries of risk_drivers
    T_date is the latest observation's date
    Tau_date is the date for the half life parameter
    this function return a series of time-decaying probabilities'''
    if data_freq=='d':
        X=X.loc[:T_date]
        Tau_integer=X.loc[:Tau_date].shape[0] # associating an integer to the Tau_date
        T_integer=X.shape[0]                 # associating an integer to the T date
        exponent=[-(np.log(2)/Tau_integer)*abs((t-T_integer))for t in np.arange(0,T_integer)]
        P=1/np.sum(np.exp(exponent))
        time_conditioned_p=P*np.exp(exponent)
        return pd.Series(time_conditioned_p,name='T_cond_prob',index=X.index)
    elif data_freq == 'm':
        X=X.loc[:T_date]
        Tau_integer=X.loc[:Tau_date].shape[0] # associating an integer to the Tau_date
        T_integer=X.shape[0]              # associating an integer to the T date
        X=X.loc[:T_date]
        exponent=[-(np.log(2)/Tau_integer)*abs((t-T_integer))for t in np.arange(0,T_integer)]
        P=1/np.sum(np.exp(exponent))
        time_conditioned_p=pd.Series(P*np.exp(exponent),name='T_cond_prob',index=X.index)
        return time_conditioned_p.resample('M').sum()
    else:
        X=X.loc[:T_date]
        Tau_integer=X.loc[:Tau_date].shape[0] # associating an integer to the Tau_date
        T_integer=X.shape[0]              # associating an integer to the T date
        X=X.loc[:T_date]
        exponent=[-(np.log(2)/Tau_integer)*abs((t-T_integer))for t in np.arange(0,T_integer)]
        P=1/np.sum(np.exp(exponent))
        time_conditioned_p=pd.Series(P*np.exp(exponent),name='T_cond_prob',index=X.index)
        return time_conditioned_p.resample('Y').sum()

    
def neg_Dual_func_eq_constr(Lmbda_vector,P_0,H_matrix,h):
   '''Lmbda_vector is a ndarray with (k_ineq + k_eq) number of elements
   P_0 is a series of prior probabilities with T number of elements
   H matrix must be a dataframe K_eq(number of equality constraints) rows and T columns (T number of scenarios)
   h is a series with equality constraints values
   lmbda vector is an array with initial values for Lagrange multipliers
   The function returns the objective function value to optimize '''
   K_eq=len(h)
   lmbda_2=Lmbda_vector[0:K_eq]
   Lmbda_vector[K_eq:]=0
   q=np.exp(np.log(P_0) - 1 - H_matrix.T.dot(lmbda_2))
   Dual_func= q.T.dot(np.log(q) - np.log(P_0)) + lmbda_2.T.dot(H_matrix.dot(q)-h)
   return - Dual_func

def neg_Dual_func_constr(Lmbda_vector,P_0,F_matrix,H_matrix,f,h):
   '''Lmbda_vector is a ndarray with (k_ineq + k_eq) number of elements
   P_0 is a series of prior probabilities with T number of elements
   F matrix must be a dataframe with K_ineq(number of inequality constraints) rows and T columns (T number of scenarios)
   H matrix must be a dataframe K_eq(number of equality constraints) rows and T columns (T number of scenarios)
   f is a series with intensity views for inequality constraints 
   h is a sereis with intensity views for equality constraints
   lmbda vector is an array with initial values for Lagrange multipliers
   The function returns the objective function value to optimize'''

   K_eq=len(h)
   K_ineq=len(f)
   lmbda_1=Lmbda_vector[K_eq:K_ineq+1]
   lmbda_2=Lmbda_vector[0:K_eq]
   q=np.exp(np.log(P_0) - 1 - F_matrix.T.dot(lmbda_1) - H_matrix.T.dot(lmbda_2))
   Dual_func=  q.T.dot(np.log(q) - np.log(P_0)) + lmbda_1.T.dot(F_matrix.dot(q)-f) + lmbda_2.T.dot(H_matrix.dot(q)-h)
   return - Dual_func

def neg_Lagrangian_derivative(x,p_0,H_matrix,h,F_matrix=pd.DataFrame([]),f=pd.Series([]),eps=1e-5,function=neg_Dual_func_constr,data_freq=scaling_factor,num_eval=100):
    '''This function is used to estimate the gradient of the objective function
        x is our vector of lagrangian multipliers
        p_0 is the series of prior probabilities
        F matrix must be a dataframe with K_ineq(number of inequality constraints) rows and T columns (T number of scenarios)
        is initialized with an empty dataframe in the case we do not have inequality constraints
        H matrix must be a dataframe K_eq(number of equality constraints) rows and T columns (T number of scenarios)
        f is a series with intensity views for inequality constraints is initialized with an empty series in the case we do not have inequality constraints
        h is a sereis with intensity views for equality constraints
        eps is the infinitesimal change in the function input in order to estimate the derivative, be carefull if the number is too small
        the problem can be numerical instable
        function is the function for which we want to evaluate the derivative
        data_freq is the frequency of the data
        num_eval is the number of function evaluation to estimate the derivative, the higher is the number the more precise results we get
        even though it comes at the cost of more computational time
    '''
    if data_freq=='y':
        num_eval=len(x)
    if function == neg_Dual_func_constr:
        K_eq=len(h) # number of equality constraint
        K_ineq=len(f) # number of inequality constraint
        lmbda_1=x[K_eq:K_ineq+1] # Lagrange multipliers for inequality constraints
        lmbda_2=x[0:K_eq]        # Lagrange multipliers for equality constraints 
        q=np.exp(np.log(p_0) - 1 - F_matrix.T.dot(lmbda_1) - H_matrix.T.dot(lmbda_2))
        L_addendum= lmbda_1.T.dot(F.dot(q)-f) + lmbda_2.T.dot(H.dot(q)-h)
    else:
        K_eq=len(h)
        lmbda_2=x[0:K_eq] 
        q=np.exp(np.log(p_0) - 1 - H_matrix.T.dot(lmbda_2))
        L_addendum= lmbda_2.T.dot(H.dot(q)-h)
    def neg_Lagrangian(q):
        return - (q.T.dot(np.log(q)-np.log(p_0)) + L_addendum)
    n=q.shape[0]
    g_x= lambda x:neg_Lagrangian(x)
    random.seed(20)
    lst = random.sample(range(1,len(q)),num_eval) #extract randomly some elements of q in which evaluate the derivative
    k=len(lst)
    stoc_gradient = np.full(k,np.nan) #pre-allocation of the Gradient
    j=0
    for i in lst:
        # creating a diagonal matrix with the infinitesimal increment in the diagonal, 
        # indexing with [i] we create a vector of 0 except for the i-th element which is equal to eps
        # then we add 0 to all q elements, except for the i-th element which is added by eps
        stoc_gradient[j]=(g_x(q + eps*np.eye(n)[i]) - 2*g_x(q) + g_x(q - eps*np.eye(n)[i])) / (2*eps)
        j=j+1
    return stoc_gradient
#neg_Lagrangian_derivative(lmbda_vector_0,p_0,F,H,f,h,eps=1e-5)

def lambda1_fun_eq(x,F_matrix,H_matrix,f,h):
     K_ineq=len(f)
     K_eq=len(h)
     lmbda_1=x[K_eq:K_ineq+1] # Lagrange multipliers for inequality constraints
     lmbda_2=x[0:K_eq]        # Lagrange multipliers for equality constraints
     q=np.exp(np.log(p_0) - 1 - F_matrix.T.dot(lmbda_1) - H_matrix.T.dot(lmbda_2))
     return lmbda_1*(F.dot(q)-f)

def lambda2_fun_eq(x,F_matrix,H_matrix,f,h,function=neg_Dual_func_constr):
    if function == neg_Dual_func_constr:
        K_ineq=len(f)
        K_eq=len(h)
        lmbda_1=x[K_eq:K_ineq+1] # Lagrange multipliers for inequality constraints
        lmbda_2=x[0:K_eq]        # Lagrange multipliers for equality constraints 
        q=np.exp(np.log(p_0) - 1 - F_matrix.T.dot(lmbda_1) - H_matrix.T.dot(lmbda_2))
    else:
         K_eq=len(h)
         lmbda_2=x[0:K_eq]
         q=np.exp(np.log(p_0) - 1 - H_matrix.T.dot(lmbda_2))
    return H.dot(q)- h

def ineq_cons(x,F_matrix,H_matrix,f,h):
    K_ineq=len(f)
    K_eq=len(h)
    lmbda_1=x[K_eq:K_ineq+1] # Lagrange multipliers for inequality constraints
    lmbda_2=x[0:K_eq]        # Lagrange multipliers for equality constraints 
    q=np.exp(np.log(p_0) - 1 - F_matrix.T.dot(lmbda_1) - H_matrix.T.dot(lmbda_2))
    return F.dot(q)-f
#%%
isin=["US78378X1072","US2605661048","IE0031719473","US4642876894","CH0012138530"]
start_date='2000-12-31'
end_date='2022-12-31'
response=time_series_flex(isin, start_date, end_date)
response_list=response['response']['instruments']
#%%
df=pd.DataFrame()
for k in response_list:
    response_dict=k['timeseries']
    dates_index = list(map(itemgetter('date'), response_dict))
    dates_index=[datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates_index]
    close_prices=list(map(itemgetter('close_price'), response_dict))
    prices=pd.DataFrame(close_prices,dates_index)
    #x=np.log(prices).diff().dropna()
    #x=x.resample('M').sum()
    df=pd.concat([df,prices],axis=1)
#%%
#df = pd.read_excel(path+"dsws_timeseries.xlsx", parse_dates = ["date"], index_col=("date") )
start_date = df.index.min()
end_date  = df.index.max() #  last available date
df = df[start_date : end_date]
bdate = pd.bdate_range(start_date, end_date) # get only business day
x = df.copy()
for i in bdate:
    if (i in x.index) == False: #checking missing values
        x.loc[i,:] = np.nan
x = x.sort_index(ascending=True)
spline = False
if spline:
    x = x.interpolate(method = "cubic")
else:
    x = x.fillna(method = "ffill")
name = ['SP500','DOW_Jones','Fixed_Income','Russell3000','Credit_Suisse']
x.columns = name
dates=x.index
x=x.pct_change().dropna()
#x=np.log(x).diff().dropna()
Time_scaling={'daily':'d','monthly':'m','yearly':'y'}
data_frequency='daily'
scaling_factor=Time_scaling[data_frequency]
if scaling_factor=='m':
    x=(1+x).resample('M').prod()-1
    print('monthly data')
elif scaling_factor=='y':
    x=(1+x).resample('Y').prod()-1
    print('yearly data')
else:
    print('daily data')

#x.index=np.arange(0,len(x))
#print(x.loc[x.index[0]:x.index[-1]])
#%%
w=[0.2, 0.2, 0.2, 0.2, 0.2] #weights
if scaling_factor=='d':
    wndw=252
elif scaling_factor=='m':
    wndw=12
else:
    wndw=2
#x_r=x.iloc[0:(len(x)-wndw+1)]
x_restricted=x.iloc[wndw-1:]
data_sample_mean=x.mean()
data_sample_volat=x.std()
std_data=(x-data_sample_mean)/data_sample_volat
epsilon=std_data.copy()
epsilon_restricted=epsilon.iloc[wndw-1:]
tau_date='2020-01-04'
T_date=epsilon_restricted.index[-1]
time_cond_prob= Exp_Decay_prob(epsilon_restricted,T_date,tau_date)
print(np.sum(time_cond_prob))

exp_decay_flag=True
if exp_decay_flag:
    p_0=time_cond_prob
else: #equally weighted probability as a prior
    p_0=pd.Series(np.ones(len(epsilon_restricted))*1/len(epsilon_restricted),index=epsilon_restricted.index)
#%%
v_1x=(epsilon_restricted@w)
if data_frequency=='daily':
    scaling_adjustment=252
elif data_frequency=='monthly':
    scaling_adjustment=22
else:
    scaling_adjustment=1
v_2x=(epsilon@w).rolling(window=wndw).std().dropna()
H=pd.DataFrame(np.ones(len(p_0)),index=p_0.index,columns=['ones']).T
h=pd.Series([1],index=H.index)
#%%
mean_distribution=(epsilon@w).rolling(window=wndw).mean().dropna()
vola_distribution=(epsilon@w).rolling(window=wndw).std().dropna()
port_mean_distribution=(x@w).rolling(window=wndw).mean().dropna()
port_vola_distribution=(x@w).rolling(window=wndw).std().dropna()
df, loc_t, scale_t=t.fit(port_mean_distribution)
shape,loc_vol,scale_vol=lognorm.fit(port_vola_distribution)
obj_mean=-0.12
obj_vol=0.2
quantile_mean=round(t.cdf(obj_mean/252,df,loc_t,scale_t),2)
#quantile_mean=round(kde_mean.integrate_box_1d(-np.inf,obj_mean/252),4)
print(quantile_mean)
quantile_vol=round(lognorm.cdf(obj_vol/np.sqrt(252),shape,loc_vol,scale_vol),2)
#quantile_vol=round(kde_vol.integrate_box_1d(-np.inf,obj_vol/np.sqrt(252)),2)
print(quantile_vol)
#%%
v_star1=mean_distribution.quantile(quantile_mean)
v_star2=vola_distribution.quantile(quantile_vol)
F=pd.DataFrame([-v_1x, v_2x],index=['Exp_value_const','Volatility_const'],columns=p_0.index)
f=pd.Series([-v_star1,v_star2],index=F.index)
#%%
K_eq=len(h)
K_ineq=len(f)
# initial guess
lmbda_vector_0=np.ones(K_eq+K_ineq)
lmbda_vector_0[K_eq:K_ineq+1]=-1 
lmbda_2=lmbda_vector_0[0:K_eq]       # Lagrange multipliers for equality constraints
lmbda_1=lmbda_vector_0[K_eq:K_ineq+1]# Lagrange multipliers for inequality constraints

if (K_eq!=0) & (K_ineq!=0):
    obj_fun= neg_Dual_func_constr
else:
    obj_fun= neg_Dual_func_eq_constr
#%%
if (K_ineq!=0):
    cons =    ({'type': 'eq', 'fun': lambda1_fun_eq, 'args': (F,H,f,h)},
           {'type': 'eq', 'fun': lambda2_fun_eq, 'args': (F,H,f,h)},
           {'type': 'ineq', 'fun': ineq_cons,    'args': (F,H,f,h)})
    arguments=(p_0,F,H,f,h)
else: 
    cons = ({'type': 'eq', 'fun': lambda2_fun_eq, 'args': (F,H,f,h)})
    arguments=(p_0,H,h)

if (K_ineq!=0):
    #bnds= [(1*10e4,-1*10e4),(-1*10e-4,0),(-1*10e4,0)]
    bnds= [(None,None),(None,0),(None,0)]
else:
    bnds= [(None,None),(None,None),(None,None)]
#%%
res=spopt.minimize(obj_fun,lmbda_vector_0,method='SLSQP',args=arguments,bounds=bnds,constraints=cons,options={'maxiter':200,'disp': True})
Lagrangian_mltps=res.x
lmbda_2=Lagrangian_mltps[0:K_eq]
lmbda_1=Lagrangian_mltps[K_eq:K_ineq+1]
print(res)
#%%
Lagrangian_mltps=res.x
lmbda_2=Lagrangian_mltps[0:K_eq]
lmbda_1=Lagrangian_mltps[K_eq:K_ineq+1]
opt_lagran=res.x
post_prob=np.exp(np.log(p_0) - 1 - F.T.dot(lmbda_1) - H.T.dot(lmbda_2))
#%%
(p_0 - post_prob).plot()
#%%
ax=post_prob.plot()
p_0.plot()
ax.legend(['posterior','prior'])
#%%
plt.scatter(v_1x.index,v_1x.values,c=p_0)
plt.xticks(rotation=45)
color_map=plt.cm.get_cmap('Blues')
cbar = plt.colorbar()
cbar.set_label('Likelihood')

# add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Returns')
plt.title('portfolio returns\likelihood  scatter plot')
#%%
plt.scatter(v_1x.index,v_1x.values,c=post_prob)
plt.xticks(rotation=45)
color_map=plt.cm.get_cmap('Blues')
cbar = plt.colorbar()
cbar.set_label('Likelihood')

# add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Returns')
plt.title('Portfolio returns-likelihood  scatter plot')
#%%
post_prob=post_prob/post_prob.sum() 
test_1=post_prob.T.dot(H.T)
# Remember we have an inequality like: Ax>b
#equal to: -Ax<-b
#-Ax=F*p_post
#-b=f
#-Ax+b<0= (F*p_post - f)<0
test_2= post_prob.T.dot(F.iloc[0].values)-f.values[0]
test_3= post_prob.T.dot(F.iloc[1].values)-f.values[1]
print(test_1) #the sum must be 1
print(test_2) #the difference must be greater or equal to 0'
print(test_3) #the difference must be greater or equal to 0