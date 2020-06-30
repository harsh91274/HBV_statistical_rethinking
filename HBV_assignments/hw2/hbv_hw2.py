import pymc3 as pm
import numpy as np
import pandas as pd
import theano

from scipy import stats
from sklearn.metrics import mean_squared_error
import math

import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable('default', max_rows=None)
import arviz as az

import warnings
warnings.filterwarnings('ignore')

#%config InlineBackend.figure_format = 'retina'
az.style.use('arviz-darkgrid')

#%%
"""Exercise 1: 
    The weights listed below were recorded in the !Kung census, but heights were not recorded for these individuals. 
    Provide predicted heights and 89% compatibility intervals for each of these individuals. 
    That is, fill in the table below, using model-based predictions.
    """

aux = pd.DataFrame(np.array([45, 40, 65, 31, 53])).reset_index().rename(columns={'index':'Individual',0:'weight'});
aux['expected_hight']=np.nan;
aux['89%_interval']=np.nan;
print(aux.head());

d=pd.read_csv('../Data/Howell1.csv',sep=';',header=0);
print(d.head());

d2 =d[d.age >= 18].reset_index(drop=True);
d2 = d2.assign(weight_c=pd.Series(d2.weight - d2.weight.mean()));
print(d2.head());

with pm.Model() as model_1:
    #data
    weight = pm.Data('weight', d2['weight_c'].values)
    height = pm.Data('height', d2['height'].values)
    
    # Priors
    alpha = pm.Normal('alpha', mu=178, sd=20)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    # Regression
    mu = alpha + beta * weight;
    height_hat = pm.Normal('height_hat', mu=mu, sd=sigma, observed=height);
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive();
    posterior_1 = pm.sample(draws=1000, tune=1000, cores=1);
    posterior_pred_1 = pm.sample_posterior_predictive(posterior_1);
    
    #posterior distribution
    az.summary(posterior_1, credible_interval=.89).round(2)
    #pm.summary(posterior_1, alpha=.11).round(2)# also possible
    
    #Traceplot
    pm.traceplot(posterior_1);    
    
    posterior_pred_1['height_hat'].shape
    d2 = d2.assign(height_hat=np.mean(posterior_pred_1['height_hat'],axis=0));
    
    plot = alt.Chart(d2)\
   .mark_circle()\
   .encode(
        x=alt.X('height', title='height',scale=alt.Scale(domain=(135, 180))),
        y=alt.Y('height_hat', title='height_hat',scale=alt.Scale(domain=(135, 180)))
          )

    plot
    plt.savefig('f1.jpeg');
    
    print(f'The RMSE in X_train is {round(math.sqrt(mean_squared_error(d2.height.values, d2.height_hat.values)),2)} cm.')
    
    #height_hat_hpd = pm.hpd(posterior_pred_1['height_hat'], alpha=0.11);
    height_hat_hpd = pm.hpd(posterior_pred_1['height_hat'], 0.89);
    
    hpdi = pd.DataFrame(height_hat_hpd).rename({0:'lower_hpdi',1:'upper_hpdi'}, axis=1)
    d3 = pd.concat([d2,hpdi], axis = 1)
    print(round(d3.head(5),2));
    
    weight.set_value(aux.weight.values-np.mean(d2.weight))
    posterior_pred_1 = pm.sample_posterior_predictive(trace = posterior_1, samples = 500, model = model_1);
    aux['expected height'] = posterior_pred_1['height_hat'].mean(axis=0)
    aux['89% interval'] = pd.Series(list(pm.hpd(posterior_pred_1['height_hat'], 0.11)))
    print('AUX');
    print(aux);
    

    
    
    
    