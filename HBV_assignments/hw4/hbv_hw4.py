# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:15:23 2020

@author: harshv
"""
import pymc3 as pm
import numpy as np
import pandas as pd
import theano
from scipy import stats
from sklearn import preprocessing

import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable('default', max_rows=None)
import arviz as az

import warnings
warnings.filterwarnings('ignore');

#Q1
data = [['Island 1', 0.2, 0.2, 0.2, 0.2, 0.2], ['Island 2', 0.8, 0.1, 0.05, 0.025, 0.025], ['Island 3', 0.05, 0.15, 0.7, 0.05, 0.05]];
df = pd.DataFrame(data, columns = ['Island', 'Bird A', 'Bird B', 'Bird C', 'Bird D', 'Bird E']);
df

df['entropy']= -(df[df.columns[1:]]*np.log(df[df.columns[1:]])).sum(axis=1);

for i in range(3):
    df[f'entropy_model{i+1}']= (df[df.columns[1:6]]*(np.log(df[df.columns[1:6]]) - np.log(df[df.columns[1:6]]).loc[i,])).sum(axis=1);
    
#The first island has the largest entropy, followed by the third, and then the second in last place

#Q2

d = pd.read_csv('../../data/happiness.csv', header=0);
d.head();
d = d.loc[d.age > 17,];
d['age'] = ( d['age'] - 18 ) / ( 65 - 18 );
d['married'] = d['married'].astype('Int64');

married = theano.shared(np.array(d.married))
with pm.Model() as model_69:
    # Data
    age = pm.Data('age', d['age'].values);
    #married = pm.Data('married', d['married'].values)
    happiness = pm.Data('happiness', d['happiness'].values);
    
    # Priors
    a = pm.Normal('a', mu=0, sd=1, shape=2);
    bA = pm.Normal('bA', mu=0, sd=2);
    sigma = pm.Exponential('sigma', lam=1);
    
    # Regression
    mu = a[married] + bA * age;
    happy_hat = pm.Normal('happy_hat', mu=mu, sd=sigma, observed=happiness);
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30);
    posterior_69 = pm.sample();
    posterior_pred_69 = pm.sample_posterior_predictive(posterior_69);
    
az.summary(posterior_69, credible_interval=.89).round(2);
pm.traceplot(posterior_69);

with pm.Model() as model_610:
    # Data
    age = pm.Data('age', d['age'].values);
    happiness = pm.Data('happiness', d['happiness'].values);
    
    # Priors
    a = pm.Normal('a', mu=0, sd=1);
    bA = pm.Normal('bA', mu=0, sd=2);
    sigma = pm.Exponential('sigma', lam=1);
    
    # Regression
    mu = a + bA * age;
    happy_hat = pm.Normal('happy_hat', mu=mu, sd=sigma, observed=happiness);
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30);
    posterior_610 = pm.sample();
    posterior_pred_610 = pm.sample_posterior_predictive(posterior_610);
    
az.summary(posterior_610, credible_interval=.89).round(2);
pm.traceplot(posterior_610);

model_69.name = 'model_69';
model_610.name = 'model_610';
pm.compare({model_69: posterior_69, model_610: posterior_610});

#The model that produces the invalid inference, m6.9, is expected to predict much better. 
#And it would. This is because the collider path does convey actual association. 
#We simply end up mistaken about the causal inference. 
#We should not use WAIC (or LOO) to choose among models, unless we have some clear sense of the causal model. 

#Q3
d = pd.read_csv('../../data/foxes.csv', sep=';', header=0);
d.head();
d[['avgfood','groupsize','area','weight']] = preprocessing.scale(d[['avgfood','groupsize','area','weight']]);
d.head();

avgfood = theano.shared(np.array(d.avgfood));
groupsize = theano.shared(np.array(d.groupsize));
area = theano.shared(np.array(d.area));
weight = theano.shared(np.array(d.weight));

with pm.Model() as model_1:
    # Priors
    a = pm.Normal('alpha', mu=0, sd=0.2)
    b = pm.Normal('beta', mu=0, sd=0.5, shape=3)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Regression
    mu = a + b[0] * avgfood + b[1] * groupsize + b[2] * area
    weight_hat = pm.Normal('weight_hat', mu=mu, sd=sigma, observed=weight)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30)
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)

posterior_1 = posterior

with pm.Model() as model_2:
    # Priors
    a = pm.Normal('alpha', mu=0, sd=0.2)
    b = pm.Normal('beta', mu=0, sd=0.5, shape=3)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Regression
    mu = a + b[0] * avgfood + b[1] * groupsize
    weight_hat = pm.Normal('weight_hat', mu=mu, sd=sigma, observed=weight)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30)
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)

posterior_2 = posterior

with pm.Model() as model_3:
    # Priors
    a = pm.Normal('alpha', mu=0, sd=0.2)
    b = pm.Normal('beta', mu=0, sd=0.5, shape=3)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Regression
    mu = a + b[1] * groupsize + b[2] * area
    weight_hat = pm.Normal('weight_hat', mu=mu, sd=sigma, observed=weight)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30)
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)

posterior_3 = posterior

with pm.Model() as model_4:
    # Priors
    a = pm.Normal('alpha', mu=0, sd=0.2)
    b = pm.Normal('beta', mu=0, sd=0.5, shape=3)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Regression
    mu = a + b[0] * avgfood
    weight_hat = pm.Normal('weight_hat', mu=mu, sd=sigma, observed=weight)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30)
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)

posterior_4 = posterior

with pm.Model() as model_5:
    # Priors
    a = pm.Normal('alpha', mu=0, sd=0.2)
    b = pm.Normal('beta', mu=0, sd=0.5, shape=3)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Regression
    mu = a + b[2] * area
    weight_hat = pm.Normal('weight_hat', mu=mu, sd=sigma, observed=weight)
    
    # Prior sampling, trace definition and posterior sampling
    prior = pm.sample_prior_predictive(samples = 30)
    posterior = pm.sample(draws=1000, tune=1000)
    posterior_pred = pm.sample_posterior_predictive(posterior)

posterior_5 = posterior

model_1.name = 'model_1'
model_2.name = 'model_2'
model_3.name = 'model_3'
model_4.name = 'model_4'
model_5.name = 'model_5'

pm.compare({model_1: posterior_1,
            model_2: posterior_2,
            model_3: posterior_3,
            model_4: posterior_4,
            model_5: posterior_5})

#the top three models are m1, m3, and m2. They have very similar WAIC values. 
#The differences are small and smaller in all cases than the standard error of the difference. 
#WAIC sees these models are tied. 

az.summary(posterior_4, credible_interval=.89).round(2);
az.summary(posterior_5, credible_interval=.89).round(2);









