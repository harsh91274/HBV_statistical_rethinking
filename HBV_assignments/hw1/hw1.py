# -*- coding: utf-8 -*-
"""
Statistical Rethinking - Assignment 1 - Code by HBV
"""
#%%
#Questions
#1: Globe tossing data had 8 water in 15 tosses - constuoct posterior using grid approximation and flat prior. 
#2: prior is zero below p=0.5 and constant above p=0.5. 
#3: You want the 99% percentile interval of the posterior distribution of p to be only 0.05 wide.
#   How many times will you have to toss the globe to do this? 

#%%
#imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from random import choices
from scipy import stats
import altair as alt
import warnings

warnings.filterwarnings('ignore');
plt.close('all');

#plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid']);
#az.style.use('arviz-darkgrid')
#%%
#question 1
W=8;    #number of water
L=7;    #number of land
N=15;   #total number of throws
p=0.5;  #probability of water assumed 

n_points=1000;                                  #no of points
p_grid=np.linspace(0,1,n_points);               #parameter values on grid

prior=np.ones(n_points);                        #defining the prior
#prior=np.random.binomial(1,p,n);

likelihood=stats.binom.pmf(W,N,p_grid);         #likelihood [8 waters in 15 tosses]

raw_posterior=likelihood*prior;                 #raw posterior
posterior=raw_posterior/raw_posterior.sum()     #normalized posterior
sum(posterior)

#plotting
aux = pd.DataFrame(posterior).reset_index().rename({0:'prob'}, axis=1);
aux['p'] = aux['index']/1000;

#plot
f1=plt.figure(1); ax1=plt.axes; 
plt.plot(aux['p'],aux['prob'], color='blue',label='pdf1');
plt.ylabel("probability density"); plt.xlabel("p")
plt.savefig('q1_probdenfun',edgecolor='k');

#sample the posterior for check
samples=pd.DataFrame(np.random.choice(p_grid,5000,p=posterior))\
    .reset_index()\
        .rename({0:'prob'},axis=1);

#plot the samples 
f2=plt.figure(2);
plt.scatter(samples.index,samples['prob'],marker='o'); 
plt.xlabel('samples'); plt.ylabel("posterior values");
plt.savefig('q1_samples.jpeg',edgecolor='k');

f3=plt.figure(3)
plt.hist(samples['prob'], bins=20,histtype='stepfilled', color='blue',alpha=0.4)
plt.xlabel('p'); plt.ylabel("number of posterior values");
plt.savefig('q1_samples_hist.jpeg',edgecolor='k');

#stats
post_mean=round(np.mean(samples.prob),2); print(post_mean);
post_hpd = pm.stats.hpd(np.array(samples.prob), alpha=0.1); print(post_hpd);
post_quant=round(np.percentile(np.array(samples.prob), 0.5),2), round(np.percentile(np.array(samples.prob), 99.5),2); print(post_quant)
#pm.stats.quantiles(np.array(samples.prob), qlist=[0.5, 99.5]);
#%%
#question 2
prior2=np.concatenate((np.zeros(500), np.full(500,0.5)))
likelihood2=stats.binom.pmf(W,N,p_grid);         #likelihood [8 waters in 15 tosses]

raw_posterior2=likelihood2*prior2;                 #raw posterior
posterior2=raw_posterior2/raw_posterior2.sum()     #normalized posterior
sum(posterior2)

aux2 = pd.DataFrame(posterior2).reset_index().rename({0:'prob'}, axis=1);
aux2['p'] = aux2['index']/1000;

#plot
f1=plt.figure(1); #ax1=plt.axes;
#ax1.axvline(0.7,color='red');
plt.vlines(x=0.7, ymin=0, ymax=max(aux2['prob']), color='red',label='truth');
plt.plot(aux2['p'],aux2['prob'], color='green',label='pdf2');
plt.ylabel("probability density"); plt.xlabel("p")
plt.savefig('q2_probdenfun',edgecolor='k');
plt.legend();

samples2=pd.DataFrame(np.random.choice(p_grid,5000,p=posterior2))\
    .reset_index()\
        .rename({0:'prob'},axis=1);

#stats
post_mean2=round(np.mean(samples2.prob),2); print(post_mean2)
post_hpd2 = pm.stats.hpd(np.array(samples2.prob), alpha=0.1); print(post_hpd2)
post_quant2=round(np.percentile(np.array(samples2.prob), 0.5),2), round(np.percentile(np.array(samples2.prob), 99.5),2); print(post_quant2)
#pm.stats.quantiles(np.array(samples2.prob), qlist=[0.5, 99.5]);
#%%
#question 3
for n in [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000]:
    k=sum(np.random.binomial(1, p, n)); 
    likelihood3=stats.binom.pmf(k,n,p_grid);
    prior3=np.concatenate((np.zeros(500), np.full(500,0.5)));
    raw_posterior3=likelihood3*prior3;
    posterior3=raw_posterior3/sum(raw_posterior3);
    
    samples3=pd.DataFrame(np.random.choice(p_grid,5000,p=posterior3))\
    .reset_index()\
        .rename({0:'prob'},axis=1);
    
    post_quant3=round(np.percentile(np.array(samples3.prob), 0.5),2), round(np.percentile(np.array(samples3.prob), 99.5),2); 
    print(post_quant3);
    int=post_quant3[1]-post_quant3[0]; print(str(n)+" throws, 99 percentile diff: "+str(round(int,2)));

    


