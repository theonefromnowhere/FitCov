import numpy as np
from pycorr import project_to_multipoles
from iminuit import Minuit



def GetFittedCovariance(estimators,cuts=None,path='',ret_alpha=True,ret_sep=True):
    #Cuts specify which part of the diagonals of the covariance will be fitted
    if(not (cuts is None)):
        c_max=cuts[1]
        c_min=cuts[0]
    else:
        c_max = 20
        c_min = 1
    
    res_s=estimators
    
    pls=[]
    
    #Creating the rough estimate of a covariance matrix
    for res in res_s:
        ells=[0,2,4]
        s, xiell, cov = project_to_multipoles(res, ells=ells)
        s_len=len(s)
        pls.append(np.concatenate([xiell[0][c_min:c_max],xiell[1][c_min:c_max],xiell[2][c_min:c_max]]))
        
    etal = np.cov(pls,rowvar=False,ddof=1)
    s = s[c_min:c_max]
    inds=np.concatenate([np.arange(c_min,c_max),s_len+np.arange(c_min,c_max),2*s_len+np.arange(c_min,c_max)])
    cov_jk=[]
    #Creating likelihood to use for the fitting
    for i in range(len(res_s)):
        mask = np.ones(len(res_s),dtype='bool')
        mask[i] = False
        
        cov_jk.append(np.diag(np.cov(np.array(pls)[mask],rowvar=False,ddof=1)))
    ss = np.concatenate([s,s,s])
    #Covariance of covariances for the likelihood
    cov_cov = np.linalg.pinv(np.cov(ss**2*cov_jk,rowvar=False))/len(res_s)
    
   
        
    def likelihood(alpha):
        cov_s=[]
        for res in res_s:
            s, xiell, cov = project_to_multipoles(res, ells=ells,correction = alpha)
            cov_s.append(cov)
        cov_m = np.mean(cov_s,axis=0)
        
        #The likelihood of the sample is quite weird by construction, and small values of chi2 appear. So, in order for the iminuit to resolve it properly, without complications, a scaling factor is added.
        chi2 = 10000000*(ss**2*np.diag(cov_m)[inds]-ss**2*np.diag(etal))@np.linalg.pinv(cov_cov)@(ss**2*np.diag(cov_m)[inds]-ss**2*np.diag(etal)).T
        return chi2

    print('Fitting alpha (Will take some time)')
    m = Minuit(likelihood, alpha=0.5)
    #Fitting with the help of iminuit
    m.limits=[-3,10]
    m.migrad()
    
    cov_s=[]
    #the final mean with the best-fit alpha is created
    for res in res_s:
        s, xiell, cov = project_to_multipoles(res, ells=ells,correction = m.values[0])
        cov_s.append(cov)
    cov_m = np.mean(cov_s,axis=0)
    print('Done')
    out = []
    out.append(cov_m)
    if(ret_alpha):
        out.append(m.values['alpha'])
    if(ret_sep):
        out.append(s)
    if(len(out)==1):
        return out[0]
    return out



 