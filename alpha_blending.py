import numpy as np
from scipy.odr import *
import matplotlib.pyplot as plt

def linear_regression_dualE(sigma1,sigma2,sigma1_err,sigma2_err,plot=False):
    """
    Deriving alpha blending parameter of Möhler et al. (2017) method for DECT
    
    Paramaters
    ----------
    sigma1 : ndarray
        photon absorption cross sections per electron (sigmas) of high energy scan
    sigma2 : ndarray
        photon absorption cross sections per electron (sigmas) of low energy scan
    sigma1_err : ndarray
        uncertainties associated with sigma1
    sigma2_err : ndarray
        uncertainties associated with sigma2
    plot : bool
        if True shows plot of sigma1 vs sigma2
        
    Returns
    -------
    out.beta : ndarray
        array of alpha values
    out.sd_beta : ndarray
        array of uncertainties of alpha values
    """
    def model_func_dual(p, x):
        return((1/p[0])+(1-(1/p[0]))*x)
    model = Model(model_func_dual)
    data = RealData(sigma1, sigma2, sx=sigma1_err, sy=sigma2_err)
    odr = ODR(data, model, beta0=[1])
    out = odr.run()
#     out.pprint()
    x_fit = np.linspace(sigma1[0], sigma1[-1], 1000)
    y_fit = model_func_dual(out.beta, x_fit)
    if plot:
        fig, ax = plt.subplots()
        plt.plot(sigma1,sigma2,'bo',markersize=4)
#         ax.errorbar(sigma1, sigma2, xerr=sigma1_err, yerr=sigma2_err,color='black', linestyle='None', marker='x',alpha=0.05)
#         ax.plot(x_fit, y_fit,color='blue')
    #m=out.beta[0],c=out.beta[1],m_err=out.sd_beta[0]    
    return(out.beta,out.sd_beta)

def linear_regression_multiE(sigmas,sigmas_err,plot=False):
    """
    Deriving alpha blending parameters of Möhler et al. (2017) method for PCCT
    
    Paramaters
    ----------
    sigmas : ndarray
        array of arrays of photon absorption cross sections per electron (sigmas) of different energy bins images
    sigmas_err : ndarray
        uncertainties associated with sigmas
    plot : bool
        if True shows plot of sigma1 vs sigma2
        
    Returns
    -------
    out.beta : ndarray
        array of alpha values
    out.sd_beta : ndarray
        array of uncertainties of alpha values
    """
    global store
    store=np.copy(sigmas)
    def model_func(p, x):
        q=np.array([x.copy()]*len(store))
        for i in range(len(q)):
            q[i][0+(i*int(len(q[0])/len(q))):int(len(q[0])/len(q))+(i*int(len(q[0])/len(q)))]=0
        tmp=0
        for i in range(1,len(q)):
            tmp+=p[i]*q[i]
        return(1/p[0]-(1/p[0])*tmp)
        
    model = Model(model_func)

    tmp=sigmas[0]
    lhs=tmp.copy()
    for i in range(len(sigmas)-2):
        lhs=np.append(lhs,tmp)
    tmp=sigmas_err[0]
    lhs_err=tmp.copy()
    for i in range(len(sigmas)-2):
        lhs_err=np.append(lhs_err,tmp)

    rhs=np.array([sigmas[i] for i in range(1,len(sigmas))]).flatten()
    rhs_err=np.array([sigmas_err[i] for i in range(1,len(sigmas_err))]).flatten()

    data = RealData(lhs, rhs, sx=lhs_err, sy=rhs_err)
    odr = ODR(data, model, beta0=[1]*len(store))
    out = odr.run()
#     out.pprint()
    x_fit = np.linspace(lhs[0], lhs[-1], 1000)
    y_fit = model_func(out.beta, x_fit)
    if plot:
        fig, ax = plt.subplots()
        plt.plot(lhs,rhs,'bo',markersize=4,alpha=0.005)
#         ax.errorbar(lhs, rhs, xerr=lhs_err, yerr=rhs_err,color='black', linestyle='None', marker='x',alpha=0.005)
        ax.plot(x_fit, y_fit,'ro')
    return(out.beta,out.sd_beta)


# def mapa(fn,inserts=np.empty(0),error_map=False,plot=True,return_array=True):
#     """
#     Return coordinates of all pixels corresponding to the phantom insters
    
#     Parameters
#     ----------
#     fn: str
#         file path of phantom image
#     inserts: np.array
    
#     """
    
    
    
    
#     #specify where inserts are:
#     centres_inserts=[(int(512/2-1), int(58)),(int(117), int(117)),(int(512-117), int(115)),(int(181), int(179)),\
#                      (int(512/2), int(150+2)),(int(512-183), int(179)),(int(57), int(512/2)),(int(152), int(512/2)),\
#                      (int(512/2), int(512/2-2)),(int(358),int(512/2-2)),(int(453), int(512/2-2)),(int(184), int(512-184)),\
#                      (int(512/2+3), int(360-1)),(int(512-182), int(512-184)),(int(117), int(512-117)),\
#                      (int(512-115), int(512-117)),(int(512/2), int(440))]
#     radii=[25]*4+[9]+[25]*3+[20]+[25]*8
#     coords=[[0,0]]*len(centres_inserts)
#     for i in range(len(centres_inserts)):
#         for x in range(int(centres_inserts[i][0] - radii[i]), int(centres_inserts[i][0] + radii[i])):
#             for y in range(centres_inserts[i][1]- radii[i], centres_inserts[i][1] + radii[i]):
#                 if (x - centres_inserts[i][0])**2 + (y-centres_inserts[i][1])**2 <=radii[i]**2:
#                     coords[i]=coords[i]+[[y,x]]
#         coords[i]=coords[i][2:]
#     return(coords)

# def alpha_blending_dualE(fps,inserts_densities,inserts_densities_err):
#     d=dcmis(fps)
#     sigms=np.array(d.sigmas(inserts_densities,inserts_densities_err))
#     coords=mapa(fps)
#     sigma_vals,sigma_vals_err=[],[]
#     for k in sigms:
#         for i in coords:
#             for j in i:
#                 sigma_vals.append(k[0][j[0],j[1]])
#                 sigma_vals_err.append(k[1][j[0],j[1]])  
    
#     sigma_vals=np.array(sigma_vals)
#     sigma_vals=np.reshape(sigma_vals,(len(fps),int(len(sigma_vals)/len(fps))))
#     sigma_vals_err=np.array(sigma_vals_err)
#     sigma_vals_err=np.reshape(sigma_vals_err,(len(fps),int(len(sigma_vals_err)/len(fps))))

#     tmp=linear_regression_dualE(sigma_vals[0],sigma_vals[1], sigma_vals_err[0], sigma_vals_err[1],True)
#     return(tmp[0],tmp[1])

# def alpha_blending_multiE(fps,inserts_densities,inserts_densities_err):
#     d=dcmis(fps)
#     sigms=np.array(d.sigmas(inserts_densities,inserts_densities_err))
#     coords=mapa(fps)
#     sigma_vals,sigma_vals_err=[],[]
#     for k in sigms:
#         for i in coords:
#             for j in i:
#                 sigma_vals.append(k[0][j[0],j[1]])
#                 sigma_vals_err.append(k[1][j[0],j[1]])  
    
#     sigma_vals=np.array(sigma_vals)
#     sigma_vals=np.reshape(sigma_vals,(len(fps),int(len(sigma_vals)/len(fps))))
#     sigma_vals_err=np.array(sigma_vals_err)
#     sigma_vals_err=np.reshape(sigma_vals_err,(len(fps),int(len(sigma_vals_err)/len(fps))))
#     params=linear_regression_multiE(sigma_vals,sigma_vals_err,True)
#     return(sigma_vals,sigma_vals_err)