# utils.py
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from scipy.interpolate import interp1d
from bahamas_lig.config import data_dir
from bahamas_lig.config import model_dir
from scipy.stats import chi2
import os
from IPython import display
import datetime as dt
import matplotlib.dates as mdates
import pymc3 as pm
import theano.tensor as tt
from theano import shared
from pymc3.distributions.dist_math import SplineWrapper
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import gaussian_kde
import seaborn as sns

def weighted_rsl_trace(pred_list, comp, lat, lon, var="f_pred",iters=20000):
    model_names = list(comp.index)
    weighted_trace =[]
    for i in range(iters):
        choice = np.random.choice(np.arange(len(comp)), 1, p=comp['weight'])
        key=comp.index[choice][0]
        f_preds = pred_list[key][var]
        C=np.random.choice(np.arange(len(f_preds)), 1)
        gmsl=f_preds[C].ravel()

        md_id = np.where(model_names == key)[0][0]
        if "_S" in key:
            ESL_CURVE = np.copy(Wael)
        if "_T" in key:
            ESL_CURVE = np.copy(Wael_T)
        z_functions = interpolation_functions(
            [lat], [lon], models[md_id], ESL_CURVE, Age_ESL
        )
        
        GIA_to_add = z_functions[0](X_new).ravel()

        weighted_trace.append(GIA_to_add+gmsl)
    weighted_trace=np.array(weighted_trace)
    
    
def plot_gmsl_inference(X_new,inference,color,ax,plot_max_like=False):
    
    bot = np.nanpercentile(inference, 2.5, axis=0)
    top = np.nanpercentile(inference, 97.5, axis=0)
    if plot_max_like:
        dy = np.linspace(-15,25,200)
        max_like = np.zeros(X_new.size)
        for i in range(X_new.size):
            time_slice = inference[:,i][~np.isnan(inference[:,i])]
            max_like[i]=dy[np.argmax(gaussian_kde(time_slice,bw_method=1)(dy))]
        max_like = gaussian(max_like,2)
    ax.fill_between(
            X_new.ravel(),
            bot,
            top,
            fc=(1,1,1),
            zorder=2,
            alpha=1,
            lw=0,
            ec=color,
            aa=True,
            capstyle="round",
        )
    ax.fill_between(
            X_new.ravel(),
            bot,
            top,
            fc=color,
            zorder=3,
            alpha=.1,
            lw=0,
            ec=color,
            aa=True,
            capstyle="round",
        )
    ax.fill_between(
        X_new.ravel(),
        bot,
        top,
        fc='none',
        zorder=4,
        alpha=1,
        lw=1.5,linestyle='--',
        ec=color,
        aa=True,
        capstyle="round",#hatch=''
    )
    
    bot = np.nanpercentile(inference, 33/2, axis=0)
    top = np.nanpercentile(inference, 100-33/2, axis=0)
    
    ax.fill_between(
        X_new.ravel(),
        bot,
        top,
        fc=color,
        zorder=3,
        alpha=.1,
        lw=0,
        ec=color,
        aa=True,
        capstyle="round",
    )
    
    ax.fill_between(
        X_new.ravel(),
        bot,
        top,
        fc='none',
        zorder=4,
        alpha=1,
        lw=1.5,linestyle='-',
        ec=color,
        aa=True,
        capstyle="round",#hatch=''
    )
    
    if plot_max_like:
        ax.plot(X_new.ravel(),max_like,zorder=13,color=color,lw=4)
    
    ## make legend here
    ax.plot([],[],color=color,lw=1.5,linestyle='--',label='95% GMSL envelope')
    ax.plot([],[],color=color,lw=1.5,linestyle='-',label='66% GMSL envelope')
    ax.plot([],[],color=color,lw=4,label='Most likely GMSL')
    

    
#     lig_only=((X_new<128) & (X_new>117)).ravel()
#     ax.plot(X_new,max_like,
#              zorder=13,color=color,lw=4)
    

    return ax


def weighted_trace(pred_list, comp, var="f_pred", iters=20000):
    weighted_trace =[]
    
    for i in range(iters):
        choice = np.random.choice(np.arange(len(comp)), 1, p=comp['weight'])
        key=comp.index[choice][0]
        f_preds = pred_list[key][var]
        C=np.random.choice(np.arange(len(f_preds)), 1)
        gmsl=f_preds[C].ravel()
        weighted_trace.append(gmsl)
        
    weighted_trace=np.array(weighted_trace)

    return weighted_trace

def plot_weights(dataframe, experiments, value, synthetic = False, holocene = False):
    sns.set_context("paper")
    if not holocene:
        fig = plt.figure(figsize=(12, 12))
    elif holocene:
        fig = plt.figure(figsize=(6, 2.2))

    if value == "hmc_divergence_pct":
        vmin = 0
        vmax = 100
        fun = np.array
    elif value == "weight":
        vmin = np.log(dataframe["weight"].values.astype(float)[np.nonzero(dataframe["weight"].values.astype(float))].min())
        vmax = 0
        fun = np.log
    else:
        vmin = 0
        vmax = 1
        fun = np.array
    axes = []
    for i in range(0, len(experiments)):
        experiment = experiments[i]
        if not holocene:
            axes.append(plt.subplot(6, 4, 1 + i))
        elif holocene:
            axes.append(plt.subplot(1, 2, 1 + i))

        experiment = experiments[i]
        filtered = dataframe[
            (dataframe.ice_history == experiment[0])
            & (dataframe.esl_curve == experiment[1])
            & (dataframe.Lithosphere == experiment[2])
        ]

        lmv = filtered["LMV"] 
        umv = filtered["UMV"] 
        vals = fun(np.array(filtered[value].values.astype(float)))

        lmv_vals, lmv_idx = np.unique(lmv, return_inverse=True)
        umv_vals, umv_idx = np.unique(umv, return_inverse=True)
        vals_array = np.empty(lmv_vals.shape + umv_vals.shape)
        vals_array.fill(np.nan)  # or whatever yor desired missing data flag is
        vals_array[lmv_idx, umv_idx] = vals
        vals_array = vals_array.T
        im = plt.imshow(
            vals_array, interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax
        )
        if synthetic==True:
            if any(filtered['true_model']):
                generative_model = filtered[filtered['true_model']==True]
                xi=np.sort(lmv.unique().astype(int)).ravel()
                yi=np.sort(umv.unique()).ravel()
                xp=np.where(xi==generative_model['LMV'][0])[0][0]
                yp=np.where(yi==generative_model['UMV'][0])[0][0]
                plt.plot(xp,yp,'X',c='k',markersize=20,zorder=5)
                
        _ = plt.gca().set_yticks(np.arange(umv.unique().size))
        _ = plt.gca().set_xticks(np.arange(lmv.unique().size))
        _ = plt.gca().set_yticklabels(np.sort(umv.unique()))
        _ = plt.gca().set_xticklabels(np.sort(lmv.unique()))
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)

        if i > len(experiments) - 5:
            plt.gca().set_xlabel("LMV $10^{21}$ Pa·s")
        else:
            plt.gca().set_xticklabels([])
        if i % 4 == 0:
            plt.gca().set_ylabel("UMV\n$10^{21}$ Pa·s")
        else:
            plt.gca().set_yticklabels([])

        plt.minorticks_off()
        if not holocene:
            a = int(132.7-2.7-16.6 - filtered["SIS"][0])
            a = str(a)
            b = str(experiment[2])
            if experiment[1] == "S":
                c = "Standard"
            else:
                c = "Slow"
            title = (
                "LIS: " + a + "m\nT-II Rate: " + c + "\nLithosphere Thickness: " + b + "km"
            )
            #         plt.gca().set_title(experiment[0] + "_" + experiment[1] + "_" + str(experiment[2]))
            plt.gca().set_title(title)
        elif holocene:
            b = str(experiment[2])            
            title = (
                "Lithosphere Thickness: " + b + "km"
            )
            #         plt.gca().set_title(experiment[0] + "_" + experiment[1] + "_" + str(experiment[2]))
            plt.gca().set_title(title)
        plt.gca().grid(False)

        style = {"lw": 1, "c": "k"}
        plt.plot([-0.5, 7.5], [1.5, 1.5], **style)
        plt.plot([-0.5, 7.5], [0.5, 0.5], **style)
        plt.plot([-0.5, 7.5], [2.5, 2.5], **style)
        plt.plot([-0.5, 7.5], [-0.5, -0.5], **style)
        for i in range(9):
            plt.plot([-0.5 + i, -0.5 + i], [-0.5, 2.5], **style)

    #     plt.suptitle(str(value)+' status by experiment',fontsize=10,y=.85)
    cb=fig.colorbar(
        im,
        shrink=0.05,
        label="log(Weight)",
        ax=axes,
        orientation="horizontal",
        anchor=(-.15,7.23),
        aspect=3
    )
    cb.ax.xaxis.set_label_position('top')
    if holocene:
        fig.suptitle('Weight of each GIA model given the Holocene data',y=.82,x=.55)
    fig.tight_layout()
    fig.tight_layout(pad=0, w_pad=2, h_pad=-10)

    return fig


def get_model_status(inference_dir,model_dir,year=2021):

    model_posterior_dir = str(inference_dir)+'/'+str(f'arviz_traces_{year}')
    model_posterior_list=[o[:-3] for o in os.listdir(model_posterior_dir) if '.nc' in o]

    model_predict_dir = str(inference_dir)+'/'+str(f'pymc3_post_predict_{year}')
    model_predict_list=[o[:-4] for o in os.listdir(model_predict_dir) if '.pkl' in o]

    model_files_dir = str(model_dir)
    model_files_raw=[o for o in os.listdir(model_files_dir) if '.dat' in o]
    unique_models=list(set(['_'.join(a.split('_')[:-1]) for a in model_files_raw]))


    model_weights = pd.read_csv(str(inference_dir)+'/'+str('model_weights/model_weights.csv'),index_col=0)

    models={}
    for u in unique_models:
        if '_new' in u:
            u_proc=u.replace('_new','')
        else:
            u_proc = u
        models[u]={}
        
        if '3D' not in u:

            models[u]['Lithosphere']= int(u_proc.split('output')[1].split('.dat')[0].split('Cp')[0])
            models[u]['UMV']= int(u_proc.split('output')[1].split('.dat')[0].split('Cp')[1][0])
            models[u]['LMV']= int(u_proc.split('output')[1].split('.dat')[0].split('Cp')[1][1:].split('_')[0])
            models[u]['ice_history']= u_proc.split('output')[1].split('.dat')[0].split('_')[1]
            models[u]['esl_curve']= u_proc.split('output')[1].split('.dat')[0].split('Wael_')[1][0]

        else:
            models[u]['Lithosphere']= 99
            models[u]['UMV']= 99
            models[u]['LMV']= 99
            models[u]['ice_history']= '3D GIA'
            models[u]['esl_curve']= '3D GIA'

        if any([u.split('output')[1] in mpl for mpl in model_posterior_list]):
            models[u]['posterior_trace']= True
        else:
            models[u]['posterior_trace']= False
        if any([u.split('output')[1] in mpl for mpl in model_predict_list]):
            models[u]['posterior_predict']= True
        else:
            models[u]['posterior_predict']= False
        if u in list(model_weights.index):
            models[u]['weight']=model_weights.loc[u]['weight']
        else:
            models[u]['weight']=0



    models_df = pd.DataFrame.from_dict(models).T
    return models_df  


def load_data():
    """
    Load data.

    Parameters
    ----------
    
    Returns
    -------
    data: pandas.DataFrame
        Dataframe containing sea level observations.

    """

    data = pd.read_csv(data_dir / "processed/gmsl_inference_data.csv")
    
    return data

def load_model(name, output_dir = 'get_GIA/', rsl_dir = 'output_new'):
    """
    Load GIA model.

    Parameters
    ----------
    name: str
    Model filename
    
    Returns
    -------
    rsl
    age
    model_dims

    """

    lats = pd.read_csv(model_dir / output_dir / "lats", delimiter=",", header=None)
    lons = pd.read_csv(model_dir / output_dir / "lons", delimiter=",", header=None)
    directory = model_dir / output_dir / rsl_dir
    age = np.arange(115, 131, 1)

    extent = [0, 1, 0, 1]
    model_dims = [
        np.min(lons.values),
        np.max(lons.values),
        np.min(lats.values),
        np.max(lats.values),]

    files = np.sort(os.listdir(directory))
    files = [f for f in files if 'output' in f]  # ignores non output
    
    files = [f for f in files if name in f]

    rsl = []
    age=[]
    for i in range(0, len(files)):
        rsl.append(
        pd.read_csv(
            str(directory) + "/" + files[i], delimiter=",", header=None
        ).values
        )
        age.append(float(files[i].split('ka')[0].split('_')[-1]))
    return rsl, age, model_dims

def interpolation_functions(LAT, LON, GIA_MODEL, age, model_dims):
    """
    Creates interpolated GIA functions at each sample location for a specific GIA model.

    Parameters
    ----------
    LAT:
    LON:
    GIA_MODEL:
    age:
    model_dims:
    
    Returns
    -------
    Zfuns: list
        A list of scipy.UnivariateSplines.

    """
    island_Zs = [
        [lookup_z(lat, lon, m, model_dims) for lat, lon in zip(LAT, LON)]
        for m in GIA_MODEL
    ]
    island_Zs = np.array(island_Zs)
    Zfuns = []
    for k in range(island_Zs.shape[1]):
        rsl_function = UnivariateSpline(age, island_Zs[:, k], k=1, ext=3, s=0)
        Zfuns.append(rsl_function)  ## 3 returns boundary value at extrapolation
    return Zfuns

def inference_model(data, z_functions, keys = ["coral", "highstand"], holocene=False):
    """
    Create the PyMC3 GP regression model.

    Parameters
    ----------
    data: Pandas.DataFrame
        A pandas dataframe containing at least age, age_uncertainty, elevation, elevation_uncertainty, 
        and type fields. If type is coral, water depth max (m) and water depth mean (m) should be set
        according to the species or outcrop context.
    z_functions: scipy.UnivariateSpline
        Interpolation functions for the GIA curve over time at each sample location for the specified GIA
        model.
    keys: list
        A list of the unique sample 'types' from data that should be included in this inference model.
    
    Returns
    -------
    model: pymc3.Model
        PyMC Model object
    gp: pymc3.gp.Marginal
        The gaussian process prior for this model. The gp object is used to generate posterior predictions
        across the LIG.

    """
    
    with pm.Model(check_bounds=False) as model: #create pymc3 model
        
        
        #### Create Gaussian Process Prior
        #### LIG
        if not holocene:
            ## hyper-parameters
            gp_ls = pm.Wald("gp_ls", mu=2, lam=5, shape=1) #lengthscale of covariance kernel
            gp_var = pm.Normal("gp_var", mu=0, sd=5, shape=1) #variance of covariance kernel
            m_gmsl = pm.Normal("m_gmsl", 0, 10) #mean gmsl

            ## mean and covariance functions
            mean_fun = pm.gp.mean.Constant(m_gmsl) #mean function for gp
            cov1 = gp_var[0]**2 * pm.gp.cov.ExpQuad(1, gp_ls[0]) #cov kernel. variance forced to positive

            ## GP prior
            gp = pm.gp.Marginal(mean_func=mean_fun,cov_func=cov1) #gp prior
        #### Holocene
        elif holocene:
            ## For holocene we force GMSL deviation from ESL curve to 0
            mean_fun = pm.gp.mean.Constant(0)
            gp = pm.gp.Marginal(mean_func=mean_fun, cov_func=pm.gp.cov.Constant(0))
            
        #### Create sample elevation priors
        ELEVATION = shared(data["elevation"].values)
        ELEVATION_U = shared(data["elevation_uncertainty"].values)
        elevations_sd = pm.Normal("elev_sd", 0, 1, shape=(data['age'].size))
        elevations = pm.Deterministic("elev", ELEVATION + elevations_sd * ELEVATION_U)
        
        #### Create sample age priors
        age_sd = {}
        age = {}
        ## Loop through each data type in keys
        for key in keys: 
            type_filter = data["type"].values == key
            AGE = data[type_filter]["age"].values
            AGE_U = data[type_filter]["age_uncertainty"].values
            N = data[type_filter]["age"].size
            
            # age priors by data type
            if (key == "coral" or key == "index" or key == "marine"): #normal age errors for LIG ONLY corals or index points
                if not holocene: ## normal ages bounded by GIA model LIG bounds
                    BoundedNormal = pm.Bound(pm.Normal, lower=117, upper=128)
                    age[key] = BoundedNormal(str(key + "_age"), mu=shared(AGE), sd=shared(AGE_U), shape=(N))
                elif holocene: ## unbounded normal ages used in holocene fitting
                    age_sd[key] = pm.Normal(str(key + "_age_sd"), 0, 1, shape=(N))
                    age[key] = pm.Deterministic(
                        str(key + "_age"), shared(AGE) + age_sd[key] * shared(AGE_U)
                    ) 
        
            elif (key == "highstand" or key == "highstand_marine" or key == "ordinary berm"):
                age_sd[key] = pm.Wald(str(key + "_age_sd"), mu=2, lam=5, shape=(N), testval=.1)
                age[key] = pm.Deterministic(
                    str(key + "_age"), shared(AGE)-1 + age_sd[key]
                )  # reshaped to improve Hamiltonian Monte Carlo, likely not needed in new version
            else:
                print("data type not implemented or key error, check dataframe")

        ## collect ages from all types of data
        ages = [age[x] for x in keys]
        ages = pm.Deterministic("ages", tt.concatenate(ages))

        #### GIA corrections for each sample
        ## One correction per sample
        N = data["age"].size
        GIA = tt.zeros(N, dtype="float64")
        
        ## interpolate the fixed time-step model runs to the estimated GIA at sampled age
        for i in range(N):
            GIA = tt.set_subtensor(GIA[i], SplineWrapper(z_functions[i])(ages[i]))
            
        ## Collect GIA corrections to be logged into trace
        gia_collect = pm.Deterministic(
            "GIA", GIA
        )

        #### Priors for water depth or indicative range for each sample type in keys
        water_depth_sd = {}
        water_depth = {}
        ## Loop through each data type in keys
        for key in keys:
            type_filter = data["type"].values == key
            N = data[type_filter]["age"].size
            if key == "coral":
                mu = data[type_filter]["Param 1"]
                lam = data[type_filter]["Param 2"]
                water_depth[key] = pm.Wald(
                    str(key + "_water_depth"), mu=mu, lam=lam, shape=(N)
                )
                water_depth[key]=water_depth[key]
                
            elif (key == "marine" or key == "highstand_marine"):
#                 water_depth[key] = pm.HalfFlat(str(key + "_water_depth"), shape=(N))
                water_depth[key] = pm.HalfCauchy(str(key + "_water_depth"), beta=5, shape=(N))
                

            elif (key == "highstand" or key == "index"): #no added water depth
                water_depth[key] = pm.Deterministic(
                    str(key + "_water_depth"), shared(np.zeros(N))
                )
            elif (key == "ordinary berm"):
                mu = -1*data[type_filter]["Param 1"]
                sd = data[type_filter]["Param 2"]
                water_depth[key] = pm.Normal(
                    str(key + "_water_depth"), mu=mu, sd=sd, shape=(N)
                )
                water_depth[key]=water_depth[key]
            else:
                print("data type not implemented or key error, check dataframe")

        ## collect water depths for logging to trace
        water_depths = [water_depth[x] for x in keys]
        water_depths = pm.Deterministic("water_depths", tt.concatenate(water_depths))
        
        ## long term subsidence
        if not holocene:
            subsidence = pm.Normal("subsidence", 2.5, 0.1)
        elif holocene:
            subsidence = 0.02*ages  #2.5 m per 125 ky

        #### The Master Equation:
        # GMSL = Elevation observation +/- elevation uncertainty +/- water depth - GIA + SUBSIDENCE
        # keep in mind we're solving for change in GMSL from the GMSL used in GIA model, which is zero
                
        gmsl_points = pm.Deterministic(
            "gmsl_points", elevations + subsidence + water_depths - GIA.flatten()
        )
        
        ## 'Geologic' noise -- or noise that is not fit by the model.

        ## noise = pm.InverseGamma('noise',alpha=3,beta=1,testval=.5)
        ## These two choices could be considered if you expect quite a bit more un-explained variance
        ## noise = pm.HalfStudentT('noise',nu=1,sigma=1)
        ## noise = pm.HalfFlat("noise")+0.01
        if not holocene:
            noise = pm.HalfStudentT('noise',nu=1,sigma=.15)
        elif holocene:
            noise = pm.HalfCauchy('noise',beta=5)
        
        if not holocene:
        ## Here we fit the GP defined above to the age and GMSL values sampled by the inference model
            gmsl_inference = gp.marginal_likelihood(
                "gmsl",
                X=ages[:, np.newaxis],
                y=gmsl_points,
                shape=((N),),
                noise=noise,
            )  # GMSL 
        elif holocene:
            gmsl_inference = pm.Normal('gmsl',mu=0,sd=noise,observed=gmsl_points,shape=((N),))
        
    return model, gp



def load(file):
    """
    Custom load command for pickle objects.

    Parameters
    ----------
    file: str or path to file

    Returns
    -------
    Object saved through pickle.

    """
    with open(file, "rb") as input_file:
        return pickle.load(input_file)


def GMSL(trace, model_name, ax=None, Xr=None, n_new=200):
    """
    Returns a figure axis with a GMSL envelope for a given GIA model.

    Parameters
    ----------
    trace: a collection of GMSL realizations a from pymc3 trace.
    model_name: the GIA model that corresponds to this the trace
    ax: a matplotlib axis if you do not wish to create a new axis
    Xr: An array that contains the ages that correspond to each value in a single trace
    n_new: If no Xr is provided, n_new sets the age resolution on the default age range

    Returns
    -------
    ax: a matplotlib axis containing the GMSL envelope figure.

    """
    if Xr == None:
        X_new = np.linspace(128, 117, n_new)[:, None]
    else:
        X_new = Xr
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    smoothing = 1  # 1000 y smoothing

    timescale = -1 * np.diff(X_new.ravel())[0]
    bot = np.percentile(
        gaussian(np.array(trace["f_pred"], timescale), smoothing), 2.5, axis=0
    )
    top = np.percentile(
        gaussian(np.array(trace["f_pred"], timescale), smoothing), 97.5, axis=0
    )
    mean = np.mean(gaussian(np.array(trace["f_pred"], timescale), smoothing), axis=0)

    ax.fill_between(
        X_new.ravel(),
        bot,
        top,
        color=(0.2, 0.2, 0.2),
        zorder=3,
        alpha=0.5,
        label="GMSL (2$\sigma$)",
        lw=0,
    )

    ax.legend(loc="upper right", frameon=False)
    ax.set_ylabel("GMSL (m)")
    ax.invert_xaxis()
    ax.set_xlabel("Age (kya)")
    ax.set_title("GMSL\n" + model_name[:-4])
    ax.set_ylim([-5, 20])
    ax.set_xticks([130, 128, 126, 124, 122, 120, 118, 116])
    ax.set_xlim([129, 116])

    return ax


def RSL(trace, model_name, lat, lon, model, model_dims, ax=None, Xr=None, n_new=200):
    """
    Returns a figure axis with a RSL/local sea level envelope for a given GIA model at a specific lat/lon coord.


    Parameters
    ----------
    trace: a collection of GMSL realizations a from pymc3 trace.
    model_name: the GIA model that corresponds to this the trace
    lat: latitude of interest
    lon: longitude of interest
    ax: a matplotlib axis if you do not wish to create a new axis
    Xr: An array that contains the ages that correspond to each value in a single trace
    n_new: If no Xr is provided, n_new sets the age resolution on the default age range

    Returns
    -------
    ax: a matplotlib axis containing the GMSL envelope figure.

    """
    if any(Xr) == None:
        X_new = np.linspace(128, 117, n_new)[:, None]
    else:
        X_new = Xr
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    smoothing = 1  # 1000 y smoothing

    timescale = -1 * np.diff(X_new.ravel())[0]
    bot = np.percentile(
        gaussian(np.array(trace["f_pred"], timescale), smoothing), 2.5, axis=0
    )
    top = np.percentile(
        gaussian(np.array(trace["f_pred"], timescale), smoothing), 97.5, axis=0
    )
    mean = np.mean(gaussian(np.array(trace["f_pred"], timescale), smoothing), axis=0)

    nassau_RSL = [lookup_z(lat, lon, timeslice, model_dims) for timeslice in model]
    age = np.arange(115, 131, 1)
    RSL_dt = interp1d(age, nassau_RSL, kind="linear", fill_value="extrapolate")
    GIA_to_add = RSL_dt(X_new).ravel()

    bot = bot + GIA_to_add
    top = top + GIA_to_add
    mean = mean + GIA_to_add
    ax.fill_between(
        X_new.ravel(),
        bot,
        top,
        color=(0.2, 0.2, 0.2),
        zorder=3,
        alpha=0.5,
        label="RSL (2$\sigma$)",
        lw=0,
    )
    ax.plot(X_new.ravel(), GIA_to_add, color=(0.2, 0.2, 0.2), alpha=1, label="GIA")

    ax.legend(loc="best", frameon=False)
    ax.set_ylabel("RSL (m)")
    ax.invert_xaxis()
    ax.set_xlabel("Age (kya)")
    ax.set_title("RSL at (" + str(lat) + "," + str(lon) + ")\n" + model_name[:-4])
    ax.set_ylim([-5, 20])
    ax.set_xticks([130, 128, 126, 124, 122, 120, 118, 116])
    ax.set_xlim([129, 116])

    return ax

def lookup_z(lat, lon, model, model_dims):
    """
    Returns the RSL prediction at a specific lat, lon, on a specific GIA model timeslice.
    Parameters
    ----------
    lat: Latitude value
    lon: Longitude value
    model: A 2d matrix from a GIA model output representing a single timeslice.
    model_dims: The real word lat/lon dimensions of the model. [left, right, top, bottom]
    Returns
    -------
    The model RSL prediction nearest the lat, lon pair.
    """
    try:
        model=model.values
    except AttributeError:
        model=model
    lat_len = model.shape[0]
    lon_len = model.shape[1]
    lon_list = np.linspace(model_dims[0], model_dims[1], lon_len)
    lat_list = np.linspace(model_dims[3], model_dims[2], lat_len)
    lon_id = np.argmin(
        np.abs(np.linspace(model_dims[0], model_dims[1], lon_len) - (lon))
    )
    lat_id = np.argmin(
        np.abs(np.linspace(model_dims[3], model_dims[2], lat_len) - (lat))
    )
    return model[lat_id, lon_id]



def make_tidal_subplot(gca, tidal_data, tidal_time):
    """
    Generate tidal subplot for tide data and timeseries.
    """

    msl = np.mean(tidal_data)
    gca.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    gca.xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(tidal_time, tidal_data, lw=0.5, color=(0.4, 0.4, 0.4))
    _ = plt.gca().set_ylabel("meters")
    _ = plt.gca().set_yticks(
        [
            np.round(np.min(tidal_data), 2),
            np.round(msl, 2),
            np.round(np.max(tidal_data), 2),
        ]
    )
    for item in gca.get_xticklabels():
        item.set_fontsize(10)
    plt.plot(
        [tidal_time[0], tidal_time[-1]],
        [np.round(msl, 2), np.round(msl, 2)],
        lw=3,
        color=(0.2, 0.2, 0.2),
    )



def get_field_data():
    """
    Loads and prepares the field data from master_elevations.csv

    Parameters
    ----------
    none

    Returns
    -------
    field_island_data: RSL peak sea level elevation and location data for Bahamas field data
    fu: uncertainty (2-sigma) of the elevations in field_island_data
    island_data_combo: RSL peak sea level elevation and location data for Bahamas field data as well as satellite derived estimates of peak sea level
    cu: uncertainty (2-sigma) of the elevations in island_data_combo
    """
    island_data = pd.read_csv(data_dir / "processed/master_elevations.csv")
    added_field_u = 0.15  # somewhat arbitrary 'uncertainty' -- ie, how close do we think field obs is to max sea level?

    island_data_combo = island_data.values[:, [0, 1, 3, 4]][
        [0, 3, 5, 9, 14, 15, 16, 17, 18, 19, 20], :
    ]  # indices for specific islands
    cu = island_data.values[:, [2]][
        [0, 3, 5, 9, 14, 15, 16, 17, 18, 19, 20], :
    ]  # indices for specific islands

    field_island_data = island_data[island_data["Type"] == "Field"].values[
        :, [0, 1, 3, 4]
    ]
    fu = island_data[island_data["Type"] == "Field"].values[:, [2]] + added_field_u

    cu[-1 * len(fu) :] += added_field_u

    return field_island_data, fu, island_data_combo, cu


def extent_from_gdal(ds):
    """
    Extracts the extent (ie. model_dims) from a georeferenced raster.

    Parameters
    ----------
    ds: gdal dataset from gdal.Open(file)

    Returns
    -------
    [left_x,right_x,bot_y,top_y]
    """
    ##function for grabbing extent of geotiff for imshow
    left_x = ds.GetGeoTransform()[0]
    top_y = ds.GetGeoTransform()[3]
    bot_y = ds.GetGeoTransform()[3] + ds.RasterYSize * ds.GetGeoTransform()[5]
    right_x = ds.GetGeoTransform()[0] + ds.RasterXSize * ds.GetGeoTransform()[1]
    return [left_x, right_x, bot_y, top_y]


def label_image(im, norm_max, model, memory_buffer=1000, side_size=50):
    """
    Applied a convolutional neural network (model) to label the pixels of a raster dataset (im).

    Parameters
    ----------
    im: raster dataset
    norm_max: The normalizing value for the digital elevation data at the time of training. Typically this value would be equal to the maximum elevation of the training data.
    model: a CNN model generated and saved using tensorflow.keras
    memory_buffer: the number of images to load into memory at a time.

    Returns
    -------
    L2: labeled image
    """
    im2 = np.copy(im)
    im2 = clear_edges(im2, side_size, non_val=np.min(im2))
    xv, yv = np.where(im2 > 0)
    xv = np.array(xv).ravel()
    yv = np.array(yv).ravel()
    L2 = np.zeros(im.shape) - 1
    counts = np.ceil(xv.size / memory_buffer).astype(int)
    for i in range(counts):

        to_label_X = np.array(
            [
                im[
                    int(x - side_size / 2) : int(x + side_size / 2),
                    int(y - side_size / 2) : int(y + side_size / 2),
                ]
                for x, y in zip(xv[:memory_buffer], yv[:memory_buffer])
            ]
        )
        to_label_X = to_label_X.reshape(-1, side_size, side_size, 1)
        to_label_X = to_label_X / norm_max
        L = np.argmax(model.predict(to_label_X, verbose=1), axis=1)
        k = 0
        for x, y in zip(xv[:memory_buffer], yv[:memory_buffer]):
            L2[x, y] = L[k]
            k += 1
        xv = np.roll(xv, memory_buffer)
        yv = np.roll(yv, memory_buffer)
        display.clear_output(wait=True)
        print("finished chunk #" + str(i + 1) + " out of " + str(counts))
    return L2


def clear_edges(im, side_size, non_val=-1):
    """
    Sets a 'side_size' fixed-width boundary of 'im' to value 'non_val'. This function is used to 
    make sure training data for the CNN does not fall too close to the raster edge.

    Parameters
    ----------
    im: raster dataset
    side_size: width of boundary to clear.
    non_val: the raster value to set the boundary space.

    Returns
    -------
    im: copy of 'im' with 'side_size' fixed-width boundary set to value 'non_val'
    """
    im = np.copy(
        im
    )  # makes a copy of image, so that one is edited and other stays the same
    im[: int(side_size / 2), :] = non_val
    im[-1 * int(side_size / 2) :, :] = non_val
    im[:, : int(side_size / 2)] = non_val
    im[:, -1 * int(side_size / 2) :] = non_val
    return im
    
def inference_model_new(data, z_functions, keys = ["coral", "highstand"], holocene=False):
    """
    Create the PyMC3 GP regression model.

    Parameters
    ----------
    data: Pandas.DataFrame
        A pandas dataframe containing at least age, age_uncertainty, elevation, elevation_uncertainty, 
        and type fields. If type is coral, water depth max (m) and water depth mean (m) should be set
        according to the species or outcrop context.
    z_functions: scipy.UnivariateSpline
        Interpolation functions for the GIA curve over time at each sample location for the specified GIA
        model.
    keys: list
        A list of the unique sample 'types' from data that should be included in this inference model.
    
    Returns
    -------
    model: pymc3.Model
        PyMC Model object
    gp: pymc3.gp.Marginal
        The gaussian process prior for this model. The gp object is used to generate posterior predictions
        across the LIG.

    """
    
    with pm.Model(check_bounds=False) as model: #create pymc3 model
        
        
        #### Create Gaussian Process Prior
        #### LIG
        if not holocene:
            ## hyper-parameters
            gp_ls = pm.Wald("gp_ls", mu=2, lam=5, shape=1) #lengthscale of covariance kernel
            gp_var = pm.Normal("gp_var", mu=0, sd=50, shape=1) #variance of covariance kernel
            m_gmsl = pm.Normal("m_gmsl", 0, 200) #mean gmsl

            ## mean and covariance functions
            mean_fun = pm.gp.mean.Constant(m_gmsl) #mean function for gp
            cov1 = gp_var[0]**2 * pm.gp.cov.ExpQuad(1, gp_ls[0]) #cov kernel. variance forced to positive

            ## GP prior
            gp = pm.gp.Marginal(mean_func=mean_fun,cov_func=cov1) #gp prior
        #### Holocene
        elif holocene:
            ## For holocene we force GMSL deviation from ESL curve to 0
            mean_fun = pm.gp.mean.Constant(0)
            gp = pm.gp.Marginal(mean_func=mean_fun, cov_func=pm.gp.cov.Constant(0))
            
        #### Create sample elevation priors
        ELEVATION = shared(data["elevation"].values)
        ELEVATION_U = shared(data["elevation_uncertainty"].values)
        elevations_sd = pm.Normal("elev_sd", 0, 1, shape=(data['age'].size))
        elevations = pm.Deterministic("elev", ELEVATION + elevations_sd * ELEVATION_U)
        
        #### Create sample age priors
        age_sd = {}
        age = {}
        ## Loop through each data type in keys
        for key in keys: 
            type_filter = data["type"].values == key
            AGE = data[type_filter]["age"].values
            AGE_U = data[type_filter]["age_uncertainty"].values
            N = data[type_filter]["age"].size
            
            # age priors by data type
            if (key == "coral" or key == "index" or key == "limiting"): #normal age errors for LIG ONLY corals or index points
                if not holocene: ## normal ages bounded by GIA model LIG bounds
                    BoundedNormal = pm.Bound(pm.Normal, lower=117, upper=128)
                    age[key] = BoundedNormal(str(key + "_age"), mu=shared(AGE), sd=shared(AGE_U), shape=(N))
                elif holocene: ## unbounded normal ages used in holocene fitting
                    age_sd[key] = pm.Normal(str(key + "_age_sd"), 0, 1, shape=(N))
                    age[key] = pm.Deterministic(
                        str(key + "_age"), shared(AGE) + age_sd[key] * shared(AGE_U)
                    ) 
        
            elif (key == "highstand" or key == "highstand_marine"):
                age_sd[key] = pm.Wald(str(key + "_age_sd"), mu=2, lam=5, shape=(N), testval=.1)
                age[key] = pm.Deterministic(
                    str(key + "_age"), shared(AGE)-1 + age_sd[key]
                )  # reshaped to improve Hamiltonian Monte Carlo, likely not needed in new version
            else:
                print("data type not implemented or key error, check dataframe")

        ## collect ages from all types of data
        ages = [age[x] for x in keys]
        ages = pm.Deterministic("ages", tt.concatenate(ages))

        #### GIA corrections for each sample
        ## One correction per sample
        N = data["age"].size
        GIA = tt.zeros(N, dtype="float64")
        
        ## interpolate the fixed time-step model runs to the estimated GIA at sampled age
        for i in range(N):
            GIA = tt.set_subtensor(GIA[i], SplineWrapper(z_functions[i])(ages[i]))
            
        ## Collect GIA corrections to be logged into trace
        gia_collect = pm.Deterministic(
            "GIA", GIA
        )

        #### Priors for water depth or indicative range for each sample type in keys
        water_depth_sd = {}
        water_depth = {}
        ## Loop through each data type in keys
        for key in keys:
            type_filter = data["type"].values == key
            N = data[type_filter]["age"].size
            if key == "coral":
                mean = 2
                lam=5 
                max_depth= data[type_filter]["water depth max (m)"]
                rescale = max_depth/lam
                mean_conversion = data[type_filter]["water depth mean (m)"]/rescale
                lam = np.ones(N)*lam
                water_depth[key] = pm.Wald(
                    str(key + "_water_depth"), mu=mean, lam=lam, shape=(N)
                )
                water_depth[key]=water_depth[key]*rescale
            
            elif (key == "marine" or key == "highstand_marine"):
                water_depth[key] = pm.HalfFlat(str(key + "_water_depth"), shape=(N))
                
            elif key == "limiting":
                mean = 2
                lam=5 
  
                water_depth[key] = pm.Wald(
                    str(key + "_water_depth"), mu=mean, lam=lam, shape=(N)
                )
                water_depth[key]=-1*(water_depth[key]-1.15) #negative to make terrestrial, 1.15 sets max_like at 0


            elif (key == "highstand" or key == "index"): #no added water depth
                water_depth[key] = pm.Deterministic(
                    str(key + "_water_depth"), shared(np.zeros(N))
                )
            else:
                print("data type not implemented or key error, check dataframe")

        ## long term subsidence
        N=data["elevation"].values.size
        uplift = pm.Normal("uplift_master", 0, 1)
        uplift = np.ones(N)*uplift * shared(data["uplift_rate (std)"].values) + shared(data["uplift_rate (per ky)"].values)
        uplift = uplift*ages.flatten() ##had /1000 here for last iteration
        uplift_for_each = pm.Deterministic("uplift",uplift)
#         ## collect all through concat
#         water_depths = [water_depth[x] for x in keys]
#         water_depths = pm.Deterministic("water_depths", tt.concatenate(water_depths))
        
        ## collect water depths for logging to trace
        water_depths = [water_depth[x] for x in keys]
        water_depths = pm.Deterministic("water_depths", tt.concatenate(water_depths))
        
       

        #### The Master Equation:
        # GMSL = Elevation observation +/- elevation uncertainty +/- water depth - GIA + SUBSIDENCE
        # keep in mind we're solving for change in GMSL from the GMSL used in GIA model, which is zero
                
        gmsl_points = pm.Deterministic(
            "gmsl_points", elevations - uplift + water_depths - GIA.flatten()
        )
        
        ## 'Geologic' noise -- or noise that is not fit by the model.

        ## noise = pm.InverseGamma('noise',alpha=3,beta=1,testval=.5)
        ## These two choices could be considered if you expect quite a bit more un-explained variance
        ## noise = pm.HalfStudentT('noise',nu=1,sigma=1)
        ## noise = pm.HalfFlat("noise")+0.01
        noise = pm.HalfCauchy('noise',beta=5)
        
        ## Here we fit the GP defined above to the age and GMSL values sampled by the inference model
        gmsl_inference = gp.marginal_likelihood(
            "gmsl",
            X=ages[:, np.newaxis],
            y=gmsl_points,
            shape=((N),),
            noise=noise,
        )  # GMSL 
        
    return model, gp

