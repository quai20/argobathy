import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.geodesic import Geodesic
land_color = [85/255, 92/255, 105/255]
land_feature=cfeature.NaturalEarthFeature(category='physical',name='land',scale='50m',facecolor=land_color)
plt.rcParams['axes.grid'] = True
import numpy as np
import xarray as xr
import pandas as pd
import gsw, os
import scipy.stats as stats
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
etopo = xr.open_dataset('/home1/datawork/kbalem/ETOPO1_Bed_g_gmt4.nc')

def plt_use_case(ixs,df1,df):
    df=df.squeeze()
    print(df.WMO.values,df.GROUNDING_DATE.values,df.GROUNDING_DEPTH.values,df.CYCLE_NUMBER.values, df.gebco_min.values, df.gebco_max.values)        
    
    fig=plt.figure(figsize=(16,8),tight_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.4,0.6],width_ratios=[1.3,1.1,0.6])
    ax = fig.add_subplot(gs[0,0], projection=ccrs.Mercator())
    ax.add_feature(land_feature, edgecolor=None)

    etopo.where((etopo['x']>df['LONGITUDE'].values-5)&(etopo['x']<df['LONGITUDE'].values+5)&(etopo['y']>df['LATITUDE'].values-5)&(etopo['y']<df['LATITUDE'].values+5),
                drop=True)['z'].plot(vmax=0,cmap=cmocean.cm.deep_r, cbar_kwargs={'shrink': 0.5},ax=ax, transform=ccrs.PlateCarree())
    etopo.where((etopo['x']>df['LONGITUDE'].values-5)&(etopo['x']<df['LONGITUDE'].values+5)&(etopo['y']>df['LATITUDE'].values-5)&(etopo['y']<df['LATITUDE'].values+5),
                drop=True)['z'].plot.contour(x='x',y='y',levels=np.arange(-df['GROUNDING_DEPTH']-10,-df['GROUNDING_DEPTH']+10),colors='y',ax=ax, transform=ccrs.PlateCarree())

    ax.plot(df['LONGITUDE'],df['LATITUDE'],'*r', markersize=10, markeredgecolor='k', alpha=1, transform=ccrs.PlateCarree())
    rads = [10,50,100,200]
    for r in rads:
        PTS=np.array(Geodesic().circle(lon=df['LONGITUDE'].values,lat=df['LATITUDE'].values,radius=r*1000,n_samples=50))
        ax.plot(PTS[:,0],PTS[:,1],'y-',linewidth=1, transform=ccrs.PlateCarree())
        ax.text(PTS[0,0],PTS[0,1],str(r),color='r',transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    x=df1['GROUNDING_DEPTH'].values    
    yb=((df1['gebco_min']+df1['gebco_max'])/2).values
    res = stats.linregress(x, yb)
    
    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(x,yb,'o',label=f"{len(x)} groundings", color='lightgrey', markersize=3, markeredgecolor='k', markeredgewidth=0.5)
    ax1.plot(x, res.intercept + res.slope*x, 'r',linewidth=0.5,label=f"R-squared: {res.rvalue**2:.3f}")
    #ax1.set_xlim([1000,5100])
    #ax1.set_ylim([1000,5100])
    ax1.legend()
    ax1.set_xlabel('Grounding depth [m]')
    ax1.set_ylabel('~Bathy [m]')
    ax1.plot(df['GROUNDING_DEPTH'],((df['gebco_min']+df['gebco_max'])/2),'*',color='red', markersize=8, markeredgecolor='k', markeredgewidth=0.5)
    
    d = ixs[ixs['wmo']==int(df['WMO'].values)]["dac"].values[0]
    urlt="/home/ref-argo/gdac/dac/"+d+"/"+f"{int(df['WMO'].values)}"+"/"+f"{int(df['WMO'].values)}"+"_Rtraj.nc"     
    dt = xr.open_dataset(urlt,engine='argo')            
    print(urlt)
        
    park_last = dt[['JULD','PRES']].where((dt['MEASUREMENT_CODE']==290.)&(dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True).sortby('JULD').isel(N_MEASUREMENT=-1) 
    park_end = dt['JULD'].where((dt['MEASUREMENT_CODE']==300.)&(dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True)    
    prof_start = dt['JULD'].where((dt['MEASUREMENT_CODE']==500.)&(dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True)    
    descent = dt[['CYCLE_NUMBER','JULD','PRES','LATITUDE','LONGITUDE']].where((dt['MEASUREMENT_CODE']==389.)&(dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True)
    ground = dt[['JULD','PRES']].where((dt['MEASUREMENT_CODE']==901.)&(dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True).sortby('JULD').isel(N_MEASUREMENT=-1) 
    
    ax2 = fig.add_subplot(gs[:,2])
    ax2.plot(park_last['JULD'],park_last['PRES'],'go')
    ax2.axvline(park_end.values,color='g',label='PET')
    ax2.plot(descent['JULD'],descent['PRES'],'-o',linewidth=0.5,label='Descent')    
    ax2.axvline(prof_start.values,color='y',label='AST')
    ax2.plot(ground['JULD'],ground['PRES'],'ro',label='grounding')
    ax2.invert_yaxis()
    ax2.set_ylabel('PRES')    
    ax2.set_xlim([park_last['JULD'].values-np.timedelta64(5,'m'),prof_start.values+np.timedelta64(5,'m')])
    
    ax2.set_xticks(ax2.get_xticks())
    a = ax2.get_xticklabels()
    ax2.set_xticklabels(a,rotation=45)    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))    
    ax2.legend()
        
    cyc = dt[['MEASUREMENT_CODE','CYCLE_NUMBER','JULD','PRES','LATITUDE','LONGITUDE']].where((dt['CYCLE_NUMBER']==df['CYCLE_NUMBER'].values),drop=True)
    ax3 = fig.add_subplot(gs[1,0:2])
    ax3.plot(cyc.JULD,cyc.PRES,'.',linewidth=2)
    ax3.plot(cyc.where((cyc['MEASUREMENT_CODE']==901.),drop=True).JULD,cyc.where((cyc['MEASUREMENT_CODE']==901.),drop=True).PRES,'or',label='grounding')
    ax3.invert_yaxis()
    ax3.set_ylabel('PRES')    
    ax3.set_xticks(ax3.get_xticks())
    a = ax3.get_xticklabels()
    ax3.set_xticklabels(a,rotation=45)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax3.legend()    