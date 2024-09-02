import numpy as np
import xarray as xr
from cartopy.geodesic import Geodesic
import pandas as pd
import gsw, os
import copernicus_marine_client as copernicus_marine

#Load Groundings
df1 = xr.open_dataset('working_ds_2.nc')

def get_coords_corrected_glo(i):
    du = df1.isel(N_GRD=i)
    # Set parameters for CMEMS Download : Glory Daily
    if du.GROUNDING_DATE < np.datetime64('2020-12-31'):
        ds_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    else :
        ds_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"        
    # Load xarray dataset from CMEMS
    vel_glo = copernicus_marine.open_dataset(
        dataset_id = ds_id,
        minimum_longitude = du.LONGITUDE.values-.2,
        maximum_longitude = du.LONGITUDE.values+.2,
        minimum_latitude = du.LATITUDE.values-.2,
        maximum_latitude = du.LATITUDE.values+.2,
        start_datetime = str(np.datetime64(du.GROUNDING_DATE.values,'D')),
        end_datetime = str(np.datetime64(du.GROUNDING_DATE.values,'D')),
        variables = ["uo","vo"]
    )    
    # Duration
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9)
    # Velocity interpolation & depth selection
    uom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['uo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    vom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['vo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    # Drift correction
    azim = np.degrees(np.arctan2(uom,vom))
    dist = np.sqrt(uom**2 + vom**2)
    origin = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim,dist)    
    return origin[0][0], origin[0][1] 

def get_coords_corrected_cli(i):
    du = df1.isel(N_GRD=i)
    # Set parameters for CMEMS Download : Glory climatology
    ds_id = "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"    
    # Load xarray dataset from CMEMS
    vel_glo = copernicus_marine.open_dataset(
        dataset_id = ds_id,
        minimum_longitude = du.LONGITUDE.values-.2,
        maximum_longitude = du.LONGITUDE.values+.2,
        minimum_latitude = du.LATITUDE.values-.2,
        maximum_latitude = du.LATITUDE.values+.2,
        start_datetime = '1993-'+str(pd.to_datetime(du.GROUNDING_DATE.values).month).zfill(2)+'-01',
        end_datetime = '1993-'+str(pd.to_datetime(du.GROUNDING_DATE.values).month).zfill(2)+'-01',
        variables = ["uo","vo"]
    )    
    # Duration
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9)
    # Velocity interpolation & depth selection
    uom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['uo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    vom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['vo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    # Drift correction
    azim = np.degrees(np.arctan2(uom,vom))
    dist = np.sqrt(uom**2 + vom**2)
    origin = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim,dist)    
    return origin[0][0], origin[0][1]

import multiprocessing

if __name__ == '__main__':    

    # GLORYS DAILY         
    pool0 = multiprocessing.Pool()
    results0 = pool0.map(get_coords_corrected_glo, df1.N_GRD.values)              
    #print(np.array(results)[:,0])        
    df1['LATITUDE_C0'] = xr.DataArray(np.array(results0)[:,1],dims='N_GRD')
    df1['LONGITUDE_C0'] = xr.DataArray(np.array(results0)[:,0],dims='N_GRD')
    
    # GLORYS MONTHLY CLIMATOLOGY
    pool1 = multiprocessing.Pool()
    results1 = pool1.map(get_coords_corrected_cli, df1.N_GRD.values)              
    #print(np.array(results)[:,0])        
    df1['LATITUDE_C1'] = xr.DataArray(np.array(results1)[:,1],dims='N_GRD')
    df1['LONGITUDE_C1'] = xr.DataArray(np.array(results1)[:,0],dims='N_GRD')  
    
    # Save results
    df1.to_netcdf('working_ds_nightpool.nc')
