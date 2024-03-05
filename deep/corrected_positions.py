import numpy as np
import xarray as xr
from cartopy.geodesic import Geodesic
import pandas as pd
import gsw, os
import copernicus_marine_client as copernicus_marine

def get_coords_corrected_glo(du):
    
    # Set parameters
    if du.GROUNDING_DATE < np.datetime64('2020-12-31'):
        ds_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    else :
        ds_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"        
    # Load xarray dataset
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
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9)
    uom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['uo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    vom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['vo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    azim = np.degrees(np.arctan2(uom,vom))
    dist = np.sqrt(uom**2 + vom**2)
    origin = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim,dist)    
    return origin[0][0], origin[0][1] #, vel_glo

def get_coords_corrected_cli(du):    
    # Set parameters
    ds_id = "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"    
    # Load xarray dataset
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
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9)
    uom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['uo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    vom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['vo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    azim = np.degrees(np.arctan2(uom,vom))
    dist = np.sqrt(uom**2 + vom**2)
    origin = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim,dist)    
    return origin[0][0], origin[0][1] #, vel_glo

df1 = xr.open_dataset('working_ds_2_tid_mindist_to_mb.nc')

lon_c0 = np.zeros(len(df1.N_GRD.values))
lat_c0 = np.zeros(len(df1.N_GRD.values))
lon_c1 = np.zeros(len(df1.N_GRD.values))
lat_c1 = np.zeros(len(df1.N_GRD.values))
for i in df1.N_GRD.values:
    try:
        lon_c0[i],lat_c0[i] = get_coords_corrected_glo(df1.isel(N_GRD=i))
    except:
        print("some kind of error with ",i)
    try:    
        lon_c1[i],lat_c1[i] = get_coords_corrected_cli(df1.isel(N_GRD=i))
    except:
        print("some kind of error with ",i)   
        
    if(i%100==0):
        print(i)

df1['LATITUDE_C0'] = xr.DataArray(lat_c0,dims='N_GRD')
df1['LATITUDE_C1'] = xr.DataArray(lat_c1,dims='N_GRD')
df1['LONGITUDE_C0'] = xr.DataArray(lon_c0,dims='N_GRD')
df1['LONGITUDE_C1'] = xr.DataArray(lon_c1,dims='N_GRD')

df1.to_netcdf('working_ds_tonight.nc')
