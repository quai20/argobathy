import numpy as np
import xarray as xr
from cartopy.geodesic import Geodesic
import pandas as pd
import gsw, os
import copernicus_marine_client as copernicus_marine

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
        start_datetime = '1993-01-01',
        end_datetime = '1993-12-31',
        variables = ["uo","vo"]
    )    
    vel_glo = vel_glo.mean('time')
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9)
    uom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['uo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    vom = -1*ascent_duration*vel_glo.where(vel_glo['depth']<du.GROUNDING_DEPTH.values,drop=True)['vo'].interp(latitude=du['LATITUDE'].values,
                                                                                                              longitude=du['LONGITUDE'].values).mean('depth').values
    azim = np.degrees(np.arctan2(uom,vom))
    dist = np.sqrt(uom**2 + vom**2)
    origin = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim,dist)    
    return origin[0][0], origin[0][1] #, vel_glo

df1 = xr.open_dataset('DeepArvorGroundings_all_tids.nc')

lon_c2 = np.zeros(len(df1.N_GRD.values))
lat_c2 = np.zeros(len(df1.N_GRD.values))

for i in df1.N_GRD.values:
    try:
        lon_c2[i],lat_c2[i] = get_coords_corrected_cli(df1.isel(N_GRD=i))
    except:
        print("some kind of error with ",i)           
    if(i%100==0):
        print(i)

df1['LATITUDE_C2'] = xr.DataArray(lat_c2,dims='N_GRD')
df1['LONGITUDE_C2'] = xr.DataArray(lon_c2,dims='N_GRD')

df1['LATITUDE_C2'].attrs = df1['LATITUDE'].attrs
df1['LATITUDE_C2'].attrs['long_name'] = 'Corrected latitude from Glorys climatology velocities'
df1['LONGITUDE_C2'].attrs = df1['LATITUDE'].attrs
df1['LONGITUDE_C2'].attrs['long_name'] = 'Corrected longitude from Glorys climatology velocities'

df1.to_netcdf('working_ds_third_positions.nc')
