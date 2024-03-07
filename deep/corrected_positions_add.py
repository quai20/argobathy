import numpy as np
import xarray as xr
from cartopy.geodesic import Geodesic
import pandas as pd
import gsw, os

def correct_from_glorysclim(du):
    #times
    ascent_duration = int((du.PROFILE_DATE - du.GROUNDING_DATE).values)/(1e9) #seconds
    drift_duration = 2000 #seconds
    #displacements
    iu = -1*ascent_duration*glomu['uo'].interp(latitude=du.LATITUDE.values,longitude=du.LONGITUDE.values).values
    iv = -1*ascent_duration*glomv['vo'].interp(latitude=du.LATITUDE.values,longitude=du.LONGITUDE.values).values
    ius = -1*drift_duration*glomus['uo'].interp(latitude=du.LATITUDE.values,longitude=du.LONGITUDE.values).values
    ivs = -1*drift_duration*glomvs['vo'].interp(latitude=du.LATITUDE.values,longitude=du.LONGITUDE.values).values    
    #surface drift correction
    azim1 = np.degrees(np.arctan2(ius,ivs))
    dist1 = np.sqrt(ius**2 + ivs**2)
    origin1 = Geodesic().direct([du.LONGITUDE.values,du.LATITUDE.values],azim1,dist1) 
    #water column correction
    azim0 = np.degrees(np.arctan2(iu,iv))
    dist0 = np.sqrt(iu**2 + iv**2)
    origin0 = Geodesic().direct([origin1[0][0], origin1[0][1]],azim0,dist0)
    
    return origin0[0][0], origin0[0][1]

glomu = xr.open_dataset('Glorys_mean/mercatorglorys12v1_gl12_mean_uo-ave2000-2020-ave-all-deps.nc')
glomv = xr.open_dataset('Glorys_mean/mercatorglorys12v1_gl12_mean_vo-ave2000-2020-ave-all-deps.nc')
glomus = xr.open_dataset('Glorys_mean/mercatorglorys12v1_gl12_mean_uo-ave2000-2020-surf.nc')
glomvs = xr.open_dataset('Glorys_mean/mercatorglorys12v1_gl12_mean_vo-ave2000-2020-surf.nc')

df1 = xr.open_dataset('DeepArvorGroundings_all_tids.nc')

lon_c2 = np.zeros(len(df1.N_GRD.values))
lat_c2 = np.zeros(len(df1.N_GRD.values))

for i in df1.N_GRD.values:
    try:
        lon_c2[i],lat_c2[i] = correct_from_glorysclim(df1.isel(N_GRD=i))
    except:
        print("some kind of error with ",i)           
    if(i%50==0):
        print(i)

df1['LATITUDE_C2'] = xr.DataArray(lat_c2,dims='N_GRD')
df1['LONGITUDE_C2'] = xr.DataArray(lon_c2,dims='N_GRD')

df1['LATITUDE_C2'].attrs = df1['LATITUDE'].attrs
df1['LATITUDE_C2'].attrs['long_name'] = 'Corrected latitude from Glorys 2000-2020 climatology velocities'
df1['LONGITUDE_C2'].attrs = df1['LATITUDE'].attrs
df1['LONGITUDE_C2'].attrs['long_name'] = 'Corrected longitude from Glorys 2000-2020 climatology velocities'

df1.to_netcdf('working_ds_2000-2020_positions.nc')
