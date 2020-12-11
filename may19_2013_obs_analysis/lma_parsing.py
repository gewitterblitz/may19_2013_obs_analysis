#!/usr/bin/env python

"""
This script contains most of the functions I use for
parsing lma data files for my May 19, 2013 observational analysis

Date created: May 17, 2020 
"""

from datetime import datetime, timedelta
import numpy as np
from may19_2013_obs_analysis.utils import scan_vol_time,sec2time,ceil
import os,glob
import subprocess
from lmatools.flashsort.gen_autorun import logger_setup, sort_files
from lmatools.grid.make_grids import grid_h5flashfiles, dlonlat_at_grid_center, write_cf_netcdf_latlon, write_cf_netcdf_3d_latlon
from lmatools.vis.multiples_nc import make_plot, make_plot_3d, read_file_3d
from six.moves import map

def time2sec(f):
    date_start = datetime.strptime(f"{f.split('/')[-1].split('_')[2]}", "%H%M%S")
    interval_left = timedelta(
        hours=date_start.hour, minutes=date_start.minute, seconds=date_start.second
    ).total_seconds()
    for i in range(600):
        interval_right = interval_left + 60
        yield [interval_left, interval_right]
        interval_left = interval_right
        
def bbox_lma_data(d):
    if (d >= 200000) & (d < 201000):
        ll_lon = -98.1
        ul_lon = -97.6
        ll_lat = 35.35
        ul_lat = 35.8
    if (d >= 201000) & (d < 202000):
        ll_lon = -98
        ul_lon = -97.3
        ll_lat = 35.35
        ul_lat = 35.8
    if (d >= 202000) & (d < 203000):
        ll_lon = -98
        ul_lon = -97.3
        ll_lat = 35.35
        ul_lat = 35.8
    if (d >= 203000) & (d < 204000):
        ll_lon = -97.9
        ul_lon = -97.2
        ll_lat = 35.4
        ul_lat = 35.9
    if (d >= 204000) & (d < 205000):
        ll_lon = -97.8
        ul_lon = -97.1
        ll_lat = 35.4
        ul_lat = 36
    if (d >= 205000) & (d < 210000):
        ll_lon = -97.75
        ul_lon = -97.1
        ll_lat = 35.45
        ul_lat = 36
    if (d >= 210000) & (d < 211000):
        ll_lon = -97.8
        ul_lon = -97
        ll_lat = 35.4
        ul_lat = 36
    if (d >= 211000) & (d < 212000):
        ll_lon = -97.7
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36
    if (d >= 212000) & (d < 213000):
        ll_lon = -97.6
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36
    if (d >= 213000) & (d < 214000):
        ll_lon = -97.6
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36.1
    if (d >= 214000) & (d < 215000):
        ll_lon = -97.5
        ul_lon = -96.6
        ll_lat = 35.5
        ul_lat = 36.2
    if (d >= 215000) & (d < 220000):
        ll_lon = -97.4
        ul_lon = -96.6
        ll_lat = 35.55
        ul_lat = 36.1
    if (d >= 220000) & (d < 221000):
        ll_lon = -97.15
        ul_lon = -96.6
        ll_lat = 35.62
        ul_lat = 36.1
    if (d >= 221000) & (d < 222000):
        ll_lon = -97.25
        ul_lon = -96.75
        ll_lat = 35.63
        ul_lat = 36.1
    if (d >= 222000) & (d < 223000):
        ll_lon = -97
        ul_lon = -96.5
        ll_lat = 35.62
        ul_lat = 36.1
    return ll_lon, ul_lon, ll_lat, ul_lat


def point_to_line_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) +
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        #        print('endpoint distance reported')
        return 9999


def tfromfile(name):                #reads h5files and their indices
    parts = name.split('_')
    y, m, d = list(map(int, (parts[-3][0:2], parts[-3][2:4], parts[-3][4:6])))
    H, M, S = list(map(int, (parts[-2][0:2], parts[-2][2:4], parts[-2][4:6])))
    return y+2000,m,d,H,M,S


##### BE VERY CAREFUL HERE! THE INDICES CHANGE AS THE FOLDER NAME etc ARE CHANGED
#### Always name the sorting folder without an underscore '_' sign

def tfromfile1(name):          #reads grid files (2d and 3d) and their indices
    
    """ This function was added because the original one did not have the 
    functionality to make date specific sub-folders for the plots folder.
    The same functionality existed for grid files though.

    The indices differ from those in tfromfile function because the names of 
    files (h5files as input to tfromfile) have different structure than the 
    nc_names_2d and nc_names_3d files (input to tfromfile1 function).
    """
    parts = name.split('_')
    y, m, d = list(map(int, (parts[4][2:4], parts[4][4:6], parts[4][6:8])))
    H, M, S = list(map(int, (parts[5][0:2], parts[5][2:4], parts[5][4:6])))
    return y+2000,m,d,H,M,S


def grab_h5_files(radar):
    """ extract information for all the h5 files corresponding to the 
    scan duration of each radar volume scan"""
    x,y,date_start,date_end = scan_vol_time(radar)[2:]
    
    
    b1, b2 = datetime.strptime(sec2time(x)[0:8],'%H:%M:%S'),datetime.strptime(sec2time(y)[0:8],'%H:%M:%S')
    
    c1 = ceil(b1); c2 = ceil(b2)
    
    if c1==c2:
        
        c = c1.strftime('%H%M%S')
        
        interval_left = timedelta(hours=date_start.hour,minutes=date_start.minute,seconds=date_start.second).total_seconds()
        interval_left = interval_left + date_start.microsecond/10**6
        
        interval_right = timedelta(hours=date_end.hour,minutes=date_end.minute,seconds=date_end.second).total_seconds()
        interval_right = interval_right + date_end.microsecond/10**6
    
        old = radar.time['units'][14:33]
        new = old.replace("T"," ")
        
        date = new[0:10]
        
        dt = datetime.strptime(date,'%Y-%m-%d')
        new_dt = dt.strftime('%y%m%d')
        
        final_dt = new_dt + '_' + c
        h5_filenames = [os.path.expanduser('~/Desktop/mount/May_19_2013_all_stuff/May19_LMA_sorted/flashsort/h5_files/2013/May/19/LYLOUT_%s_0600.dat.flash.h5'%final_dt)]
        
        frame_interval = date_end - date_start
        frame_interval  = frame_interval.total_seconds()

        return h5_filenames,frame_interval

    if c1!=c2:
        interval_left1  = timedelta(hours=date_start.hour,minutes=date_start.minute,seconds=date_start.second).total_seconds()
        interval_left1 = interval_left1 + date_start.microsecond/10**6
        
        interval_right1 = timedelta(hours=date_end.hour,minutes=round(date_end.minute,-1),seconds=0).total_seconds()
    
        interval_left2 = timedelta(hours=date_end.hour,minutes=round(date_end.minute,-1),seconds=0).total_seconds()

        interval_right2 = timedelta(hours=date_end.hour,minutes=date_end.minute,seconds=date_end.second).total_seconds()
        interval_right2 = interval_right2 + date_end.microsecond/10**6
        
        old = radar.time['units'][14:33]
        new = old.replace("T"," ")
        
        date = new[0:10]
        
        dt1 = datetime.strptime(date,'%Y-%m-%d')
        new_dt1 = dt1.strftime('%y%m%d')
        final_dt1 = new_dt1 + '_' + c1.strftime('%H%M%S')
        
        dt2 = datetime.strptime(date,'%Y-%m-%d')
        new_dt2 = dt2.strftime('%y%m%d')
        final_dt2 = new_dt2 + '_' + c2.strftime('%H%M%S')
        
        h5_filenames = [os.path.expanduser('~/Desktop/mount/May_19_2013_all_stuff/May19_LMA_sorted/flashsort/h5_files/2013/May/19/LYLOUT_%s_0600.dat.flash.h5')%(final_dt1),
                        os.path.expanduser('~/Desktop/mount/May_19_2013_all_stuff/May19_LMA_sorted/flashsort/h5_files/2013/May/19/LYLOUT_%s_0600.dat.flash.h5'%(final_dt2))]

        frame_interval = date_end - date_start
        frame_interval  = frame_interval.total_seconds()

        return h5_filenames,frame_interval
    
    
    
def grid_lma_data(radar,h5_filenames, base_sort_dir, frame_interval,dx=0.5e3, 
                  dy=0.5e3, dz=0.5e3, x_bnd=(-74.75e3, 45.25e3),
                  y_bnd=(0.25e3, 120.25e3), z_bnd=(0.25e3, 20.25e3),ctr_lat=35.3331,  
                  ctr_lon=-97.2775,center_ID='KTLX',n_cols=2, base_date=None):         
    """ Given a list of HDF5 filenames (sorted by time order) in h5_filenames,
        create 2D and 3D NetCDF grids with spacing dx, dy, dz in meters,
        frame_interval in seconds, and tuples of grid edges
        x_bnd, y_bnd, and z_bnd in meters

        The actual grids are in regular lat,lon coordinates, with spacing at the
        grid center matched to the dx, dy values given.

        n_cols controls how many columns are plotted on each page.

        Grids and plots are written to base_sort_dir/grid_files/ and  base_sort_dir/plots/

        base_date is used to optionally set a common reference time for each of the NetCDF grids.
    """
    # not really in km, just a different name to distinguish from similar variables below.
    dx_km=dx
    dy_km=dy
    x_bnd_km = x_bnd
    y_bnd_km = y_bnd
    z_bnd_km = z_bnd

    grid_dir = os.path.join(base_sort_dir, 'grid_files20002300_500x500x500m_KTLX_nov2020')

    # There are similar functions in lmatools to grid on a regular x,y grid in some map projection.
    dx, dy, x_bnd, y_bnd = dlonlat_at_grid_center(ctr_lat, ctr_lon,
                                dx=dx_km, dy=dy_km,
                                x_bnd = x_bnd_km, y_bnd = y_bnd_km )

    start_time = scan_vol_time(radar)[4]
    end_time   = scan_vol_time(radar)[5]
    act_start =  scan_vol_time(radar)[4]
    date = start_time
    print (start_time, end_time)

    outpath = grid_dir+'/20%s' %(date.strftime('%y/%b/%d'))
    if os.path.exists(outpath) == False:
        os.makedirs(outpath)
        subprocess.call(['chmod', 'a+w', outpath, grid_dir+'/20%s' %(date.strftime('%y/%b')), grid_dir+'/20%s' %(date.strftime('%y'))])
    if True:
        grid_h5flashfiles(
                h5_filenames, start_time, end_time,frame_interval=frame_interval, proj_name='latlong',
                base_date = base_date, energy_grids=True,dx=dx, dy=dy, dz=dz, x_bnd=x_bnd, y_bnd=y_bnd, z_bnd=z_bnd_km,
                ctr_lon=ctr_lon, ctr_lat=ctr_lat, outpath = outpath,output_writer = write_cf_netcdf_latlon, 
                output_writer_3d = write_cf_netcdf_3d_latlon,output_filename_prefix=center_ID, spatial_scale_factor=1.0
                )

    # Create plots
    mapping = { 'source':'lma_source',
                'flash_extent':'flash_extent',
                'flash_init':'flash_initiation',
                'footprint':'flash_footprint',
                'specific_energy':'specific_energy',
                'flashsize_std':'flashsize_std',
                'total_energy': 'total_energy'
               }
    