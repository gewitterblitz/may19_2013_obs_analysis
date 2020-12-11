#!/usr/bin/env python

"""
This script contains most of the functions I use for
parsing radar data files for my May 19, 2013 observational analysis

Date created: May 17, 2020 
"""

from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from math import sin, cos, sqrt, atan2, radians
from skewt import SkewT
from dateutil.parser import parse
from may19_2013_obs_analysis.utils import scan_vol_time,sec2time,ceil,grab_time_intervals


def get_xarray_dataset(self, field):
    """
    Create an xarray dataset from PyART radar GRID object

    Arguments: 
                self: grid object
                field: field name string for which xarray dataset
                is desired

    TODO: Could make it more comprehensive by allowing 
    multiple fields in a single xarray datset

    """

    time = np.array([datetime.strptime(self.time['units'].split(
        ' ')[-1], '%Y-%m-%dT%H:%M:%SZ')], dtype='object')

    ds = xr.Dataset()

    field_data = self.fields[field]['data']
    data = xr.DataArray(np.ma.expand_dims(field_data, 0),
                        dims=('time', 'z', 'y', 'x'),
                        coords={'time': (['time'], time),
                                'z': (['z'], self.z['data']),
                                'y': (['y'], self.y['data']),
                                'x': (['x'], self.x['data']),
                                'lon': (['y', 'x'], self.get_point_longitude_latitude(0)[0]),
                                'lat': (['y', 'x'], self.get_point_longitude_latitude(0)[1])})

    ds[field] = data
    ds.close()

    return ds


# approximate radius of earth in km (needed for distance_from_radar function)
R = 6373.0


def distance_from_radar(self, target_lat):
    """
    Calculate perpendicular distance in N direction from radar 
    This is very specific to analysis done in
    x_sec_src_density_Z_ZDR.ipynb notebook

    The objective is to be able to find the distance from radar 
    to calculate the index of that grid point in xarray dataset 
    from which the latitude value can be calculated for plotting
    radar field (e.g. reflectivity field) overlaid by lma source
    density on a vertical cross section drawn in a E-W direction.

    Arguments: 
                radar: PyART radar object

                target_lat: Latitude at which the vertical xsec
                is desired to be plotted

    """
    lat1 = radians(self.latitude['data'])
    lon1 = radians(self.longitude['data'])
    lat2 = radians(target_lat)
    lon2 = radians(self.longitude['data'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000

    return distance


def plot_point(point, angle, length):
    '''
    Used to plot azimuth line on PPI map 

    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''
    angle = 90 - angle
    # unpack the first point
    x, y = point

    # find the end point
    endy = length * sin(radians(angle))
    endx = length * cos(radians(angle))

    return endx*1000, endy*1000  # convert km to meters


def check_sounding_for_montonic(sounding):
    """
    So the sounding interpolation doesn't fail, force the sounding to behave
    monotonically so that z always increases. This eliminates data from
    descending balloons.

    Used with csuradartools HID 
    """
    snd_T = sounding.soundingdata['temp']  # In old SkewT, was sounding.data
    snd_z = sounding.soundingdata['hght']  # In old SkewT, was sounding.data
    dummy_z = []
    dummy_T = []
    if not snd_T.mask[0]:  # May cause issue for specific soundings
        dummy_z.append(snd_z[0])
        dummy_T.append(snd_T[0])
        for i, height in enumerate(snd_z):
            if i > 0:
                if snd_z[i] > snd_z[i-1] and not snd_T.mask[i]:
                    dummy_z.append(snd_z[i])
                    dummy_T.append(snd_T[i])
        snd_z = np.array(dummy_z)
        snd_T = np.array(dummy_T)
    return snd_T, snd_z


def hid_grid(grid, sndfile):
    """
    Arguments:

    grid: Gridded radar object upon which HCA algo will be performed.

    sndfile: Sounding file to be used for vertical interpolation of temmperature profile
    """
    sounding = SkewT.Sounding(sndfile)

    snd_T, snd_z = check_sounding_for_montonic(sounding)

    test = np.interp(grid.z['data'], snd_z, snd_T)
    trial = np.ones([grid.ny, grid.nx])
    result = [x * trial for x in test]
    T_grid = np.asarray(result)

    grid.add_field('grid_T', {'data': T_grid})

    scores = csu_fhc.csu_fhc_summer(dz=grid.fields['reflectivity']['data'], zdr=grid.fields['differential_reflectivity']['data'],
                                    rho=grid.fields['cross_correlation_ratio']['data'], kdp=grid.fields['kdp']['data'],
                                    use_temp=True, band='S',
                                    T=grid.fields['grid_T']['data'])
    fh = np.argmax(scores, axis=0) + 1
    grid.add_field(
        'hid', {'data': fh, 'mask': grid.fields['reflectivity']['data'].mask})

    locs = np.where(grid.fields['hid']['mask'] == True)
    aa = list(zip(locs[0], locs[1], locs[2]))

    for i in range(len(aa)):
        grid.fields['hid']['data'][aa[i]] = 1e10

    return grid


def adjust_fhc_colorbar_for_pyart(cb):
    """
    Used to create labeled colorbar for csuradartools HID product
    """
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb


def interval(radar):
    """ 

    Slightly different version of grab_time_intervals. Needed to
    correctly extract the datetime object for manual charge classification
    data files

    """
    x, y, date_start, date_end = scan_vol_time(radar)[2:6]

    b1, b2 = datetime.strptime(sec2time(x)[0:8], '%H:%M:%S'), datetime.strptime(
        sec2time(y)[0:8], '%H:%M:%S')

    c1 = ceil(b1)
    c2 = ceil(b2)

    if c1 == c2:

        c = c1.strftime('%H%M%S')

        interval_left = timedelta(
            hours=date_start.hour, minutes=date_start.minute, seconds=date_start.second).total_seconds()
        interval_left = interval_left + date_start.microsecond/10**6

        interval_right = timedelta(
            hours=date_end.hour, minutes=date_end.minute, seconds=date_end.second).total_seconds()
        interval_right = interval_right + date_end.microsecond/10**6

        old = radar.time['units'][14:33]
        new = old.replace("T", " ")

        date = new[0:10]

        dt = datetime.strptime(date, '%Y-%m-%d')
        new_dt = dt.strftime('%y%m%d')

        final_dt = new_dt + '_' + c

        return interval_left, interval_right

    if c1 != c2:
        interval_left1 = timedelta(
            hours=date_start.hour, minutes=date_start.minute, seconds=date_start.second).total_seconds()
        interval_left1 = interval_left1 + date_start.microsecond/10**6

        interval_right1 = timedelta(hours=date_end.hour, minutes=round(
            date_end.minute, -1), seconds=0).total_seconds()

        interval_left2 = timedelta(hours=date_end.hour, minutes=round(
            date_end.minute, -1), seconds=0).total_seconds()

        interval_right2 = timedelta(
            hours=date_end.hour, minutes=date_end.minute, seconds=date_end.second).total_seconds()
        interval_right2 = interval_right2 + date_end.microsecond/10**6

        old = radar.time['units'][14:33]
        new = old.replace("T", " ")

        date = new[0:10]

        dt1 = datetime.strptime(date, '%Y-%m-%d')
        new_dt1 = dt1.strftime('%y%m%d')
        final_dt1 = new_dt1 + '_' + c1.strftime('%H%M%S')

        dt2 = datetime.strptime(date, '%Y-%m-%d')
        new_dt2 = dt2.strftime('%y%m%d')
        final_dt2 = new_dt2 + '_' + c2.strftime('%H%M%S')

        return interval_left1, interval_left2, interval_right1, interval_right2

    
def extent_pid(*args):
    """

    This function provides the lat lon extent limits 
    for different durations during which the Edmond-Carney 
    supercell was analyzed SPECIFICALLY FOR PID ANALYSIS.

    The utility is to restrict our computations only to the domain
    within which this isolated storm was propagating avoiding 
    contamination from storms that developed South of the EC storm
    
    It differs from extent_of_interest function in the sese that the
    bounds or limits specified in this function were manually picked
    looking at the low-level KTLX reflectivity field RATHER THAN 
    looking at the extent of lightning activity as done in the former case.
    
    Arguments:
    ----------
    Arguments kept as optional because only one argument is needed for radar or LMA gridded file

    args: It expects the datetime object corresponding to the start
    of the radar volume scan or the time corresponding to the 
    gridded lma data files

    Define corners of bounding box of EC storm for each 10 min window (based on LMA data)

    Ignore the date (year, month, day) values since they don't matter here. Only time matters
    """
    args = args[0]
    
    if (args >= datetime(2013, 5, 19, 20, 0, 0)) & (args < datetime(2013, 5, 19, 20, 5, 0)):
        ll_lat = 35.4
        ul_lat = 35.75
        ll_lon = -98.10
        ul_lon = -97.65

    elif (args >= datetime(2013, 5, 19, 20, 5, 0)) & (args < datetime(2013, 5, 19, 20, 10, 0)):
        ll_lat = 35.4
        ul_lat = 35.8
        ll_lon = -98
        ul_lon = -97.6

    elif (args >= datetime(2013, 5, 19, 20, 10, 0)) & (args < datetime(2013, 5, 19, 20, 15, 0)):
        ll_lat = 35.4
        ul_lat = 35.8
        ll_lon = -98
        ul_lon = -97.6

    elif (args >= datetime(2013, 5, 19, 20, 15, 0)) & (args < datetime(2013, 5, 19, 20, 20, 0)):
        ll_lat = 35.4
        ul_lat = 35.9
        ll_lon = -98
        ul_lon = -97.5

    elif (args >= datetime(2013, 5, 19, 20, 20, 0)) & (args < datetime(2013, 5, 19, 20, 25, 0)):
        ll_lat = 35.4
        ul_lat = 35.9
        ll_lon = -98
        ul_lon = -97.5

    elif (args >= datetime(2013, 5, 19, 20, 25, 0)) & (args < datetime(2013, 5, 19, 20, 30, 0)):
        ll_lat = 35.4
        ul_lat = 36.05
        ll_lon = -98
        ul_lon = -97.4

    elif (args >= datetime(2013, 5, 19, 20, 30, 0)) & (args < datetime(2013, 5, 19, 20, 35, 0)):
        ll_lat = 35.4
        ul_lat = 35.9
        ll_lon = -98
        ul_lon = -97.3

    elif (args >= datetime(2013, 5, 19, 20, 35, 0)) & (args < datetime(2013, 5, 19, 20, 40, 0)):
        ll_lat = 35.5
        ul_lat = 35.9
        ll_lon = -97.9
        ul_lon = -97.25


    elif (args >= datetime(2013, 5, 19, 20, 40, 0)) & (args < datetime(2013, 5, 19, 20, 45, 0)):
        ll_lat = 35.5
        ul_lat = 36
        ll_lon = -97.8
        ul_lon = -97.25

    elif (args >= datetime(2013, 5, 19, 20, 45, 0)) & (args < datetime(2013, 5, 19, 20, 49, 0)):
        ll_lat = 35.5
        ul_lat = 36
        ll_lon = -97.8
        ul_lon = -97.2

    elif (args >= datetime(2013, 5, 19, 20, 49, 0)) & (args < datetime(2013, 5, 19, 20, 50, 0)):
        ll_lat = 35.5
        ul_lat = 36
        ll_lon = -97.8
        ul_lon = -97.15

    elif (args >= datetime(2013, 5, 19, 20, 50, 0)) & (args < datetime(2013, 5, 19, 20, 55, 0)):
        ll_lat = 35.45
        ul_lat = 36.1
        ll_lon = -97.8
        ul_lon = -97.1

    elif (args >= datetime(2013, 5, 19, 20, 55, 0)) & (args < datetime(2013, 5, 19, 21, 0, 0)):
        ll_lat = 35.5
        ul_lat = 36.1
        ll_lon = -97.8
        ul_lon = -97

    elif (args >= datetime(2013, 5, 19, 21, 0, 0)) & (args < datetime(2013, 5, 19, 21, 5, 0)):
        ll_lat = 35.5
        ul_lat = 36.2
        ll_lon = -97.8
        ul_lon = -97

    elif (args >= datetime(2013, 5, 19, 21, 5, 0)) & (args < datetime(2013, 5, 19, 21, 10, 0)):
        ll_lat = 35.5
        ul_lat = 36.2
        ll_lon = -97.7
        ul_lon = -97

    elif (args >= datetime(2013, 5, 19, 21, 10, 0)) & (args < datetime(2013, 5, 19, 21, 15, 0)):
        ll_lat = 35.55
        ul_lat = 36.2
        ll_lon = -97.7
        ul_lon = -96.9

    elif (args >= datetime(2013, 5, 19, 21, 15, 0)) & (args < datetime(2013, 5, 19, 21, 19, 0)):
        ll_lat = 35.5
        ul_lat = 36.2
        ll_lon = -97.65
        ul_lon = -96.85

    elif (args >= datetime(2013, 5, 19, 21, 19, 0)) & (args < datetime(2013, 5, 19, 21, 20, 0)):
        ll_lat = 35.55
        ul_lat = 36.2
        ll_lon = -97.6
        ul_lon = -96.8

    elif (args >= datetime(2013, 5, 19, 21, 20, 0)) & (args < datetime(2013, 5, 19, 21, 25, 0)):
        ll_lat = 35.55
        ul_lat = 36.2
        ll_lon = -97.6
        ul_lon = -96.8

    elif (args >= datetime(2013, 5, 19, 21, 25, 0)) & (args < datetime(2013, 5, 19, 21, 30, 0)):
        ll_lat = 35.58
        ul_lat = 36.2
        ll_lon = -97.6
        ul_lon = -96.8

    elif (args >= datetime(2013, 5, 19, 21, 30, 0)) & (args < datetime(2013, 5, 19, 21, 35, 0)):
        ll_lat = 35.6
        ul_lat = 36.2
        ll_lon = -97.5
        ul_lon = -96.75

    elif (args >= datetime(2013, 5, 19, 21, 35, 0)) & (args < datetime(2013, 5, 19, 21, 40, 0)):
        ll_lat = 35.6
        ul_lat = 36.2
        ll_lon = -97.5
        ul_lon = -96.75

    elif (args >= datetime(2013, 5, 19, 21, 40, 0)) & (args < datetime(2013, 5, 19, 21, 45, 0)):
        ll_lat = 35.6
        ul_lat = 36.3
        ll_lon = -97.5
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 21, 45, 0)) & (args < datetime(2013, 5, 19, 22, 5, 0)):
        ll_lat = 35.6
        ul_lat = 36.3
        ll_lon = -97.4
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 5, 0)) & (args < datetime(2013, 5, 19, 22, 10, 0)):
        ll_lat = 35.65
        ul_lat = 36.3
        ll_lon = -97.4
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 10, 0)) & (args < datetime(2013, 5, 19, 22, 15, 0)):
        ll_lat = 35.7
        ul_lat = 36.3
        ll_lon = -97.3
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 15, 0)) & (args < datetime(2013, 5, 19, 22, 30, 0)):
        ll_lat = 35.75
        ul_lat = 36.3
        ll_lon = -97.2
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 30, 0)) & (args < datetime(2013, 5, 19, 22, 40, 0)):
        ll_lat = 35.8
        ul_lat = 36.3
        ll_lon = -97.1
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 40, 0)) & (args < datetime(2013, 5, 19, 22, 45, 0)):
        ll_lat = 35.8
        ul_lat = 36.3
        ll_lon = -97
        ul_lon = -96.7

    elif (args >= datetime(2013, 5, 19, 22, 45, 0)) & (args < datetime(2013, 5, 19, 22, 50, 0)):
        ll_lat = 35.85
        ul_lat = 36.3
        ll_lon = -97
        ul_lon = -96.7

    else:
        ll_lat = np.nan
        ul_lat = np.nan
        ll_lon = np.nan
        ul_lon = np.nan
    
    return ll_lon,ul_lon,ll_lat,ul_lat