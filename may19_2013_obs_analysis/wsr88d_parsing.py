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


def extent_of_interest(*args):
    """

    This function provides the lat lon extent limits 
    for different durations during which the Edmond-Carney 
    supercell was analyzed.

    The utility is to restrict our computations only to the domain
    within which this isolated storm was propagating avoiding 
    contamination from storms that developed South of the EC storm

    Also copied in the lma_parsing.py script

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

    if ((args >= datetime(1900, 1, 1, 20, 0, 0)) & (args < datetime(1900, 1, 1, 20, 10, 0))):
        ll_lon = -98.1
        ul_lon = -97.6
        ll_lat = 35.35
        ul_lat = 35.8
    if ((args >= datetime(1900, 1, 1, 20, 10, 0)) & (args < datetime(1900, 1, 1, 20, 20, 0))):
        ll_lon = -98
        ul_lon = -97.3
        ll_lat = 35.35
        ul_lat = 35.8
    if ((args >= datetime(1900, 1, 1, 20, 20, 0)) & (args < datetime(1900, 1, 1, 20, 30, 0))):
        ll_lon = -98
        ul_lon = -97.3
        ll_lat = 35.35
        ul_lat = 35.8
    if ((args >= datetime(1900, 1, 1, 20, 30, 0)) & (args < datetime(1900, 1, 1, 20, 40, 0))):
        ll_lon = -97.9
        ul_lon = -97.2
        ll_lat = 35.4
        ul_lat = 35.9
    if ((args >= datetime(1900, 1, 1, 20, 40, 0)) & (args < datetime(1900, 1, 1, 20, 50, 0))):
        ll_lon = -97.8
        ul_lon = -97.1
        ll_lat = 35.45
        ul_lat = 36
    if ((args >= datetime(1900, 1, 1, 20, 50, 0)) & (args < datetime(1900, 1, 1, 21, 0, 0))):
        ll_lon = -97.75
        ul_lon = -97.1
        ll_lat = 35.45
        ul_lat = 36
    if ((args >= datetime(1900, 1, 1, 21, 0, 0)) & (args < datetime(1900, 1, 1, 21, 10, 0))):
        ll_lon = -97.8
        ul_lon = -97
        ll_lat = 35.4
        ul_lat = 36
    if ((args >= datetime(1900, 1, 1, 21, 10, 0)) & (args < datetime(1900, 1, 1, 21, 20, 0))):
        ll_lon = -97.7
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36
    if ((args >= datetime(1900, 1, 1, 21, 20, 0)) & (args < datetime(1900, 1, 1, 21, 30, 0))):
        ll_lon = -97.6
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36
    if ((args >= datetime(1900, 1, 1, 21, 30, 0)) & (args < datetime(1900, 1, 1, 21, 40, 0))):
        ll_lon = -97.6
        ul_lon = -96.8
        ll_lat = 35.5
        ul_lat = 36.1
    if ((args >= datetime(1900, 1, 1, 21, 40, 0)) & (args < datetime(1900, 1, 1, 21, 50, 0))):
        ll_lon = -97.5
        ul_lon = -96.6
        ll_lat = 35.5
        ul_lat = 36.2
    if ((args >= datetime(1900, 1, 1, 21, 50, 0)) & (args < datetime(1900, 1, 1, 22, 0, 0))):
        ll_lon = -97.4
        ul_lon = -96.6
        ll_lat = 35.55
        ul_lat = 36.1
    if ((args >= datetime(1900, 1, 1, 22, 0, 0)) & (args < datetime(1900, 1, 1, 22, 10, 0))):
        ll_lon = -97.15
        ul_lon = -96.6
        ll_lat = 35.62
        ul_lat = 36.1
    if ((args >= datetime(1900, 1, 1, 22, 10, 0)) & (args < datetime(1900, 1, 1, 22, 20, 0))):
        ll_lon = -97.25
        ul_lon = -96.75
        ll_lat = 35.63
        ul_lat = 36.1
    if ((args >= datetime(1900, 1, 1, 22, 20, 0)) & (args < datetime(1900, 1, 1, 22, 30, 0))):
        ll_lon = -97
        ul_lon = -96.5
        ll_lat = 35.62
        ul_lat = 36.1

    return ll_lon, ul_lon, ll_lat, ul_lat


def scan_vol_time(radar):
    """ 
    extract scan time and duration of scan of radar

        Helpful while creating gridded lma products as per
        the duration of radar scan.

        Helpful for generating time stamp for title in radar plots

        Helpful for filtering h5 flash data and NLDN CG data
        to restrict only CGs within radar scan duration.

        Helpful for setting the extent of radar fields and gridded
        lma products within extent of interest.

    Arguments: PyART radar object corresponding to time of interest.

    Returns: start seconds of scan, end seconds of scan,
             start date, and end date

    Also copied in lma_parsing.py

    """
    # old = radar.time['units'][14:33]
    # new = old.replace("T", " ")
    date_start = parse(radar.time['units'], fuzzy=True).replace(tzinfo=None)
    # date_start = datetime.strptime(new, '%Y-%m-%d %H:%M:%S')
    date_end = date_start + timedelta(0, radar.time['data'].max())
    title = date_start.strftime('%H:%M:%S') + " - " + \
        date_end.strftime('%H:%M:%S')
    day_start = datetime(2013, 5, 19, 0, 0, 0)
    start_sec = (date_start - day_start).total_seconds()
    end_sec = (date_end - day_start).total_seconds()

    left = sec2time(round(start_sec))[0:8]
    right = sec2time(round(end_sec))[0:8]

    time_left = datetime.strptime(left, '%H:%M:%S')
    time_right = datetime.strptime(right, '%H:%M:%S')

#    return  start_sec, end_sec,date_start,date_end
    return time_left, time_right, start_sec, end_sec, date_start, date_end


def sec2time(sec, n_msec=3):
    ''' 
    Convert seconds to 'D days, HH:MM:SS.FFF' 

        Used in scan_vol_time, grab_time_intervals functions 
        in this package

        Sometimes also in generating title for radar plots

    Arguments: time in seconds 

    Also copied in lma_parsing.py

    '''
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)


def ceil(t):
    """ 
    need for grab_h5_files function to allow differentiation
    between a radar scan overlapping between two 10 minute 
    intervals as per lma data

        Used in grab_time_intervals function in this package

    Arguments: datetime object from radar scan

    Returns: Datetime object but with time rounded to the beginning
            of that 10 minute interval of argument time 

    Also copied in 88d_parsing.py 
    """

    if (t.minute % 10 >= 0):
        return t - timedelta(minutes=(t.minute % 10), seconds=(t.second % 60))
    else:
        return t


def grab_time_intervals(radar):
    """ 
    Extract the datetime limits of radar scan but with the caveat
    that if the scan duration was such that it crosses a 10 minute 
    mark (e.g. radar scan from 2028 to 2032 crosses the 2030 mark)

    In the latter case, this function would return two datetime 
    limits such that the first one is from start time to that 
    10 minute mark and the second one is from that 10 min mark
    to the end of radar scan time.

    (e.g. 2028-2032 would give (2028-2030) and (2030-2032))

    Arguments: PyART radar object

    Returns: 
            final_left1: left datetime limit of first range
            final_right1: right datetime limit of first range
            final_left2:  left datetime limit of second range
            final_right2: right datetime limit of second range

    Also copied in lma_parsing.py
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

        interval_right = timedelta(
            hours=date_end.hour, minutes=date_end.minute, seconds=date_end.second).total_seconds()

        interval_left1 = sec2time(interval_left)[0:8]
        interval_right1 = sec2time(interval_right)[0:8]

        time_left1 = datetime.strptime(interval_left1, '%H:%M:%S')
        time_right1 = datetime.strptime(interval_right1, '%H:%M:%S')

        return time_left1, time_right1  # ,frame_interval

    if c1 != c2:
        interval_left1 = timedelta(
            hours=date_start.hour, minutes=date_start.minute, seconds=date_start.second).total_seconds()

        interval_right1 = timedelta(hours=date_end.hour, minutes=round(
            date_end.minute, -1), seconds=0).total_seconds()

        interval_left2 = timedelta(hours=date_end.hour, minutes=round(
            date_end.minute, -1), seconds=0).total_seconds()

        interval_right2 = timedelta(
            hours=date_end.hour, minutes=date_end.minute, seconds=date_end.second).total_seconds()

        final_left1 = datetime.strptime(
            sec2time(interval_left1)[0:8], '%H:%M:%S')
        final_left2 = datetime.strptime(
            sec2time(interval_left2)[0:8], '%H:%M:%S')
        final_right1 = datetime.strptime(
            sec2time(interval_right1)[0:8], '%H:%M:%S')
        final_right2 = datetime.strptime(
            sec2time(interval_right2)[0:8], '%H:%M:%S')

        return final_left1, final_left2, final_right1, final_right2


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
