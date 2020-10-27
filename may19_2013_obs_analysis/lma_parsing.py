#!/usr/bin/env python

"""
This script contains most of the functions I use for
parsing lma data files for my May 19, 2013 observational analysis

Date created: May 17, 2020 
"""

from datetime import datetime, timedelta
import numpy as np


def extent_of_interest(*args):
    """

    This function provides the lat lon extent limits 
    for different durations during which the Edmond-Carney 
    supercell was analyzed.

    The utility is to restrict our computations only to the domain
    within which this isolated storm was propagating avoiding 
    contamination from storms that developed South of the EC storm

    Also copied in the 88d_parsing.py script

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

    Also copied in 88d_parsing.py

    """
    old = radar.time['units'][14:33]
    new = old.replace("T", " ")
    date_start = datetime.strptime(new, '%Y-%m-%d %H:%M:%S')
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

    Also copied in 88d_parsing.py

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

    Also copied in 88d_parsing.py
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
