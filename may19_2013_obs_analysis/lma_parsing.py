#!/usr/bin/env python

"""
This script contains most of the functions I use for
parsing lma data files for my May 19, 2013 observational analysis

Date created: May 17, 2020 
"""

from datetime import datetime, timedelta
import numpy as np

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
