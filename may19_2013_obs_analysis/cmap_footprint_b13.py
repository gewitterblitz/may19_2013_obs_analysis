
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

import matplotlib.pyplot as plt
import matplotlib as mpl

# reverse function was taken from https://exceptionshub.com/invert-colormap-in-matplotlib.html
def reverse_colormap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r


cdict = {'red': [[0.0, 0.9490196078431372, 0.9490196078431372], [0.14285714285714285, 0.8666666666666667, 0.8666666666666667], [0.2857142857142857, 0.7333333333333333, 0.7333333333333333], [0.42857142857142855, 0.6862745098039216, 0.6862745098039216], [0.5714285714285714, 0.4627450980392157, 0.4627450980392157], [0.7142857142857143, 0.27058823529411763, 0.27058823529411763], [0.8571428571428571, 0.21176470588235294, 0.21176470588235294], [1.0, 0.08627450980392157, 0.08627450980392157]], 'green': [[0.0, 0.9490196078431372, 0.9490196078431372], [0.14285714285714285, 0.7372549019607844, 0.7372549019607844], [0.2857142857142857, 0.47843137254901963, 0.47843137254901963], [0.42857142857142855, 0.396078431372549, 0.396078431372549], [0.5714285714285714, 0.3176470588235294, 0.3176470588235294], [0.7142857142857143, 0.25882352941176473, 0.25882352941176473], [0.8571428571428571, 0.29411764705882354, 0.29411764705882354], [1.0, 0.08235294117647059, 0.08235294117647059]], 'blue': [[0.0, 0.8470588235294118, 0.8470588235294118], [0.14285714285714285, 0.5254901960784314, 0.5254901960784314], [0.2857142857142857, 0.5019607843137255, 0.5019607843137255], [0.42857142857142855, 0.6352941176470588, 0.6352941176470588], [0.5714285714285714, 0.592156862745098, 0.592156862745098], [0.7142857142857143, 0.5607843137254902, 0.5607843137254902], [0.8571428571428571, 0.40784313725490196, 0.40784313725490196], [1.0, 0.06274509803921569, 0.06274509803921569]]}

cmap = LinearSegmentedColormap('footprint_b13', segmentdata=cdict, N=256)
register_cmap(name='footprint_b13', cmap=cmap)

cmap_r = reverse_colormap(cmap)
register_cmap(name='footprint_b13_r',cmap=cmap_r)

