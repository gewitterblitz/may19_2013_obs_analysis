
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

cdict = {'red': [[0.0, 0.1450980392156863, 0.1450980392156863], [0.125, 0.16862745098039217, 0.16862745098039217], [0.25, 0.2901960784313726, 0.2901960784313726], [0.375, 0.34901960784313724, 0.34901960784313724], [0.5, 0.4588235294117647, 0.4588235294117647], [0.625, 0.6392156862745098, 0.6392156862745098], [0.75, 0.7450980392156863, 0.7450980392156863], [0.875, 0.8235294117647058, 0.8235294117647058], [1.0, 0.9764705882352941, 0.9764705882352941]], 'green': [[0.0, 0.12156862745098039, 0.12156862745098039], [0.125, 0.23137254901960785, 0.23137254901960785], [0.25, 0.4588235294117647, 0.4588235294117647], [0.375, 0.5411764705882353, 0.5411764705882353], [0.5, 0.6078431372549019, 0.6078431372549019], [0.625, 0.6784313725490196, 0.6784313725490196], [0.75, 0.6862745098039216, 0.6862745098039216], [0.875, 0.7215686274509804, 0.7215686274509804], [1.0, 0.9607843137254902, 0.9607843137254902]], 'blue': [[0.0, 0.29411764705882354, 0.29411764705882354], [0.125, 0.43529411764705883, 0.43529411764705883], [0.25, 0.49411764705882355, 0.49411764705882355], [0.375, 0.3568627450980392, 0.3568627450980392], [0.5, 0.3215686274509804, 0.3215686274509804], [0.625, 0.33725490196078434, 0.33725490196078434], [0.75, 0.3607843137254902, 0.3607843137254902], [0.875, 0.6235294117647059, 0.6235294117647059], [1.0, 0.9568627450980393, 0.9568627450980393]]}

cmap = LinearSegmentedColormap('extent_density_b13', segmentdata=cdict, N=256)
register_cmap(name='extent_density_b13', cmap=cmap)

cmap_r = reverse_colormap(cmap)
register_cmap(name='extent_density_b13_r',cmap=cmap_r)