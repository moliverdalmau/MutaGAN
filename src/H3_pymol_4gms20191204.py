'''
​​© 2020-2022 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

This material may be only be used, modified, or reproduced by or for the U.S. Government pursuant to the license rights granted under the clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact the Office of Technology Transfer at JHU/APL.
'''
import re
import csv
import math
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
#from PIL import Image

try:
    import pymol
except:
    pass

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  matplotlib.cm import ScalarMappable
from  matplotlib.colors import LinearSegmentedColormap
from  matplotlib.colors import ListedColormap, BoundaryNorm
#=================================================
# declare functions
def RGBtoHSV (color):
    """ 'color' should be a three-element vector
        containing RGB values from 0 to 255
        The output will be a three element vector
        of HSV values where:
            H is from 0 to 360
            S is from 0 to 1
            V is from 0 to 255
    """
    r = float(color[0])
    g = float(color[1])
    b = float(color[2])
    min = np.min(np.array(color))
    max = np.max(np.array(color))
    v = max
    delta = max - min
    if( max != 0 ):
        s = delta / max
    else:
        s = 0
        h = -1
        return [h, s, None]
    if( r == max ):
        # between yellow & magenta
        h = (g - b) / delta
    elif( g == max ):
        # between cyan & yellow
        h = 2 + ( b - r ) / delta
    else:
        # between magenta & cyan
        h = 4 + ( r - g ) / delta
    # degrees
    h = h * 60
    if( h < 0 ):
        h = h + 360
    if not h:
        h = 0
    return [h,s,v]

def HSVtoRGB (color):
    """ 'color' should be a three-element vector
        containing HSV where:
            H is from 0 to 360
            S is from 0 to 1
            V is from 0 to 255
​
        The output will be a three-element tuple
        of RGB values ranging from 0 to 255
    """
    h = float(color[0])
    s = float(color[1])
    v = float(color[2])
    if( s == 0 ):
        # achromatic (grey)
        r = v
        g = v
        b = v
        return (r,g,b)
    # sector 0 to 5
    h = h/60
    i = np.floor( h )
    # factorial part of h
    f = h - i
    p = v * ( 1 - s )
    q = v * ( 1 - s * f )
    t = v * ( 1 - s * ( 1 - f ) )
    if( int(i) == 0 ):
        r = v
        g = t
        b = p
    elif( int(i) == 1 ):
        r = q
        g = v
        b = p
    elif( int(i) == 2 ):
        r = p
        g = v
        b = t
    elif( int(i) == 3 ):
        r = p
        g = q
        b = v
    elif( int(i) == 4 ):
        r = t
        g = p
        b = v
    elif( int(i) == 5 ):
        r = v
        g = p
        b = q
    return (int(r),int(g),int(b))

#-------------------------------------------------
def HEXtoRGB ( value ):
    """ input is the hex string for a color
        output is a tuple of RGB values from 0 to 255
        from: http://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

#-------------------------------------------------
def RGBtoHEX ( rgb ):
    """ input must be a tuple of RGB values from 0 to 255
        output is the hex string for that color
        from: http://stackoverflow.com/questions/214359/converting-hex-color-to-rgb-and-vice-versa
    """
    return '#%02x%02x%02x' % rgb

#-------------------------------------------------
def residue_name(resi,resn,name):
    if name == 'CA':
        return "%s" % resi

#-------------------------------------------------
def color_HA1(resi,resn,name,colors, HA1):
    if name == 'CA':
        value = colors[int(HA1[resi])]
        value = value.lstrip('#')
        lv = len(value)
        rgb = list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        pymol.cmd.set_color("HA1_%s_color" % resi, rgb)
        pymol.cmd.color("HA1_%s_color" % resi, "((chain A,C,E) and resi %s)" % resi)

#-------------------------------------------------
def color_HA2(resi,resn,name,colors, HA2):
    if name == 'CA':
        value = colors[int(HA2[resi])]
        value = value.lstrip('#')
        lv = len(value)
        rgb = list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        pymol.cmd.set_color("HA2_%s_color" % resi, rgb)
        pymol.cmd.color("HA2_%s_color" % resi, "((chain B,D,F) and resi %s)" % resi)

#-------------------------------------------------
def shape_HA1(resi,resn,name,freq, HA1, min_freq, aa_oneletter):
    if name == 'CA':
        if freq[int(HA1[resi])] > min_freq:
            aa_name = "%s%d-HA1_%s" % (aa_oneletter[resn], HA1[resi]+1, resi)
            pymol.cmd.select(aa_name, "(chain A,C,E) and (resi %s)" % resi)
            pymol.cmd.show("spheres", aa_name)

#-------------------------------------------------
def shape_HA2(resi,resn,name,freq, HA2, min_freq, aa_oneletter):
    if name == 'CA':
        if freq[int(HA2[resi])] > min_freq:
            aa_name = "%s%d-HA2_%s" % (aa_oneletter[resn], HA2[resi]+1, resi)
            pymol.cmd.select(aa_name, "(chain B,D,F) and (resi %s)" % resi)
            pymol.cmd.show("spheres", aa_name)

#-------------------------------------------------
def get_freq(heatmap):
    # store data as numpy arrays
    v = np.array(heatmap, dtype=np.float)
    return v/np.max(v)

#-------------------------------------------------
def get_rainbow(n=101):
    # create color ramp
    x = 0.25 + 0.75*np.array(range(0,n), dtype=np.float)/(n-1)
    R = 255 * (0.472 - 0.567*x + 4.05*x**2) / (1 + 8.72*x - 19.17*x**2 + 14.1*x**3)
    G = 255 * (0.108932 - 1.22635*x + 27.284*x**2 - 98.577*x**3 + 163.3*x**4 - 131.395*x**5 + 40.634*x**6)
    B = 255 / (1.97 + 3.54*x - 68.5*x**2 + 243*x**3 - 297*x**4 + 125*x**5)
    # convert to HEX values (RGB -> HSV -> RGB -> HEX)
    hsv = []
    rgb = []
    for i in range(0,n):
        hsv.append(RGBtoHSV([R[i], G[i], B[i]]))
        hsv[i][1] = hsv[i][1]**0.25
        rgb.append(RGBtoHEX(HSVtoRGB(hsv[i])))
    return (rgb)

#-------------------------------------------------
def get_brown(n=101):
    # create color ramp
    x = np.array(range(0,n), dtype=np.float)/(n-1)
    def f(x):
        return math.erf(x)
    f2 = np.vectorize(f)
    R = 255 * (1 - 0.392 * (1 + f2((x-0.869)/0.255)))
    G = 255 * (1.021 - 0.456 * (1 + f2((x-0.527)/0.376)))
    B = 255 * (1 - 0.493 * (1 + f2((x-0.272)/0.309)))
    # convert to HEX values (RGB -> HSV -> RGB -> HEX)
    hsv = []
    rgb = []
    for i in range(0,n):
        hsv.append(RGBtoHSV([R[i], G[i], B[i]]))
        hsv[i][1] = hsv[i][1]**0.25
        rgb.append(RGBtoHEX(HSVtoRGB(hsv[i])))
    return (rgb)

#-------------------------------------------------
def get_colors(heatmap, zero_col, min_depth, shallow_col):
    # calculate mutation frequency
    freq = get_freq(heatmap)
    # send values that are less than the minimum depth, or zero change to minimum color
    ramp = freq
    # get rainbow
    rgb = get_brown(101)
    # identify colors for each value in our data
    def f(x): 
        return np.int(x)
    f2 = np.vectorize(f)
    colors = list(rgb[int(a)] for a in f2(np.round(ramp,2)*100))
    # set values where there are zero mutations to zero_col
    for i,value in enumerate(freq == 0):
        if value == True:
            colors[i] = zero_col
    # set values that are below the depth threshold to shallow_col
    for i,value in enumerate(freq < min_depth):
        if value == True:
            colors[i] = shallow_col
    return (colors)


def main(args):
    data_input = args.input
    min_depth = args.depth
    min_freq = args.freq
    out_file = args.out
    show = args.show
    
    zero_col = "#b3b3b3"
    shallow_col = "#b3b3b3"
    
    aa_oneletter = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    # read in total data
    file_heatmap = "%s" % data_input
    heatmap = []
    with open(file_heatmap, 'r') as opened:
        reader = csv.reader(opened, delimiter='\t')
        for line in reader:
            heatmap.append(int(line[0]))
    
    #-------------------------------------------------
    
    # get colors
    colors = get_colors(heatmap,zero_col, min_depth, shallow_col)

    #=================================================
    # get model
    pymol.cmd.load("../data/4gms.pdb")

    # set up selections
    pymol.cmd.select("flu", "(chain A,B,C,D,E,F)")
    pymol.cmd.select("fab", "(chain H,I,J,L,M,N)")
    pymol.cmd.select("AB", "(chain A,B)")
    pymol.cmd.select("CD", "(chain C,D)")
    pymol.cmd.select("EF", "(chain E,F)")
    pymol.cmd.select("HA1", "(chain A,C,E)")
    pymol.cmd.select("HA2", "(chain B,D,F)")
    
    # rotate to the proper orientation
    pymol.cmd.viewport(1440, 960)
    pymol.cmd.orient("flu")
    pymol.cmd.zoom("flu")
    
    #-------------------------------------------------
    # set up residue dictionaries
    
    # In 4GMS, HA1 starts at 10 and goes to 325
    # Across full-length HA, this corresponds to amino acid positions: 26 to 341 (starting with 1)
    myspace = {'residue_name': residue_name, 'HA1': []}
    pymol.cmd.iterate('(chain A)', 'HA1.append(residue_name(resi,resn,name))', space=myspace)
    # this first filters out the NA values from the residue_name output
    # then creates a dictionary with the appropriate color values
    HA1 = dict(zip(filter(lambda a: a != None, myspace['HA1']), range(25,341)))
    
    # In 4GMS, HA2 starts at 1 and goes to 171
    # Across full-length HA, this corresponds to amino acid positions: 346 to 516 (starting with 1)
    myspace = {'residue_name': residue_name, 'HA2': []}
    pymol.cmd.iterate('(chain B)', 'HA2.append(residue_name(resi,resn,name))', space=myspace)
    # this first filters out the NA values from the residue_name output
    # then creates a dictionary with the appropriate color values
    HA2 = dict(zip(filter(lambda a: a != None, myspace['HA2']), range(345,516)))
    
    
    #-------------------------------------------------
    # create image, properly colored
    
    # get mutation frequencies
    freq = get_freq(heatmap)
    # hide default view
    pymol.cmd.hide("everything")
    # change display (e.g. cartoon, surface)
    if show == "cartoon-mesh":
        pymol.cmd.show("cartoon", "flu")
        pymol.cmd.show("mesh", "flu")
    else:
        pymol.cmd.show(show, "flu")
    # re-color residues
    myspace = {'color_HA1': color_HA1, 'colors': colors, 'HA1':HA1}
    pymol.cmd.iterate('(chain A)', 'color_HA1(resi,resn,name,colors, HA1)', space=myspace)
    myspace = {'color_HA2': color_HA2, 'colors': colors, 'HA2':HA2}
    pymol.cmd.iterate('(chain B)', 'color_HA2(resi,resn,name,colors, HA2)', space=myspace)
    # show spheres if mutation frequency above threshold
    myspace = {'shape_HA1': shape_HA1, 'freq': freq, 'HA1':HA1, 'min_freq':min_freq, 'aa_oneletter':aa_oneletter}
    pymol.cmd.iterate('(chain A)', 'shape_HA1(resi,resn,name,freq, HA1, min_freq, aa_oneletter)', space=myspace)
    myspace = {'shape_HA2': shape_HA2, 'freq': freq, 'HA2':HA2, 'min_freq':min_freq, 'aa_oneletter':aa_oneletter}
    pymol.cmd.iterate('(chain B)', 'shape_HA2(resi,resn,name,freq, HA2, min_freq, aa_oneletter)', space=myspace)
    #-------------------------------------------------
    # save image
    pymol.cmd.set('ray_shadows', 0)
    pymol.cmd.set('ray_opaque_background', 0)
    pymol.cmd.ray(2400, 2000)
    filename = '%s.png' % (out_file)
    pymol.cmd.png(filename, width=1200, height=1000, dpi=300, ray=1)

    plt.rcParams.update({'font.size': 16})

    colorvaluepairs=list(set([(h,c) for h,c in zip(heatmap,colors)]))
    colorvaluepairs = pd.DataFrame(colorvaluepairs).sort_values(0)
    vals=colorvaluepairs[0].tolist()
    cols=colorvaluepairs[1].tolist()

    bounds = np.append(vals, vals[-1] + 1)

    cmap = ListedColormap(cols)
    norm = BoundaryNorm(bounds, ncolors=len(cols))

    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,4]})
    fig.subplots_adjust(bottom=0.5)

    cmap = LinearSegmentedColormap.from_list('', list(zip(np.array(vals)/vals[-1], cols)))
    norm = plt.Normalize(vals[0], vals[-1])
    steps = int(max(vals)/12)
    steps = int(np.ceil(steps/25))*25
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(0, max(vals), steps),
                 cax=ax[0], orientation='vertical')
    fig.tight_layout()
    img = mpimg.imread(filename)
    img=img[100:-100,50:-50]
    img=ndimage.rotate(img,90*3)
    ax[1].imshow(img)
    plt.axis('off')
    plt.savefig(re.sub(".png","_colorbar.png",filename))
    return re.sub(".png","_colorbar.png",filename)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
 
def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from the output of MutaGAN models.')
    
    parser.add_argument("-i", dest="input", help="path to file containing single column vector for heatmap", metavar="INPUT")
    parser.add_argument("-d", dest="depth", help="residues below this depth will be colored grey", metavar="DEPTH", type=int)
    parser.add_argument("-f", dest="freq", help="residues above this mutation frequency will be drawn as spheres", metavar="FREQ", type=float)
    parser.add_argument("-o", dest="out", help="file for output image", metavar="OUT")
    parser.add_argument("-s", dest="show", help="display mode (e.g. cartoon, surface)", metavar="SHOW")
    
    args = parser.parse_args()
    main(args)
