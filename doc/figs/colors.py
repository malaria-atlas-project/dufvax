map_outline = (.9,.7,.8)
# vivax_point = (.1,.4,.1)
vivax_point=(0,0,.2)
# duffy_point = {'gen': (.1,.1,.1), 'aphe': (.5,.5,.2), 'phe': (.3,.4,.6)}
duffy_point = {'gen': (0,0,.2), 'aphe': (0,0,.2), 'phe': (0,0,.2)}

def map_axis_format(ax):
    ax.axesPatch.set_facecolor('w')
    
def map_legend_format(l):
    l.legendPatch.set_facecolor((.9,.9,1))
    l.legendPatch.set_alpha(.5)
    