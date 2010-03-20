import pylab as pl
from mpl_toolkits import basemap
import numpy as np
import colors
import matplotlib
reload(colors)

data = pl.csv2rec('Fy_input_091126_+_qryPvPR_MBG_with_covariates.csv')

vivax_data = data[np.where(data.datatype=='vivax')]
duffy_data = data[np.where(data.datatype!='vivax')]

# b = basemap.Basemap(-19,5,52,20, resolution='i')
b = basemap.Basemap(-19,5,52,40, resolution='i')

pl.close('all')
pl.figure(figsize=(12,3))

colors.map_axis_format(pl.subplot(1,2,1))
b.drawcoastlines(color=colors.map_outline)
b.drawcountries(color=colors.map_outline)
b.plot(vivax_data.lon, vivax_data.lat, linestyle='None', marker='.', color=colors.vivax_point, markersize=4, alpha=.2, label='vivax')
colors.map_legend_format(pl.legend(loc=0))


colors.map_axis_format(pl.subplot(1,2,2))
b.drawcoastlines(color=colors.map_outline)
b.drawcountries(color=colors.map_outline)
gen_data = duffy_data[np.where(duffy_data.datatype=='gen')]
phe_data = duffy_data[np.where(duffy_data.datatype=='phe')]
aphe_data = duffy_data[np.where(duffy_data.datatype=='aphe')]
b.plot(gen_data.lon, gen_data.lat, linestyle='None', marker='o', markersize=4, color=colors.duffy_point['gen'], alpha=.4, label='gen')
b.plot(phe_data.lon, phe_data.lat, linestyle='None', marker='D', markersize=4, color=colors.duffy_point['phe'], alpha=.4, label='phe')
b.plot(aphe_data.lon, aphe_data.lat, linestyle='None', marker='*', markersize=4, color=colors.duffy_point['aphe'], alpha=.4, label='aphe')
colors.map_legend_format(pl.legend(loc=0))

matplotlib.pyplot.subplots_adjust(wspace=.1, hspace=0, left=0, right=1, top=1, bottom=0)

pl.savefig('figs/datasets.pdf')