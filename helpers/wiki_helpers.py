from datetime import date
from datetime import datetime
from helpers.history_fetcher import HistoryFetcher
import dateutil.parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import folium
from scipy.stats.stats import pearsonr
import helpers.stats_helpers as stats_helpers
import itertools


def exponential_mle(data):
    return len(data) / sum(data)
    

def get_edits_dates_and_size(country, start, end):
    history_fetcher = HistoryFetcher(country)
    response = history_fetcher.get_history(start, end)

    edits_dates = list(map(lambda revision: (revision['timestamp'], np.log(abs(revision['change_size']) + 1)), response))

    return edits_dates


def get_edits_dates(country, start, end):
    history_fetcher = HistoryFetcher(country)
    response = history_fetcher.get_history(start, end)

    edits_dates = list(map(lambda revision: revision['timestamp'], response))

    return edits_dates


def get_edits_bins(country, start, end, number_of_bins):
    history_fetcher = HistoryFetcher(country)
    edits_history = history_fetcher.get_history(start, end)

    edits_history = list(map(lambda revision: (revision['timestamp'], (abs(revision['change_size']) + 1)), edits_history))

    unzipped_history = list(zip(*edits_history))
    n_wikis = plt.hist(unzipped_history[0], bins=number_of_bins, weights=unzipped_history[1])[0]

    edits_timeframed = list(sorted(n_wikis, key=lambda e: -e))

    plt.clf()

    return edits_timeframed


def get_index(countries, start, end):
    countries_history = [list(get_edits_dates(country, start, end)) for country in countries]

    n_wikis = [list(plt.hist(history, bins=1500)[0]) for history in countries_history]
    plt.clf()

    mles = [stats_helpers.exponential_mle(n_wiki) for n_wiki in n_wikis]
    samples_exponential = [stats_helpers.sample_exponential(n_wikis[i], mle) for i, mle in enumerate(mles)]

    pearson_coef = [pearsonr(group_data(n_wiki)[1], samples_exponential[i])[0] for i, n_wiki in enumerate(n_wikis)]

    countries_with_index = zip(countries, mles, pearson_coef)

    return sorted(countries_with_index, key=lambda e: e[1])


def group_data(data, normed=False):
    groups = [[k, len(list(g))] for k, g in itertools.groupby(sorted(data))]
    edits_per_bins, corres_bin_num = list(zip(*groups))

    if normed:
        corres_bin_num = [i / sum(corres_bin_num) for i in corres_bin_num]

    return edits_per_bins, corres_bin_num


#uses mle
def get_stability_for_country(country, start, end, K = 1500, plot = False):
    history = HistoryFetcher(country)
    dates = history.get_edits_dates(start, end)

    timeframes, x, y = plt.hist(dates, bins=K)
    if(plot == True):
        plt.show()

    lambda_hat = stats_helpers.exponential_mle(timeframes)

    # computing the mean wiki changes:

    date1 = datetime.strptime(start, '%Y%m%d%H%M%S')
    date2 = datetime.strptime(end, '%Y%m%d%H%M%S')

    nr_days = (date2 - date1).days
    mean_changes = len(dates)/nr_days

    return lambda_hat, mean_changes

# returns the estimated wikipedia page stability from year_start to November 2017 (now).
# The bigger the outlier factor, the more events are ignored (should be between 2 and 100)
# can plot analytical data
def wiki_change_factor(wiki_name,year_start, year_stop,outlier_factor, plot_on = False):

    # fetch the changes
    history_fetcher = HistoryFetcher(wiki_name)
    response = history_fetcher.get_history(str(year_start)+'0101000000', str(year_stop)+'1230000000')
    dates = list(map(lambda revision: revision['timestamp'], response))
    dates_pd = pd.DataFrame(dates, columns=['date'])
    dates_pd['change']= 1;

    # aggregate per month
    changes_aggregated_month = np.zeros([(2017 - year_start)*12,1])
    date_last = date(year=year_start,month=11,day=1)
    for y in range(year_start,year_stop):
        for m in range(0,12):
            date_current = date(year=y,month=m+1,day=1)
            changes_month = np.sum(dates_pd[ dates_pd['date'] > date_last][ dates_pd['date'] < date_current]['change'])
            changes_aggregated_month[-1+m+(y-year_start)*12]=changes_month
            date_last = date_current

    changes_aggregated_month = np.squeeze(changes_aggregated_month)

    thr_val = outlier_factor * np.mean(changes_aggregated_month)
    thr2_val = thr_val*0.7

    if(plot_on):
        plt.rcParams["figure.figsize"] = (5,4)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.bar(range(len(changes_aggregated_month)), changes_aggregated_month)
        plt.title('Wiki changes/month for ' + wiki_name + ' from ' + str(year_start) + ' till now')
        plt.axhline(thr_val, color="grey")
        plt.xlabel('months from '+ str(year_start) + ' till now')
        plt.ylabel('aggregated changes/month')

        indices = [i for i,v in enumerate(changes_aggregated_month >= thr_val) if v]
        plt.bar(indices, changes_aggregated_month[indices],color='red')
        plt.text(1.02,thr_val, 'summed as extrordinary\n changes above this line', va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())
        indices2 = [i for i,v in enumerate( (changes_aggregated_month >= thr2_val) & (changes_aggregated_month < thr_val)) if v]
        plt.bar(indices2, changes_aggregated_month[indices2],color='orange')
        plt.text(1.02,thr2_val, 'significant above avg\n changes above this line', va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())
        plt.axhline(thr2_val, color="grey")
        plt.show()

     # select outliers
    sum_outliers = np.sum(changes_aggregated_month[changes_aggregated_month > thr_val])
    sum_outliers += 0.2 * np.sum(changes_aggregated_month[ (changes_aggregated_month >= thr2_val) & (changes_aggregated_month < thr_val)])
    sum_all = np.sum(changes_aggregated_month)

    return (sum_outliers/sum_all)

# makes a folium map
def make_folium_map(json_map_path, object_path,  color_func, vmin, vmax, colors_table,location, zoom_start, legend_name  ):

    cantons_path = os.path.join('', json_map_path)

    topo_json_data = json.load(open(cantons_path))
    m = folium.Map(location=location, zoom_start=zoom_start)
    folium.TopoJson(
        topo_json_data,
        object_path=object_path,
        style_function=lambda feature: {
            'fillColor': color_func(feature['id']),
            'fillOpacity': 0.9,
            'line_opacity':0.3,
            'weight': 0.4,

            }
        ).add_to(m)
    linear = folium.colormap.StepColormap( colors=colors_table, vmin=vmin, vmax=vmax,  caption=legend_name).add_to(m)

    return m;


def get_country_values(aggregated_gdelt, cntr_code, normalize=True):
    values_to_plot = aggregated_gdelt[ (aggregated_gdelt['ActionGeo_CountryCode'] == cntr_code)];
    values_to_plot['ActionGeo_Type'] = 0;
    values_to_plot.groupby(['SQLDATE','ActionGeo_CountryCode', 'ActionGeo_Type']).sum()
    values_to_plot.reset_index(inplace=True)
    if(normalize):
        x = values_to_plot['SQLDATE']
        y = values_to_plot['Counter']/ values_to_plot['Counter'].mean()
    else:
        x = values_to_plot['SQLDATE']
        y = values_to_plot['Counter']
    return x.values,y.values

# computed distance between the line y=ax +c and the point x, y
def distance_from_line(a,c,x,y):
    a = -a;
    c=-c;
    return (a*x + y +c)/np.sqrt(a*a + 1)

def truncate_names(arr_string, n):
    for i in range (len(arr_string)):
        arr_string[i] = arr_string[i][:n]
    return arr_string

def plot_most(countries_data, feat_to_plot, title = '', sort=1, nr_top=30, figsize =(8,3)):
    codes = countries_data['Code'].values
    countries = countries_data['Country'].values

    cor_val = sort*countries_data[feat_to_plot].values
    instab = countries_data['Wiki Instability old'].values
    stab = countries_data['Wiki Stability MLE'].values

    # filtering outliers
    for i in range(len(instab)):
        if(instab[i] < 0.0002) | (stab[i] > 100.0):
            cor_val[i] = np.NaN

    colors= {'Africa':'red','Europe':'green','Americas':'blue', 'Oceania':'yellow','Asia':'magenta'}

    cor_val= np.array(cor_val)
    countries = np.array(countries)

    nans = np.isnan(cor_val)
    countries = countries[~nans]
    cor_val = cor_val[~nans]

    cor_val = cor_val.astype(float)
    idx = np.argsort(cor_val)
    cor_val = cor_val[idx]
    countries = countries[idx]
    cor_val = sort*cor_val[-nr_top:]
    countries = countries[-nr_top:]

    plt.figure(figsize=figsize)
    plt.bar(range(len(cor_val)), cor_val, align='center')

    for i, val in enumerate(cor_val):
        state= countries_data[countries_data['Country'] == countries[i]]['Region'].values[0]
        #plt.bar(i, val, align='center', color=wiki_changes_colors_eu(state))
        plt.bar(i, val, align='center', color=colors[state])

    countries = truncate_names(countries, 10)

    import matplotlib.patches as mpatches

    handles = []
    for key in colors.keys():
        handles.append(mpatches.Patch(color=colors[key], label=key))
    plt.legend(handles=handles)
    plt.xticks(range(len(cor_val)), countries, rotation=90)
    plt.ylabel('Wikipedia instability [1st metric]')
    plt.title('Top '+str(len(countries)) + title)

    plt.show()

def get_country_values_perQuadClass(aggregated_gdelt, cntr_code, start, stop):
    values_to_plot = aggregated_gdelt[ (aggregated_gdelt['ActionGeo_CountryCode'] == cntr_code) &
                                     ((aggregated_gdelt['SQLDATE']).astype(pd.Timestamp) >= pd.Timestamp(start)) &
                                     ((aggregated_gdelt['SQLDATE']).astype(pd.Timestamp) <= pd.Timestamp(stop))]
    values_to_plot = values_to_plot[['SQLDATE','ActionGeo_CountryCode', 'QuadClass', 'Counter']]
    values_to_plot.groupby(['SQLDATE','ActionGeo_CountryCode', 'QuadClass']).sum()

    val1 = values_to_plot[values_to_plot['QuadClass'] == 'Verbal Cooperation']
    val2 = values_to_plot[values_to_plot['QuadClass'] == 'Material Cooperation']
    val3 = values_to_plot[values_to_plot['QuadClass'] == 'Verbal Conflict']
    val4 = values_to_plot[values_to_plot['QuadClass'] == 'Material Conflict']

    x1 = val1['SQLDATE'].values
    y1 = (val1['Counter']/val1['Counter'].mean()).values

    x2 = val2['SQLDATE'].values
    y2 = (val2['Counter']/val2['Counter'].mean()).values

    x3 = val3['SQLDATE'].values
    y3 = (val3['Counter']/val3['Counter'].mean()).values

    x4 = val4['SQLDATE'].values
    y4 = (val4['Counter']/val4['Counter'].mean()).values

    sum_all = np.sum(y1) + np.sum(y2) + np.sum(y3) + np.sum(y4)
    try:
        s1 = np.sum(y1)/sum_all
        s2 = np.sum(y2)/sum_all
        s3 = np.sum(y3)/sum_all
        s4 = np.sum(y3)/sum_all
    except:
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
    return [x1, x2, x3, x4], [y1, y2, y3, y4], [s1,s2,s3,s4]

def analyse_wiki_events_correlation_QuadClass(aggregated_gdelt,country_code, country_name, date_start, date_stop, plot=False):
    history_fetcher = HistoryFetcher(country_name)
    response = history_fetcher.get_history(date_start, date_stop)

    # Keeps only the date field for each edit
    edits_dates = list(map(lambda revision: revision['timestamp'], response))
    plt.figure(figsize=(7,4))

    bins_nr = int( (pd.Timestamp(date_stop) - pd.Timestamp(date_start) ).days/30.5  ) # 1mo per bar

    # Add historgram for the number of edits on country's wikipedia page (normalized)
    n_wiki, bins_wiki, patches_wiki = plt.hist(edits_dates,\
                                               bins=bins_nr,\
                                               normed=True,\
                                               color='blue',\
                                               alpha=0.5,\
                                               label='# of wiki edits '+country_name)
    plt.hold(True)

    x, y, s = get_country_values_perQuadClass(aggregated_gdelt,country_code , date_start, date_stop)
    corrs = list()
    surreneses =list()
    for i in range(len(x)):


        # Add histogram for the number of events (normalized)s
        n_event, bins_event, patches_event = plt.hist(x[i],\
                                                      weights=y[i],
                                                      bins=bins_nr,\
                                                      normed=True,\
                                                      alpha=0.2,\
                                                      label='# events type ' + str(i))
        corr, non_sureness = pearsonr(n_event, n_wiki)

        corrs.append(corr)
        surreneses.append(1-non_sureness)

    if(plot == True):
        plt.xticks(rotation=0)
        plt.xlabel('Time in months \n correlations per class' + str(np.array(corrs)*np.array(surreneses)))
        plt.ylabel('Normalized number of edits/events')
        plt.legend(loc='upper left')
        plt.title('Nr of Wiki Edits and GDELT Events for ' + country_name)
        plt.show()

    return np.array(corrs)*np.array(surreneses), s
