import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.request import urlopen
import json
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
import geopandas as gpd






accidents_df = pd.read_csv('Datasets/accidents_2017.csv')
air_quality_df = pd.read_csv('Datasets/air_quality_Nov2017.csv')
air_ports_df = pd.read_csv('Datasets/air_stations_Nov2017.csv')
births_df = pd.read_csv('Datasets/births.csv')
bus_stops_df = pd.read_csv('Datasets/bus_stops.csv')
death_df = pd.read_csv('Datasets/deaths.csv')
immigrants_nationality_df = pd.read_csv('Datasets/immigrants_by_nationality.csv')
migrants_age_df = pd.read_csv('Datasets/immigrants_emigrants_by_age.csv')
migrants_destination_df = pd.concat([pd.read_csv('Datasets/immigrants_emigrants_by_destination.csv'),
                                     pd.read_csv('Datasets/immigrants_emigrants_by_destination2.csv')])
migrants_sex_df = pd.read_csv('Datasets/immigrants_emigrants_by_sex.csv')
life_expectancy_df = pd.read_csv('Datasets/life_expectancy.csv')
frequent_babies_df = pd.read_csv('Datasets/most_frequent_baby_names.csv')
frequent_people_df = pd.read_csv('Datasets/most_frequent_names.csv')
population_df = pd.read_csv('Datasets/population.csv')
transport_df = pd.read_csv('Datasets/transports.csv')
unemployment_df = pd.read_csv('Datasets/unemployment.csv')

with open('Datasets/barcelonaMap.geojson', 'r') as file:
    gjson_neigh = json.load(file)
# response = urlopen('https://raw.githubusercontent.com/martgnz/bcn-geodata/master/districtes/districtes.geojson')
# gjson_distr = json.loads(response.read())
with open('Datasets/barcelonaMap_Districts.geojson', 'r') as file:
    gjson_distr = json.load(file)

barcelona_districts_geo = gpd.read_file('Datasets/barcelonaMap_Districts.geojson')

# Merge the GeoDataFrame with the population data to get population for each district
barcelona_districts_geo = barcelona_districts_geo.merge(population_df.groupby(['Year', 'District.Name']).agg({'Number': np.sum}).reset_index(), left_on='NOM', right_on='District.Name', how='left')





# Convert the geometries to a projected CRS (here we use EPSG:3395) before calculating the area
barcelona_districts_geo = barcelona_districts_geo.to_crs(epsg=3395)

# Recalculate the area in square kilometers and population density
barcelona_districts_geo['Area_km2'] = barcelona_districts_geo.geometry.area / (10**6)

barcelona_districts_geo['Population Density'] = barcelona_districts_geo['Number'] / barcelona_districts_geo['Area_km2']

# Display the data
barcelona_districts_geo[['NOM', 'Number', 'Area_km2', 'Population Density']]






metro_stations = transport_df[transport_df['Transport'] == 'Underground'].reset_index()
metro_stations = metro_stations[['Longitude', 'Latitude']]
clustering = DBSCAN(eps=0.002, min_samples=1)
clustering = clustering.fit_predict(metro_stations.to_numpy())
clusters_ = dict()
for i, raw in metro_stations.iterrows():
    if clustering[i] in clusters_:
        clusters_[clustering[i]].append([raw['Longitude'], raw['Latitude']])
    else:
        clusters_[clustering[i]] = list([[raw['Longitude'], raw['Latitude']]])
m = folium.Map(location=[41.38879, 2.15899], zoom_start=12)


folium.Choropleth(
    geo_data=gjson_distr,
    data=barcelona_districts_geo,
    columns=['NOM', 'Population Density'],
    key_on='feature.properties.NOM',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Population Density (people per sq km)',
).add_to(m)

for cluster in clusters_.values():
    center = np.mean(np.array(cluster).reshape((-1, 2)), axis=0)
    folium.CircleMarker(tuple((center[1], center[0])), radius=1, color="black", fill=True, fill_opacity=1, opacity=1, fill_color="black",
                        ).add_to(m)
# for idx, row in metro_stations.iterrows():
#     folium.Marker([row['Latitude'], row['Longitude']], popup=row['Station']).add_to(m)

m.save('Plots/Density_metro.html')




with open('Datasets/barcelonaMap.geojson', 'r') as json_file:
    gjson_neigh = json.load(json_file)
# gjson_neigh["features"][7]['properties']['NOM'] = 'el Poble Sec' #rename this district to avoid errors with geojson

transport_df['Neighborhood.Name'] = transport_df['Neighborhood.Name'].replace(['el Poble-sec'], 'el Poble Sec')
temp = transport_df.groupby(['Transport', 'Neighborhood.Name']).nunique().reset_index()

px.choropleth_mapbox(temp,
                     geojson=gjson_neigh, color='Station', locations='Neighborhood.Name',
                     featureidkey="properties.NOM",
                     color_continuous_scale="Emrld",
                     center={"lat": 41.395, "lon": 2.18},
                     animation_frame='Transport',
                     mapbox_style="carto-positron", zoom=10.3, opacity=0.9,
                     height=620)




df = population_df.groupby(['Year', 'Age', 'District.Name']).agg({'Number': sum}).reset_index()
color_sequence = sns.color_palette(palette='viridis', n_colors=len(df['District.Name'].unique()))

fig = px.histogram(df, x='Age', y='Number', animation_frame='Year', hover_name='District.Name', color='District.Name',
                   color_discrete_sequence=color_sequence[0])
fig.update_layout(
    xaxis_title='Ages',
    yaxis_title='Population',
    barmode='group'
)
# fig.show()
pyo.plot(fig, filename='Plots/AgeDistribution_Districts.html', auto_open=False)




df = migrants_age_df.groupby(['Year', 'District Name']).agg({'Immigrants': np.sum, 'Emigrants': np.sum}).reset_index()
color_sequence = sns.color_palette(palette='viridis', n_colors=len(df['District Name'].unique()))

melted_df = pd.melt(df, id_vars=['District Name', 'Year'], value_vars=['Immigrants', 'Emigrants'])
fig = px.histogram(melted_df, x='District Name', y='value', animation_frame='Year', hover_name='District Name',
                   color='variable', color_discrete_sequence=color_sequence[0])
fig.update_layout(
    xaxis_title='Districts',
    yaxis_title='Emigrants',
    barmode='group'
)
fig.show()
pyo.plot(fig, filename='Plots/MigrantsDistribution_Districts.html', auto_open=False)






df = migrants_age_df.groupby(['Year', 'Age', 'District Name']).agg(
    {'Immigrants': np.sum, 'Emigrants': np.sum}).reset_index()
df1_9 = df[(df['Age'] == '0-4') | (df['Age'] == '5-9')]
df = pd.concat([df1_9, df[(df['Age'] != '0-4') & (df['Age'] != '5-9')]])
color_sequence = sns.color_palette(palette='viridis', n_colors=len(df['District Name'].unique()))

# melted_df = pd.melt(df, id_vars=['District Name', 'Year'], value_vars=['Immigrants', 'Emigrants'])
fig = px.histogram(df, x='Age', y='Emigrants', animation_frame='Year', hover_name='District Name',
                   color='District Name', color_discrete_sequence=color_sequence[0])
fig.update_layout(
    xaxis_title='Districts',
    yaxis_title='Emigrants',
    barmode='group'
)
fig.show()
pyo.plot(fig, filename='Plots/EmigrantsDistribution_Districts.html', auto_open=False)





df = migrants_age_df.groupby(['Year', 'Age', 'District Name']).agg(
    {'Immigrants': np.sum, 'Emigrants': np.sum}).reset_index()
df1_9 = df[(df['Age'] == '0-4') | (df['Age'] == '5-9')]
df = pd.concat([df1_9, df[(df['Age'] != '0-4') & (df['Age'] != '5-9')]])
color_sequence = sns.color_palette(palette='viridis', n_colors=len(df['District Name'].unique()))

# melted_df = pd.melt(df, id_vars=['District Name', 'Year'], value_vars=['Immigrants', 'Emigrants'])
fig = px.histogram(df, x='Age', y='Immigrants', animation_frame='Year', hover_name='District Name',
                   color='District Name', color_discrete_sequence=color_sequence[0])
fig.update_layout(
    xaxis_title='Districts',
    yaxis_title='Immigrants',
    barmode='group'
)
fig.show()
pyo.plot(fig, filename='Plots/ImmigrantsDistribution_Districts.html', auto_open=False)





df = population_df.groupby(['Year', 'Age', 'District.Name']).agg({'Number': sum}).reset_index()
df1_9 = df[(df['Age'] == '0-4') | (df['Age'] == '5-9')]
df = pd.concat([df1_9, df[(df['Age'] != '0-4') & (df['Age'] != '5-9')]])
df['Change'] = 0
color_sequence = sns.color_palette(palette='viridis', n_colors=len(df['District.Name'].unique()))


def calcChange(row):
    if row['Year'] <= 2013:
        return row
    row['Change'] = row['Number'] - df[
        (df['Year'] == row['Year'] - 1) & (df['District.Name'] == row['District.Name']) & (df['Age'] == row['Age'])][
        'Number'].item()
    return row


df = df.apply(calcChange, axis=1)
fig = px.histogram(df[df['Year'] > 2013], x='Age', y='Change', animation_frame='Year', hover_name='District.Name',
                   color='District.Name', color_discrete_sequence=color_sequence[0])
fig.update_layout(
    xaxis_title='Ages',
    yaxis_title='Change (Number of People)',
    barmode='group'
)
fig.show()
pyo.plot(fig, filename='Plots/AgeChange_Districts.html', auto_open=False)






df = bus_stops_df[population_df['Year'] == 2017].groupby(['District.Name']).agg({'Bus.Stop': 'count'}).reset_index()
def calc(row):
    row['percent'] = 1 / (row['Bus.Stop'] / population_df[
        (population_df['Year'] == 2017) & (population_df['District.Name'] == row['District.Name'])]['Number'].sum())
    return row
df = df.apply(calc, axis=1)
df









barcelona_coords = [41.3851, 2.1734]
transport_colors = {
    "Underground": "red",
    "Tram": "green"
}

# Create a folium map centered at Barcelona
barcelona_map = folium.Map(location=barcelona_coords, zoom_start=12)

df = population_df.groupby(['Year', 'District.Name']).agg({'Number': sum}).reset_index()
df = df[df['Year'] == 2017]
folium.Choropleth(
    geo_data=gjson_distr,
    name='Choropleth',
    data=df,
    color = 'Number',
    locations = 'District.Name',
    # featureidkey="properties.NOM",
    columns=['District.Name', 'Number'],
    key_on='feature.properties.NOM',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Population by Neighborhood in Barcelona (2017)'
).add_to(barcelona_map)
# Add transport stops to the map with different colors based on the type of transport
# for index, row in bus_stops_df[
#     (bus_stops_df['Transport'] == 'Day bus stop') | (bus_stops_df['Transport'] == 'Bus station')].iterrows():
#     folium.CircleMarker(
#         location=[row['Latitude'], row['Longitude']],
#         radius=3,
#         color="blue",
#         fill=True,
#         fill_color="blue",
#     ).add_to(barcelona_map)

for index, row in transport_df.iterrows():
    if row['Transport'] in transport_colors:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=transport_colors[row['Transport']],
            fill=True,
            fill_color=transport_colors[row['Transport']],
            tooltip=row['Station']
        ).add_to(barcelona_map)

# Display the map
barcelona_map.save('Plots/Transport.html')





# stations = pd.concat([transport_df[
#                           (transport_df['Transport'] == 'Underground') | (transport_df['Transport'] == 'Tram')][
#                           ['Longitude', 'Latitude']], bus_stops_df[
#                           (bus_stops_df['Transport'] == 'Day bus stop') | (bus_stops_df['Transport'] == 'Bus station')][
#                           ['Longitude', 'Latitude']]])
stations = pd.concat([transport_df[
                          (transport_df['Transport'] == 'Underground')][
                          ['Longitude', 'Latitude']]])





from scipy.spatial import cKDTree
from shapely.geometry import Point, shape

# Create arrays of coordinates for transport stops
transport_points_array = np.array([(x, y) for x, y in zip(stations['Longitude'], stations['Latitude'])])

# Get the bounding box of Barcelona
x_min, y_min, x_max, y_max = 2.1, 41.33, 2.2284, 41.4687  # You might need to adjust these coordinates to fit Barcelona's bounds accurately

# Create a grid of points within the bounding box of Barcelona (1000 x 1000)
x_points = np.linspace(x_min, x_max, 200)
y_points = np.linspace(y_min, y_max, 200)
xy_points = np.array([(x, y) for x in x_points for y in y_points])

# Create a KDTree for the transport stops
transport_kdtree = cKDTree(transport_points_array)

# Find the distance to the nearest transport stop for each point in the grid
distances, _ = transport_kdtree.query(xy_points)

# Check if each point is within Barcelona boundaries
within_barcelona = [any(shape(feature['geometry']).contains(Point(xy)) for feature in gjson_neigh['features']) for xy in
                    xy_points]

# Assign a distance of -1 to points that are not within Barcelona
distances[~np.array(within_barcelona)] = -1






barcelona_map = folium.Map(location=barcelona_coords, zoom_start=12)
# df = population_df.groupby(['Year', 'Neighborhood.Name']).agg({'Number': sum}).reset_index()
# df = df[df['Year'] == 2017]
# folium.Choropleth(
#     geo_data=gjson_neigh,
#     name='Choropleth',
#     data=df,
#     columns=['Neighborhood.Name', 'Number'],
#     key_on='feature.properties.NOM',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Population by Neighborhood in Barcelona (2017)'
# ).add_to(barcelona_map)

df = pd.DataFrame(np.hstack((np.flip(xy_points, axis=1), distances.reshape((-1, 1))))[distances >= 0])
# Add a heatmap layer to visualize the most remote points
HeatMap(data=df, radius=8, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 1: 'red'}).add_to(barcelona_map)

# Save the map to an HTML file
barcelona_map.save('Plots/RemotePoints.html')









barcelona_coords = [41.3851, 2.1734]
transport_colors = {
    "Underground": "red",
    "Tram": "green"
}

# Create a folium map centered at Barcelona
barcelona_map = folium.Map(location=barcelona_coords, zoom_start=12)

df = population_df.groupby(['Year', 'Neighborhood.Name']).agg({'Number': sum}).reset_index()
df = df[df['Year'] == 2017]
folium.Choropleth(
    geo_data=gjson_neigh,
    name='Choropleth',
    data=df,
    columns=['Neighborhood.Name', 'Number'],
    key_on='feature.properties.NOM',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Population by Neighborhood in Barcelona (2017)'
).add_to(barcelona_map)
# Add transport stops to the map with different colors based on the type of transport
# for index, row in bus_stops_df[
#     (bus_stops_df['Transport'] == 'Day bus stop') | (bus_stops_df['Transport'] == 'Bus station')].iterrows():
#     folium.CircleMarker(
#         location=[row['Latitude'], row['Longitude']],
#         radius=3,
#         color="blue",
#         fill=True,
#         fill_color="blue",
#     ).add_to(barcelona_map)

for index, row in transport_df.iterrows():
    if row['Transport'] in transport_colors:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=transport_colors[row['Transport']],
            fill=True,
            fill_color=transport_colors[row['Transport']],
        ).add_to(barcelona_map)

df = pd.DataFrame(np.hstack((np.flip(xy_points, axis=1), distances.reshape((-1, 1))))[distances >= 0],
                  columns=['x', 'y', 'dist']).sort_values(by='dist', ascending=False)
# for index, row in df.loc[:200].iterrows():
#     folium.Marker(
#         location=[row['x'], row['y']],
#         icon=folium.Icon(icon='map-pin'),
#     ).add_to(barcelona_map)
# Display the map

dbscan = DBSCAN(eps=0.0008, min_samples=3)  # Adjust eps and min_samples as needed
X = df[:13000][['x', 'y']].to_numpy()
clusters = dbscan.fit_predict(X)

# Find the centroid of each cluster
unique_clusters = np.unique(clusters)
cluster_centroids = np.array([X[clusters == i].mean(axis=0) for i in unique_clusters if i != -1])

# Add markers for the centroids of the clusters
for centroid in cluster_centroids:
    folium.Marker(location=centroid[::1], icon=folium.Icon(icon='map-pin'), tooltip=(str(centroid) + "\n" + str(np.mean(transport_kdtree.query(centroid[::-1], k=3)[0]) * 111139) + "m")).add_to(barcelona_map)

barcelona_map.save('Plots/RemotePoints_Metro.html')








distFromMetro = pd.DataFrame(columns=['x', 'y', 'distances', 'avgDist'])
for centroid in cluster_centroids:
    dist = transport_kdtree.query(centroid[::-1], k=3)[0] * 111139
    avgDist = np.mean(dist)
    if avgDist > 1000:
        continue
    else:
        distFromMetro.loc[len(distFromMetro)] = {'x': centroid[0], 'y': centroid[1], 'distances': dist, 'avgDist': avgDist}

distFromMetro.to_csv('Plots/distanceFromMetro.csv')










barcelona_coords = [41.3851, 2.1734]
transport_colors = {
    "Underground": "red",
    "Tram": "green"
}

# Create a folium map centered at Barcelona
barcelona_map = folium.Map(location=barcelona_coords, zoom_start=13)

# df = population_df.groupby(['Year', 'Neighborhood.Name']).agg({'Number': sum}).reset_index()
# df = df[df['Year'] == 2017]
# folium.Choropleth(
#     geo_data=gjson_neigh,
#     name='Choropleth',
#     data=df,
#     columns=['Neighborhood.Name', 'Number'],
#     key_on='feature.properties.NOM',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Population by Neighborhood in Barcelona (2017)'
# ).add_to(barcelona_map)

for index, row in bus_stops_df[
    (bus_stops_df['Transport'] == 'Day bus stop') | (bus_stops_df['Transport'] == 'Bus station')].iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color="blue",
        fill=True,
        fill_color="blue",
    ).add_to(barcelona_map)

for index, row in transport_df.iterrows():
    if row['Transport'] in transport_colors:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=transport_colors[row['Transport']],
            fill=True,
            fill_color=transport_colors[row['Transport']],
        ).add_to(barcelona_map)

df = pd.DataFrame(np.hstack((np.flip(xy_points, axis=1), distances.reshape((-1, 1))))[distances >= 0],
                  columns=['x', 'y', 'dist']).sort_values(by='dist', ascending=False)

HeatMap(df[:13000].to_numpy(), radius=12, blur=15).add_to(barcelona_map)

barcelona_map.save('Plots/RemotePoints_Metro_Heatmap.html')







pop_df = population_df.groupby(['Year', 'District.Name']).agg({'Number': np.sum}).reset_index()
bus_df = bus_stops_df[bus_stops_df['Transport'] == 'Day bus stop'].groupby(['District.Name']).agg({'Bus.Stop': 'count'}).reset_index().sort_values(by='District.Name')
bus_df['percent'] = 1/(bus_df['Bus.Stop'] / population_df.sort_values(by='District.Name')[
        (population_df.sort_values(by='District.Name')['Year'] == 2017)]['Number'].sum())
bus_df['percent'] = 1/(bus_df['Bus.Stop'].to_numpy() / pop_df[
        (pop_df['Year'] == 2017)].sort_values(by='District.Name')['Number'].to_numpy())
metro_df = transport_df[transport_df['Transport'] == 'Underground'].groupby(['District.Name']).agg({'Station': 'nunique'}).reset_index().sort_values(by='District.Name').reset_index()
metro_df['percent'] = 1/(metro_df['Station'].to_numpy() / pop_df[
        (pop_df['Year'] == 2017)].sort_values(by='District.Name')['Number'].to_numpy())
bus_df
# metro_df










MetroPoints = np.array([[41.4020, 2.1589], [41.3853, 2.1546], [41.3992, 2.1743], [41.3926, 2.1464]])
colors = ['red', 'red', 'red', 'red']

barcelona_coords = [41.3851, 2.1734]

transport_colors = {
    "Underground": "red",
    # "Tram": "green"
}

# Create a folium map centered at Barcelona
barcelona_map = folium.Map(location=barcelona_coords, zoom_start=12)

# df = population_df.groupby(['Year', 'Neighborhood.Name']).agg({'Number': sum}).reset_index()
# df = df[df['Year'] == 2017]
# folium.Choropleth(
#     geo_data=gjson_neigh,
#     name='Choropleth',
#     data=df,
#     columns=['Neighborhood.Name', 'Number'],
#     key_on='feature.properties.NOM',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Population by Neighborhood in Barcelona (2017)'
# ).add_to(barcelona_map)


for index, row in transport_df.iterrows():
    if row['Transport'] in transport_colors:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
        ).add_to(barcelona_map)

# Add markers for the centroids of the clusters
for i in range(len(MetroPoints)):
        folium.CircleMarker(
            location=MetroPoints[i][::1],
            radius=6,
            color=colors[i],
            fill=True,
            fill_color=colors[i],
            tooltip=str(np.mean(transport_kdtree.query(MetroPoints[i][::-1], k=3)[0]) * 111139)
        ).add_to(barcelona_map)

barcelona_map.save('Plots/Key_Metro.html')








barcelona_coords = [41.3851, 2.1734]
transport_colors = {
    "Underground": "red",
    "Tram": "green"
}

# Create a folium map centered at Barcelona
barcelona_map = folium.Map(location=barcelona_coords, zoom_start=13)

# df = population_df.groupby(['Year', 'Neighborhood.Name']).agg({'Number': sum}).reset_index()
# df = df[df['Year'] == 2017]
# folium.Choropleth(
#     geo_data=gjson_neigh,
#     name='Choropleth',
#     data=df,
#     columns=['Neighborhood.Name', 'Number'],
#     key_on='feature.properties.NOM',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Population by Neighborhood in Barcelona (2017)'
# ).add_to(barcelona_map)

for index, row in bus_stops_df[
    (bus_stops_df['Transport'] == 'Day bus stop') | (bus_stops_df['Transport'] == 'Bus station')].iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color="blue",
        fill=True,
        fill_color="blue",
    ).add_to(barcelona_map)

for index, row in transport_df.iterrows():
    if row['Transport'] in transport_colors:
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=transport_colors[row['Transport']],
            fill=True,
            fill_color=transport_colors[row['Transport']],
        ).add_to(barcelona_map)

df = pd.DataFrame(np.hstack((np.flip(xy_points, axis=1), distances.reshape((-1, 1))))[distances >= 0],
                  columns=['x', 'y', 'dist']).sort_values(by='dist', ascending=False)

HeatMap(df[:13000].to_numpy(), radius=12, blur=15).add_to(barcelona_map)

barcelona_map.save('Plots/RemotePoints_Metro_Heatmap.html')