import os
import json
import glob
import matplotlib.pyplot as plt
import geopandas as gpd
from configs.utils import *
import folium
from folium.features import GeoJsonPopup, GeoJsonTooltip
from datetime import datetime, time
from argparse import ArgumentParser
from collections import defaultdict
from io import BytesIO
import base64

parser = ArgumentParser()
parser.add_argument('--config_file', type = str,
                    default = "/d/maboum/css-peft/configs/config.yml")
args = parser.parse_args()

config_file = args.config_file
config = load_config_yaml(file_path = config_file)

dataset = config["dataset"]
data_config = dataset["flair1"]

directory_path = data_config["data_path"]
metadata = data_config["metadata"]
data_sequence = data_config["task_name"]

sequence_list = data_sequence
pretrain_list = data_config["domain_sequence"]
# Specify the départements to highlight
highlight_departments = []
highlight_departments_year = []
for element in sequence_list:
    code = element.split('_')[0]
    year = element.split('_')[1]
    code = code[2:]
    highlight_departments.append(code)
    highlight_departments_year.append(year)

m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
color_mapping = {
    '2018': 'indianred',
    '2019': 'darkred',
    '2020': 'tomato',
    '2021': 'orangered'
}

# Load the GeoJSON data
geojson_path = "departements.geojson"
gdf = gpd.read_file(geojson_path)
gdf['code'] = gdf['code'].astype(str)

departments = []
departments_year = []
file_path = data_config["metadata"]
with open(file_path, "r") as file:
    metadata = json.load(file)

# Set to store unique domains
unique_domains = set()
# Iterate through each IMG entry and extract the domain
for img_data in metadata.values():
    domain = img_data['domain']
    unique_domains.add(domain)
unique_domains_list = list(unique_domains)

for element in unique_domains_list:
    code = element.split('_')[0]
    year = element.split('_')[1]
    code = code[2:]
    departments.append(code)
    departments_year.append(year)
# Create a dictionary to map department codes to years
department_year_dict = dict(zip(departments, departments_year))
gdf['year'] = gdf['code'].map(department_year_dict)


# Add GeoJSON data to the map with the custom style
def highlight_style(feature):
    properties = feature['properties']
    code = properties['code']
    year = properties['year']
    fill_color = color_mapping.get(year)

    if code in highlight_departments:
        return {'fillColor': fill_color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
    if code in list(map(str, departments)) and code not in highlight_departments:
        return {'fillColor': 'green', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5}
    elif code not in list(map(str, departments)):
        return {'fillColor': 'silver', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}


folium.GeoJson(gdf,
               style_function=highlight_style,
               highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.7},
               tooltip=GeoJsonTooltip(fields=["nom", "code", "year"],
                                      aliases=["Département: ", "Code: ", "Year: "],
                                      labels=True,
                                      sticky=True),
               popup=GeoJsonPopup(fields=["nom", "code", "year"],
                                  aliases=["Département: ", "Code: ", "Year: "],
                                  parse_html=True)).add_to(m)
m.save("french_departments_map.html")
m