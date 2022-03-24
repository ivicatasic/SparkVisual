import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import numpy as np
import arrow
from pyspark.ml.regression import LinearRegression

app = dash.Dash(__name__, )
app.title = 'Spark Visual Data'

terr2 = pd.read_csv('data/products.csv')
location1 = terr2[['subproducts']]
list_locations = location1.set_index('subproducts').T.to_dict('dict')
region = terr2['products'].unique()

conf = SparkConf().setAppName('Spark visual').setMaster('local[*]')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# upload Economy inflation file
csv_EI = 'data/Economy-inflation.csv'
data = sqlContext.read.format("csv").options(header='true').load(csv_EI)

country = pd.read_csv(csv_EI)
countries = country['GEOLABEL'].unique()

# upload Economy GDP file
csv_EGDP = 'data/Economy-GDP.csv'
dataGDP = sqlContext.read.format("csv").options(header='true').load(csv_EGDP)

# upload Economy Monthly industrial production
csv_EMIN = 'data/Economy-Monthly industrial production.csv'
dataMIN = sqlContext.read.format("csv").options(header='true').load(csv_EMIN)

# upload Economy montly volume
csv_EMV = 'data/Economy-Monthly volume.csv'
dataMV = sqlContext.read.format("csv").options(header='true').load(csv_EMV)

# upload Economy montly production in construction
csv_EMPIC = 'data/Economy-Monthly production in construction.csv'
dataMPIC = sqlContext.read.format("csv").options(header='true').load(csv_EMPIC)
####################################
# Upload Population and Health

#Ucitavanje fajla: Population and health-Monthly excess mortality
csv_PAHMEM = 'data/Population and health-Monthly excess mortality.csv'
dataPAHMEM = sqlContext.read.format("csv").options(header='true').load(csv_PAHMEM)

# Ucitavanje fajla: Population and health-Number of deaths by week
csv_PAHDBW = 'data/Population and health-Number of deaths by week.csv'
dataPAHDBW = sqlContext.read.format("csv").options(header='true').load(csv_PAHDBW)

#Ucitavanje fajla: Population and health-Monthly first-time asylum applicants
csv_PAHMFTA = 'data/Population and health-Monthly first-time asylum applicants.csv'
dataPAHMFTA = sqlContext.read.format("csv").options(header='true').load(csv_PAHMFTA)
#############################################
#Krece grupa: Society and work

#Ucitavanje fajla Monthly unemployment rate
csv_SAWMUR = 'data/Society and work-Monthly unemployment rate.csv'
dataSAWMUR = sqlContext.read.format("csv").options(header='true').load(csv_SAWMUR)

#Ucitavanje fajla: Society and work-Monthly youth unemployment rate
csv_SAWMYUR = 'data/Society and work-Monthly youth unemployment rate.csv'
dataSAWMYUR = sqlContext.read.format("csv").options(header='true').load(csv_SAWMYUR)

#Ucitavanje fajla: Society and work-Quarterly employment
csv_SAWQE = 'data/Society and work-Quarterly employment.csv'
dataSAWQE = sqlContext.read.format("csv").options(header='true').load(csv_SAWQE)

#Ucitavanje fajla: Society and work-Quarterly labour market slack
csv_SAWQLMS = 'data/Society and work-Quarterly labour market slack.csv'
dataSAWQLMS = sqlContext.read.format("csv").options(header='true').load(csv_SAWQLMS)

#Ucitavanje fajla: Society and work-Quarterly job vacancy rate
csv_SAWQJVR = 'data/Society and work-Quarterly job vacancy rate.csv'
dataSAWQJVR = sqlContext.read.format("csv").options(header='true').load(csv_SAWQJVR)

#Ucitavanje fajla:Society and work-Quarterly labour cost
csv_SAWQLC = 'data/Society and work-Quarterly labour cost.csv'
dataSAWQLC = sqlContext.read.format("csv").options(header='true').load(csv_SAWQLC)
######################################
#KRECE OBLAST AGRICULTURE, ENERGY, TRANSPORT & TOURISM

#Ucitavanje fajla Agriculture, energy, transport & tourism-Monthly air passenger transport
csv_AETTMAPT = 'data/Agriculture, energy, transport & tourism-Monthly air passenger transport.csv'
dataAETTMAPT = sqlContext.read.format("csv").options(header='true').load(csv_AETTMAPT)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly commercial air flights
csv_AETTMCAF = 'data/Agriculture, energy, transport & tourism-Monthly commercial air flights.csv'
dataAETTMCAF = sqlContext.read.format("csv").options(header='true').load(csv_AETTMCAF)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation
csv_AETTMATA = 'data/Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation.csv'
dataAETTMATA = sqlContext.read.format("csv").options(header='true').load(csv_AETTMATA)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation
csv_AETTMNSTA = 'data/Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation.csv'
dataAETTMNSTA = sqlContext.read.format("csv").options(header='true').load(csv_AETTMNSTA)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users
csv_AETTMEC = 'data/Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users.csv'
dataAETTMEC = sqlContext.read.format("csv").options(header='true').load(csv_AETTMEC)
############################################
# labels for Economy inflation
labels = data.columns[2:32]
lab= [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in labels]

# labels for Economy GDP
labelsGDP = ["Q1-2017", "Q2-2017", "Q3-2017", "Q4-2017", "Q1-2018", "Q2-2018", "Q3-2018", "Q4-2018", "Q1-2019",
             "Q2-2019", "Q3-2019", "Q4-2019", "Q1-2020", "Q2-2020", "Q3-2020", "Q4-2020", "Q1-2021", "Q2-2021",
             "Q3-2021", "Q4-2021"]
labGDP = np.array(labelsGDP)
# labels for Economy Montly volume
labelsMV = dataMV.columns[1:85]
labMV= [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in labelsMV]

#######################################
#######################################
# LABELS FOR POPULATION AND HEALTH
# labels population and health-montly excess mortality
labelsMEM = dataPAHMEM.columns[1:25]
labPAHMEM= [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in labelsMEM]

# labela za population and health-Number of deaths by week
csv_PAHDBW2 = 'data/Population and health-Number of deaths by week.csv'
dataPAHDBW2 = sqlContext.read.format("csv").options(header='false').load(csv_PAHDBW)
#Labela za population and health-Monthly first-time asylum applicants
csv_PAHMFTA2 = 'data/Population and health-Monthly first-time asylum applicants.csv'
dataPAHMFTA2 = sqlContext.read.format("csv").options(header='false').load(csv_PAHMFTA2)
labelPAHMFTA2=["I-2015","II-2015","I-2016","II-2016","I-2017","II-2017","I-2018","II-2018","I-2019","II-2019","I-2020","II-2020","I-2021","II-2021","2022"]
#Labele za fajl Population and health-Number of deaths by week
def getLabelPAHDBW2():
    value = []
    value1 = []
    data2 = dataPAHDBW2.first()
    for i in range(1, 109):
        value.append(data2[i])
    value1 = np.array(value)
    return value1
#####################
#Labele za fajl Population and health-Monthly first-time
def getLabelPAHMFTA2():
    value = []
    value1 = []
    data2 = dataPAHMFTA2.first()
    for i in range(1, 167):
        value.append(data2[i])
    value1 = np.array(value)
    return value1
##############################################
#Labele za Society and work
#Society and work-Monthly unemployment rate
d=dataSAWMUR.columns[1:38]
labelSAWMUR = [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in d]

#Labele za Society work-Monthly youth unemployment rate
d=dataSAWMYUR.columns[1:38]
labelSAWMYUR = [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in d]
#Labele za Society and work-Quarterly employment
labelSAWQE=["Q1-2017","Q2-2017","Q3-2017","Q4-2017","Q1-2018","Q2-2018","Q3-2018","Q4-2018","Q1-2019","Q2-2019","Q3-2019","Q4-2019",
            "Q1-2020","Q2-2020","Q3-2020","Q4-2020","Q1-2021","Q2-2021","Q3-2021","Q3-2021"]
#Labele za Society and work-Quarterly labour market slack
labelSAWQLMS=["Q2-2019","Q3-2019","Q4-2019","Q1-2020","Q2-2020","Q3-2020","Q4-2020","Q1-2021","Q2-2021","Q3-2021","Q3-2021"]
############################
#Labele za Agriculture,energy, transport & tourism-Monthly air passenger transport
d=dataAETTMAPT.columns[1:30]
labelsAETTMAPT = [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in d]
#LAbele za: Agriculture,energy, transport & tourism-Monthly commercial air flights
d=dataAETTMCAF.columns[1:25]
labelsAETTMCAF= [arrow.get(s, 'YYYY-MM').format('MM-YYYY') for s in d]
################################################
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H3('Spark Visual app(Interactive data visualisation)', style={"margin-bottom": "0px", 'color': 'white'}),
            ]),
        ], className="six column", id="title")
    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),
    html.Div([
        html.Div([
            dcc.Graph(id='map_1',
                      config={'displayModeBar': 'hover'}),
        ], className="create_container 12 columns"),
    ], className="row flex-display"),
    html.Div([
        html.Div([
            html.P('Izaberi kategoriju:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(id='w_countries',
                         multi=False,
                         clearable=True,
                         disabled=False,
                         style={'display': True},
                         value='Economy',
                         placeholder='Select category',
                         options=[{'label': c, 'value': c}
                                  for c in region], className='dcc_compon'),
            html.P('Izaberi podkategoriju:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(id='w_countries1',
                         multi=False,
                         clearable=True,
                         disabled=False,
                         style={'display': True},
                         placeholder='Select subcategory',
                         options=[], className='dcc_compon'),
            html.P('Izaberi zemlju:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(id='w_countries2',
                         multi=True,
                         clearable=True,
                         disabled=False,
                         style={'display': True},
                         value=['European Union', 'Belgium'],
                         placeholder='Select country',
                         options=[{'label': c, 'value': c}
                                  for c in countries], className='dcc_compon'),
        ], className="create_container three columns"),
        html.Div([
            dcc.Graph(id='bar_line_1', figure={}, clickData=None, hoverData=None,
                      # I assigned None for tutorial purposes. By defualt, these are None, unless you specify otherwise.
                      config={
                          'staticPlot': False,  # True, False
                          'scrollZoom': True,  # True, False
                          'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                          'showTips': False,  # True, False
                          'displayModeBar': True,  # True, False, 'hover'
                          'watermark': True,
                          # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                      })
        ]),
        html.Div([
            dcc.Graph(id='pie',
                      config={'displayModeBar': 'hover'}),
        ], className="create_container three columns"),
    ], className="row flex-display"),
], id="mainContainer", style={"display": "flex", "flex-direction": "column"})
@app.callback(
    Output('w_countries1', 'options'),
    Input('w_countries', 'value'))
def get_country_options(w_countries):
    terr3 = terr2[terr2['products'] == w_countries]
    return [{'label': i, 'value': i} for i in terr3['subproducts'].unique()]

@app.callback(
    Output('w_countries1', 'value'),
    Input('w_countries1', 'options'))
def get_country_value(w_countries1):
    return [k['value'] for k in w_countries1][0]

csv_World = 'data/Latitude.csv'
data_World = sqlContext.read.format("csv").options(header='true').load(csv_World)
############################
# Create scattermapbox chart
############################
@app.callback(Output('map_1', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')])
def update_graph(w_countries, w_countries1):
    dataLat = data_World.select("lat").collect()
    dataLng = data_World.select("lng").collect()

    valueLat = []
    valueL = []
    for i in range(35):
        valueL.append(float(dataLat[i][0]))
    valueLat = np.array(valueL)

    valueLng = []
    valueLn = []
    for i in range(35):
        valueLn.append(float(dataLng[i][0]))
    valueLng = np.array(valueLn)

    data_World2 = pd.read_csv('data/Latitude.csv')
    if (w_countries == 'Economy') & (w_countries1 == 'Inflation - annual growth rate'):
        ei = pd.read_csv('data/Economy-inflation.csv')
        ee=(ei.sum(axis=1)/30).map('{:,.2f}'.format)
        size=abs(ei.sum(axis=1))+20
        return {
        'data': [go.Scattermapbox(
            lon=valueLng,
            lat=valueLat,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=size,
                #color=['blue','yellow','red','green'],
                colorscale='hsv',
                showscale=False,
                sizemode='area'),
            hoverinfo='text',
            hovertext=
            '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>'+
            '<b>Average Economy inflation</b>: ' + ee.astype(str) +'%'+ '<br>')],
        'layout': go.Layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            hovermode='closest',
            mapbox=dict(
                accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                bearing=0,
                pitch=0,
                # Use mapbox token here
                center=go.layout.mapbox.Center(lat=50, lon=15),
                # style='open-street-map',
                style='outdoors',
                zoom=3),
            showlegend=True,
            autosize=True,)
        }
    ##KRECE ECONOMY GDP DEO
    elif (w_countries == 'Economy') & (w_countries1 == 'GDP – quarterly growth rate'):
        ei = pd.read_csv('data/Economy-GDP.csv')
        ee = (ei.sum(axis=1) / 19).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average Economy GDP</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
   ########KRECE DEO ECONOMY MONTHLY INDUSTRIAL PRODUCTION
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly industrial production'):
        ei = pd.read_csv('data/Economy-Monthly industrial production.csv')
        ee = (ei.sum(axis=1) / 29).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) -50
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly industrial production</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    ######KRECE ECONOMY MONTHLY VOLUME OF RETAIL TRADE
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly volume of retail trade'):
        ei = pd.read_csv('data/Economy-Monthly volume.csv')
        ee = (ei.sum(axis=1) / 83).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) -50
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average Monthly volume of retail trade</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    ##########KRECE ECONOMY MONTHLY PRODUCTION IN CONSTRUCTION
    elif (w_countries == 'Economy') and (w_countries1 == 'Monthly production in construction'):
        ei = pd.read_csv('data/Economy-Monthly production in construction.csv')
        ee = (ei.sum(axis=1) / 29).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly production in consturction</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # POPULATION AND HEALTH-Monthly excess mortality
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):
        ei = pd.read_csv('data/Population and health-Monthly excess mortality.csv')
        ee = (ei.sum(axis=1) / 22).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average Monthly excess mortality</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    # Use mapbox token here
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    # style='open-street-map',
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Pocinje Number of deaths by week
    elif (w_countries == 'Population and health') & (w_countries1 == 'Number of deaths by week'):
        ei = pd.read_csv('data/Population and health-Number of deaths by week.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1= ei.replace(',','.',regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 109).map('{:,.2f}'.format)
        size = abs(e1.astype(float).sum(axis=1)-100)
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average number of deaths by week</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    # Use mapbox token here
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    # style='open-street-map',
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    ###Uzimanje podataka iz fajla: Population and health-Monthly first-time asylum
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly first-time asylum applicants'):
        ei = pd.read_csv('data/Population and health-Monthly first-time asylum applicants.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1 = ei.replace(',', '.', regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 150).map('{:,.2f}'.format)
        size = abs(e1.astype(float).sum(axis=1) - 100)
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size-100,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly first-time asylum</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # KRECE SOCIETY AND WORK
    # oblast: Society and work-Monthly unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly unemployment rate'):
        ei = pd.read_csv('data/Society and work-Monthly unemployment rate.csv')
        ee = (ei.sum(axis=1) / 36).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly unemployment rate</b>: ' + ee.astype(str) +'%'+'<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Society and work-Monthly youth unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly youth unemployment rate'):
        ei = pd.read_csv('data/Society and work-Monthly youth unemployment rate.csv')
        ee = (ei.sum(axis=1) / 33).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly youth unemployment rate</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    # Use mapbox token here
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    # style='open-street-map',
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Krece podgrupa: Society and work-Quarterly employment
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly employment'):
        ei = pd.read_csv('data/Society and work-Quarterly employment.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1 = ei.replace(',', '', regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 20).map('{:,.2f}'.format)
        #size = abs(e1.astype(float).sum(axis=1) - 100)
        size1=[22,23,24,25,26,27,28,29,30,31,32,32,33,23,25,19,18,20,21]
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size1,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average quarterly employment</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Society and work-Quarterly labour market slack
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour market slack'):
        ei = pd.read_csv('data/Society and work-Quarterly labour market slack.csv')
        ee = (ei.sum(axis=1) / 10).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average quarterly labour market slack</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Society and work-Quarterly job vacancy rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly job vacancy rate'):
        ei = pd.read_csv('data/Society and work-Quarterly job vacancy rate.csv')
        ee = (ei.sum(axis=1) / 19).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average quarterly job vacancy rate</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # Society and work-Quarterly labour cost
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour cost'):
        ei = pd.read_csv('data/Society and work-Quarterly labour cost.csv')
        ee = (ei.sum(axis=1) / 19).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average quarterly labour cost</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly air passenger transport
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly air passenger transport'):
        ei = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly air passenger transport.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1 = ei.replace(',', '', regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 30).map('{:,.2f}'.format)
        #size = abs(e1.astype(float).sum(axis=1) - 100)
        size1 = [45, 30, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 23, 25, 19, 18, 20, 21,29,35,21,36,15,46,35,29,28,41,36]
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size1,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly air passenger transport</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly commercial air flights
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly commercial air flights'):
        ei = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly commercial air flights.csv')
        ee = (ei.sum(axis=1) / 24).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average Monthly commercial air flights</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly arrivals at tourist accommodation'):
        ei = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1 = ei.replace(',', '', regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 20).map('{:,.2f}'.format)
        #size = abs(e1.astype(float).sum(axis=1) - 100)
        size1 = [45, 30, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 23, 25, 19, 18, 20, 21,29,35,21,36,15,46,35,29,28,41,36]
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size1,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average Monthly arrivals at tourist accommodation</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly nights spent at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly nights spent at tourist accommodation'):
        ei = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation.csv')
        ei = ei.drop('GEOLABEL', 1)
        e1 = ei.replace(',', '', regex=True)
        e3 = (e1.astype(float).sum(axis=1) / 20).map('{:,.2f}'.format)
        #size = abs(e1.astype(float).sum(axis=1) - 100)
        size1 = [45, 30, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33, 23, 25, 19, 18, 20, 21,29,35,21,36,15,46,35,29,28,41,36]
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size1,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly nights spent at tourist accommodation</b>: ' + e3.astype(str) + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    # AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly electricity consumed by end-users
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly electricity consumed by end-users'):
        ei = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users.csv')
        ee = (ei.sum(axis=1) / 30).map('{:,.2f}'.format)
        size = abs(ei.sum(axis=1)) + 20
        return {
            'data': [go.Scattermapbox(
                lon=valueLng,
                lat=valueLat,
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=size,
                    # color=['blue','yellow','red','green'],
                    colorscale='hsv',
                    showscale=False,
                    sizemode='area'),
                hoverinfo='text',
                hovertext=
                '<b>Country</b>: ' + data_World2['country'].astype(str) + '<br>' +
                '<b>Average monthly electricity consumed by end-users</b>: ' + ee.astype(str) + '%' + '<br>')],
            'layout': go.Layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                hovermode='closest',
                mapbox=dict(
                    accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                    bearing=0,
                    pitch=0,
                    center=go.layout.mapbox.Center(lat=50, lon=15),
                    style='outdoors',
                    zoom=3),
                showlegend=True,
                autosize=True, )
        }
    else:
        return dash.no_update
############# LINE CHART ###########
####################################
# Create line  chart
@app.callback(Output('bar_line_1', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')],
              [Input('w_countries2', 'value')])
def update_graph(w_countries, w_countries1, country_chosen):
    coun = []
    coun = np.array(country_chosen)
    #
    # ECONOMY INFLATION
    if (w_countries == 'Economy') & (w_countries1 == 'Inflation - annual growth rate'):
        ei = pd.read_csv('data/Economy-inflation.csv')
        df1 = {}
        for i in range(0, 38):
            niz = ei.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=data.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[2][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[35][2:35] if 'Serbia' in coun else [],
                                mode='lines+markers',
                                name='RS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[35].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[19][2:35] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[1][2:35] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Economy inflation</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[3][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[4][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[5][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[6][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[7][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[8][2:35] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[9][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[10][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[11][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[12][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[13][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[14][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[15][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[16][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[17][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[18][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Economy inflation</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[20][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[21][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Economy inflation</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[22][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[23][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[24][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[25][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[26][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[27][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[28][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[29][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[31][2:35] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[32][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[32].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[33][2:35] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[34][2:35] if 'North Macedonia' in coun else [],
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[34].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[36][2:35] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[36].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=data.collect()[37][2:35] if 'United States' in coun else [],
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United States' + '<br>'+
                                '<b>Economy inflation</b>: ' + df1[37].astype(str) + '<br>'), ],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Inflation - annual growth rate' + '  ' + '<br>' +
                            "(change compared with same month of previous year)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>%</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white' )),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    #Kraj ECONOMY INFLATION
    #############################
    #Pocinje ECONOMY GDP
    elif (w_countries == 'Economy') & (w_countries1 == 'GDP – quarterly growth rate'):
        gdp = pd.read_csv('data/Economy-GDP.csv')
        df1 = {}
        for i in range(0, 35):
            niz = gdp.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labGDP,
                                y=dataGDP.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>GDP</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[2][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>GDP</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[33][2:35] if 'Serbia' in coun else [],
                                mode='lines+markers',
                                name='RS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>' +
                                '<b>GDP</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[19][2:35] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>' +
                                '<b>GDP</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[1][2:35] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>GDP</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[3][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>GDP</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[4][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>GDP</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[5][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>GDP</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[6][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>GDP</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[7][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>GDP</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[8][2:35] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>GDP</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[9][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>GDP</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[10][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>GDP</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[11][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>GDP</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[12][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>GDP</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[13][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>GDP</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[14][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>GDP</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[15][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>GDP</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[16][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>GDP</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[17][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>GDP</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[18][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>GDP</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[20][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>GDP</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[21][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>GDP</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[22][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>GDP</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[23][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>GDP</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[24][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>GDP</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[25][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>GDP</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[26][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>GDP</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[27][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>GDP</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[28][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>GDP</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[29][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>GDP</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[30][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>GDP</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[31][2:35] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>GDP</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=dataGDP.collect()[32][2:35] if 'North Macedonia' in coun else [],
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>' +
                                '<b>GDP</b>: ' + df1[32].astype(str) + '<br>'),
                     go.Scatter(x=labGDP,
                                y=data.collect()[34][2:35] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>' +
                                '<b>GDP</b>: ' + df1[34].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'GDP – quarterly growth rate' + '  ' + '<br>' +
                            "(% change compared with previous quarter)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Quartal</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>%</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # ECONOMY GDP
    #
    # ECONOMY MIN
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly industrial production'):
        eMip = pd.read_csv('data/Economy-Monthly industrial production.csv')
        df1 = {}
        for i in range(0, 36):
            niz = eMip.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=dataMIN.collect()[0][2:38] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[2][2:38] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[34][2:38] if 'Serbia' in coun else [],
                                mode='lines+markers',
                                name='RS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[34].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[19][2:38] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[1][2:38] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Monthly industrial production</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[3][2:38] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[4][2:38] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[5][2:38] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[6][2:38] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[7][2:38] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[8][2:38] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[9][2:38] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[10][2:38] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[11][2:38] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[12][2:38] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[13][2:38] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[14][2:38] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[15][2:38] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[16][2:38] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[17][2:38] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[18][2:38] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly industrial production</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[20][2:38] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[21][2:38] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly industrial production</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[22][2:38] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[23][2:38] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[24][2:38] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[25][2:38] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[226][2:38] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[27][2:38] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[28][2:38] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[29][2:38] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[30][2:38] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[31][2:38] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[33][2:38] if 'North Macedonia' in coun else [],
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMIN.collect()[35][2:38] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>'+
                                '<b>Monthly industrial production</b>: ' + df1[35].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly industrial production' + '  ' + '<br>' +
                            "(Index 2015=100)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Quartal</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>%</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white') ),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # ECONOMY MIN
    #
    # ECONOMY MV
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly volume of retail trade'):
        mv = pd.read_csv('data/Economy-Monthly volume.csv')
        df1 = {}
        for i in range(0, 38):
            niz = mv.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labMV,
                                y=dataMV.collect()[0][2:85] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[2][2:85] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[35][2:85] if 'Serbia' in coun else [],
                                mode='lines+markers',
                                name='RS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[35].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[19][2:85] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[1][2:85] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Monthly volume of retail trade</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[3][2:85] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[4][2:85] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[5][2:85] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[6][2:85] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[7][2:85] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[8][2:85] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[9][2:85] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[10][2:85] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[11][2:85] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[12][2:85] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[13][2:85] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[14][2:85] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[15][2:85] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[16][2:85] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[17][2:85] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[18][2:85] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly volume of retail trade</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[20][2:85] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[21][2:85] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly volume of retail trade</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[22][2:85] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[23][2:85] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=data.collect()[24][2:85] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[25][2:85] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[26][2:85] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[27][2:85] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[28][2:85] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[29][2:85] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[30][2:85] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[32].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[31][2:85] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[33][2:85] if 'North Macedonia' in coun else [],
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[34].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=dataMV.collect()[36][2:85] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[36].astype(str) + '<br>')],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly volume of retail trade' + '  ' + '<br>' +
                            "(Index 2015=100)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>Index 2015=100</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white') ),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # End ECONOMY MV
    #
    # Strart Economy-Monthly production in construction
    #
    elif (w_countries == 'Economy') and (w_countries1 == 'Monthly production in construction'):
        mpic = pd.read_csv('data/Economy-Monthly production in construction.csv')
        df1 = {}
        for i in range(0, 24):
            niz = mpic.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=dataMPIC.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[2][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[1][2:35] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Monthly production in construction</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[3][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[4][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[5][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[6][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[14][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[8][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[9][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[10][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[11][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[12][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly production in construction</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[13][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[14][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly production in construction</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[15][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[16][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[17][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[18][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[19][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[20][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[21][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[22][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=lab,
                                y=dataMPIC.collect()[23][2:35] if 'North Macedonia' in coun else [],
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>'+
                                '<b>Monthly production in construction</b>: ' + df1[23].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly production in construction' + '  ' + '<br>' +
                            "(Index 2015=100)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>Index 2015=100</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # END of ECONOMY
    #
    # POPULATION AND HEALTH-Monthly excess mortality
    #
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):
        pop = pd.read_csv('data/Population and health-Monthly excess mortality.csv')
        df1 = {}
        for i in range(0, 31):
            niz = pop.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly excess mortality</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[1][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly excess mortality</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[17][2:35] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[2][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[3][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[4][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[5][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[6][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[7][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[8][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[9][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[10][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[11][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[12][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[13][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[14][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[15][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[16][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[18][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[19][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[20][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[21][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[22][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[23][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[24][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[25][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[26][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[27][2:35] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[29][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labPAHMEM,
                                y=dataPAHMEM.collect()[30][2:35] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly excess mortality</b>: ' + df1[30].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly excess mortality' + '  ' + '<br>' +
                            "(% of additional deaths compared with average monthly deaths in 2016-2019)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>% of additional deaths</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white' )),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # Zavrsava se POPULATION AND HEALTH-Monthly excess mortality

    ##Pocinje Number of deaths by week
    elif (w_countries == 'Population and health') & (w_countries1 == 'Number of deaths by week'):
            labelDbw = getLabelPAHDBW2()

            dbw = pd.read_csv('data/Population and health-Number of deaths by week.csv')
            df1 = {}
            for i in range(0, 34):
                niz = dbw.loc[i, :]
                df1[i] = niz
            return {
                'data': [go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[1][1:95]] if 'Bulgaria' in coun else [],
                                    mode='lines+markers',
                                    name='Bulgaria',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#E6D1D1', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                    '<b>Number of deaths by week</b>: ' + df1[1].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[0][1:95]] if 'Belgium' in coun else [],
                                    mode='lines+markers',
                                    name='Belgium',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#FF0000', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                    '<b>Number of deaths by week</b>: ' + df1[1].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[33][1:95]] if 'Serbia' in coun else [],
                                    mode='lines+markers',
                                    name='RS',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#FF0000', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Serbia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[33].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[16][1:95]] if 'Malta' in coun else [],
                                    mode='lines+markers',
                                    name='MT',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#FF00FF', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Malta' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[16].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[2][1:95]] if 'Czechia' in coun else [],
                                    mode='lines+markers',
                                    name='CZ',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#49AF30', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[2].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[3][1:95]] if 'Denmark' in coun else [],
                                    mode='lines+markers',
                                    name='DK',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#2A4623', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[3].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[4][1:95]] if 'Germany' in coun else [],
                                    mode='lines+markers',
                                    name='DE',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#7B7D7B', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Germany' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[4].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[5][1:95]] if 'Estonia' in coun else [],
                                    mode='lines+markers',
                                    name='EE',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#C4C048', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[5].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[6][1:95]] if 'Greece' in coun else [],
                                    mode='lines+markers',
                                    name='EL',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#1A46C0', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Greece' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[6].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[7][1:95]] if 'Spain' in coun else [],
                                    mode='lines+markers',
                                    name='ES',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#2E063A', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Spain' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[7].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[8][1:95]] if 'France' in coun else [],
                                    mode='lines+markers',
                                    name='FR',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#39313C', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'France' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[8].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[9][1:95]] if 'Croatia' in coun else [],
                                    mode='lines+markers',
                                    name='HR',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#189F96', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[9].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[10][1:95]] if 'Italy' in coun else [],
                                    mode='lines+markers',
                                    name='IT',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#94A4A3', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Italy' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[10].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[11][1:95]] if 'Cyprus' in coun else [],
                                    mode='lines+markers',
                                    name='CY',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#ff3399', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[11].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[12][1:95]] if 'Latvia' in coun else [],
                                    mode='lines+markers',
                                    name='LV',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#aaaa55', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[12].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[13][1:95]] if 'Lithuania' in coun else [],
                                    mode='lines+markers',
                                    name='LT',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#ffff00', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[13].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[14][1:95]] if 'Luxembourg' in coun else [],
                                    mode='lines+markers',
                                    name='LU',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#007399', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[14].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[15][1:95]] if 'Hungary' in coun else [],
                                    mode='lines+markers',
                                    name='HU',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#ff3300', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[15].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[17][1:95]] if 'Netherlands' in coun else [],
                                    mode='lines+markers',
                                    name='NL',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#00e600', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[17].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[18][1:95]] if 'Austria' in coun else [],
                                    mode='lines+markers',
                                    name='AT',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#4dff4d', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Austria' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[18].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[19][1:95]] if 'Poland' in coun else [],
                                    mode='lines+markers',
                                    name='PL',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#003300', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Poland' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[19].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[20][1:95]] if 'Portugal' in coun else [],
                                    mode='lines+markers',
                                    name='PT',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#662900', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[20].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[21][1:95]] if 'Romania' in coun else [],
                                    mode='lines+markers',
                                    name='RO',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#993399', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Romania' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[21].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[22][1:95]] if 'Slovenia' in coun else [],
                                    mode='lines+markers',
                                    name='SI',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#d98cd9', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[22].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[23][1:95]] if 'Slovakia' in coun else [],
                                    mode='lines+markers',
                                    name='SK',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#0033cc', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[23].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[24][1:95]] if 'Finland' in coun else [],
                                    mode='lines+markers',
                                    name='FI',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#99b3ff', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Finland' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[24].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[25][1:95]] if 'Sweden' in coun else [],
                                    mode='lines+markers',
                                    name='SE',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#001a66', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[25].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[30][1:95]] if 'United Kingdom' in coun else [],
                                    mode='lines+markers',
                                    name='UK',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#00cc99', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[30].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[26][1:95]] if 'Iceland' in coun else [],
                                    mode='lines+markers',
                                    name='IS',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#336600', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[26].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[28][1:95]] if 'Norway' in coun else [],
                                    mode='lines+markers',
                                    name='NO',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#73e600', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Norway' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[28].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=[float(s.replace(',','')) for s in dataPAHDBW.collect()[29][1:95]] if 'Switzerland' in coun else [],
                                    mode='lines+markers',
                                    name='CH',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#bfff80', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                    '<b>Number of deaths by week</b>: ' + df1[29].astype(str) + '<br>'),],
                'layout': go.Layout(
                    barmode='stack',
                    plot_bgcolor='#808080',
                    paper_bgcolor='#A8A8A8',
                    title={
                        'text': 'Number of deaths by week' + '  ' + '<br>' +
                                "(absolute numbers)" + '</br>',
                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont={
                        'color': 'white',
                        'size': 20},
                    hovermode='closest',
                    showlegend=True,
                    xaxis=dict(title='<b>Year</b>',
                               spikemode='toaxis+across',
                               spikedash='solid',
                               tick0=0,
                               dtick=1,
                               color='white',
                               showline=True,
                               showgrid=True,
                               showticklabels=True,
                               linecolor='white',
                               linewidth=2,
                               ticks='outside',
                               tickfont=dict(
                                   family='Arial',
                                   size=12,
                                   color='white')),
                    yaxis=dict(title='<b>% of additional deaths</b>',
                               color='white',
                               showline=True,
                               showgrid=True,
                               showticklabels=True,
                               linecolor='white',
                               linewidth=2,
                               ticks='outside',
                               tickfont=dict(
                                   family='Arial',
                                   size=12,
                                   color='white') ),
                    legend={
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font=dict(
                        family="sans-serif",
                        size=12,
                        color='white'),)
            }
    ##################################Pocinje Polulation and healht-Monthly first-time asylum applicants
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly first-time asylum applicants'):
        mfta = pd.read_csv('data/Population and health-Monthly first-time asylum applicants.csv')
        df1 = {}
        for i in range(0, 31):
            niz = mfta.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[0][1:150]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[1][1:150]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[18][1:150]] if 'Malta' in coun else [],
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[3][1:150]] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[4][1:150]] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[5][1:150]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[6][1:150]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[8][1:150]] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[9][1:150]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[12][1:150]] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[11][1:150]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[12][1:150]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[13][1:150]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[14][1:150]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[15][1:150]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[16][1:150]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[17][1:150]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[19][1:150]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[20][1:150]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[21][1:150]] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[22][1:150]] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[23][1:150]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[24][1:150]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[25][1:150]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[26][1:150]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[27][1:150]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[30][1:150]] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[30][1:150]] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelPAHMFTA2,
                                y=[float(s.replace(',','')) for s in dataPAHMFTA.collect()[29][1:150]] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly first-time asylum applicants</b>: ' + df1[29].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly first-time asylum applicants' + '  ' + '<br>' +
                            "(absolute numbers)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white') ),
                yaxis=dict(title='<b>Number of first time asylum applicants</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'))
        }
    ##################################################
    #KRECE SOCIETY AND WORK
    #Iscrtavanje oblasti: Society and work-Monthly unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly unemployment rate'):
        mur = pd.read_csv('data/Society and work-Monthly unemployment rate.csv')
        df1 = {}
        for i in range(0, 36):
            niz = mur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[1][2:38] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[2][2:38] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly unemployment rate</b>: ' + df1[2].astype(str) + '<br>'
                                ),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[0][2:38] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[3][2:38] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[4][2:38] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[5][2:38] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[6][2:38] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[7][2:38] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[8][2:38] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[9][2:38] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[10][2:38] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[11][2:38] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[12][2:38] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[13][2:38] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[14][2:38] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[15][2:38] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[16][2:38] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[17][2:38] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[18][2:38] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[20][2:38] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[21][2:38] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[22][2:38] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[23][2:38] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[24][2:38] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[25][2:38] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[26][2:38] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[27][2:38] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[28][2:38] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[29][2:38] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[30][2:38] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[31][2:38] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[32][2:38] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[32].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[33][2:38] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMUR,
                                y=dataSAWMUR.collect()[34][2:38] if 'United States' in coun else [],
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United States' + '<br>' +
                                '<b>Monthly unemployment rate</b>: ' + df1[34].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly unemployment rate' + '  ' + '<br>' +
                            "(as % of active population aged 15 to 74 years)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Seasonally adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>% of active population aged 15-74 years</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'), )
        }
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly youth unemployment rate'):
        myur = pd.read_csv('data/Society and work-Monthly youth unemployment rate.csv')
        df1 = {}
        for i in range(0, 36):
            niz = myur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[1][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[2][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly youth unemployment rate</b>: ' + df1[2].astype(str) + '<br>'
                                ),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[0][2:35] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[3][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[4][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[5][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[6][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[7][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[8][2:35] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[9][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[10][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[11][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[12][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[13][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[14][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[15][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[16][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[17][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[18][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[20][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[21][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[22][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[23][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[24][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[25][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[26][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[27][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[28][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[29][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[30][2:35] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[31][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[32][2:35] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[32].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[35][2:35] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[33].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWMYUR,
                                y=dataSAWMYUR.collect()[34][2:35] if 'United States' in coun else [],
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United States' + '<br>' +
                                '<b>Monthly youth unemployment rate</b>: ' + df1[34].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly youth unemployment rate' + '  ' + '<br>' +
                            "(as % of active population aged less than 25 years)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Seasonally adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>% of active population aged > 25 years</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    #KRece podgrupa: Society and work-Quarterly employment
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly employment'):
        qe = pd.read_csv('data/Society and work-Quarterly employment.csv')
        df1 = {}
        for i in range(0, 26):
            niz = qe.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[0][1:20]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[2][1:20]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Quarterly employment</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[0][1:20]] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[3][1:20]] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[4][1:20]] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[5][1:20]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[6][1:20]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[7][1:20]] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[8][1:20]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[9][1:20]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[10][1:20]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[11][1:20]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[12][1:20]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[13][1:20]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[14][1:20]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[15][1:20]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[16][1:20]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[17][1:20]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[18][1:20]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[19][1:20]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[20][1:20]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[21][1:20]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[22][1:20]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[23][1:20]] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[24][1:20]] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=[float(s.replace(',','')) for s in dataSAWQE.collect()[25][1:20]] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Quarterly employment</b>: ' + df1[24].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Quarterly employment' + '  ' + '<br>' +
                            "(1 000 persons)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Calendar and seasonally adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white') ),
                yaxis=dict(title='<b>1 000 persons</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'), )
        }
    #Krece podgrupa Society and work-Quarterly labour market slack
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour market slack'):
        qlms = pd.read_csv('data/Society and work-Quarterly labour market slack.csv')
        df1 = {}
        for i in range(0, 36):
            niz = qlms.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[0][1:20]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Quarterly labour market slack</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[2][1:20]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Quarterly labour market slack</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[1][1:20]] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[3][1:20]] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[4][1:20]] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[5][1:20]] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[6][1:20]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[7][1:20]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[8][1:20]] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[9][1:20]] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[10][1:20]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[11][1:20]] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[12][1:20]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[13][1:20]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[14][1:20]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[15][1:20]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[16][1:20]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[17][1:20]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[18][1:20]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[20][1:20]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[21][1:20]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[22][1:20]] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[23][1:20]] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[24][1:20]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[25][1:20]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[26][1:20]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[27][1:20]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[28][1:20]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[29][1:20]] if 'Icelan' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[30][1:20]] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[31][1:20]] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[35][1:20]] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[35].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQLMS,
                                y=[float(s.replace(',','')) for s in dataSAWQLMS.collect()[34][1:20]] if 'Serbia' in coun else [],
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>' +
                                '<b>Quarterly labour market slack</b>: ' + df1[34].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Quarterly labour market slack' + '  ' + '<br>' +
                            "(as % of extended labour force aged 15 to 74 years)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Seasonally adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>% of extended labour force aged 15-74 years</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white' )),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    #KRece podgrupa Society and work-Quarterly job vacancy rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly job vacancy rate'):
        mur = pd.read_csv('data/Society and work-Quarterly job vacancy rate.csv')
        df1 = {}
        for i in range(0, 29):
            niz = mur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[0][2:20] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Quarterly job vacancy rate</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[2][2:20] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Quarterly job vacancy rate</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[1][2:20] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[3][2:20] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[4][2:20] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[5][2:20] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[6][2:20] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[7][2:20] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[8][2:20] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[9][2:20] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[10][2:20] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[11][2:20] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[12][2:20] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[13][2:20] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[14][2:20] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[16][2:20] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[17][2:20] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[18][2:20] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[19][2:20] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[20][2:20] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[21][2:20] if 'SLovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[22][2:20] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[23][2:20] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[24][2:20] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[27][2:20] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[25][2:20] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQJVR.collect()[26][2:20] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[26].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Quarterly job vacancy rate' + '  ' + '<br>' +
                            "%" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Not seasonally adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>%</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
# KRece podgrupa Society and work-Quarterly labour cost
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour cost'):
        qlc = pd.read_csv('data/Society and work-Quarterly labour cost.csv')
        df1 = {}
        for i in range(0, 29):
            niz = qlc.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[0][2:20] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Quarterly labour cost</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[2][2:20] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Quarterly labour cost</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[1][2:20] if 'Euro area' in coun else [],
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[3][2:20] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[4][2:20] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[6][2:20] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[7][2:20] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[8][2:20] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[9][2:20] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[10][2:20] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[14][2:20] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[15][2:20] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[16][2:20] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[17][2:20] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[18][2:20] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[20][2:20] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[21][2:20] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[22][2:20] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[23][2:20] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[24][2:20] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[25][2:20] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[26][2:20] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[27][2:20] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[28][2:20] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[31][2:20] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=dataSAWQLC.collect()[30][2:20] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[25].astype(str) + '<br>')],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Quarterly labour cost' + '  ' + '<br>' +
                            "(% change compared with the same quarter of the previous year)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Calendar adjusted</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>%</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    #KRECE OBLAST AGRICULTURE,ENERGY, TRANSPORT & TOURISM
    ###################################################
    #Krece oblast Agriculture, energy, transport & tourism-Monthly air passenger transport
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly air passenger transport'):
        mapt = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly air passenger transport.csv')
        df1 = {}
        for i in range(0, 31):
            niz = mapt.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[0][1:35]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[1][1:35]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly air passenger transport</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[2][1:35]] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[3][1:35]] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[4][1:35]] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[5][1:35]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[6][1:35]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[7][1:35]] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[8][1:35]] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[9][1:35]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[10][1:35]] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[11][1:35]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[12][1:35]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[13][1:35]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[14][1:35]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[15][1:35]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[16][1:35]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[17][1:35]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[19][1:35]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[20][1:35]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[21][1:35]] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[22][1:35]] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[23][1:35]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[24][1:35]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[25][1:35]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[26][1:35]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[27][1:35]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[30][1:35]] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[28][1:35]] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMAPT,
                                y=[float(s.replace(',','')) for s in dataAETTMAPT.collect()[29][1:35]] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly air passenger transport</b>: ' + df1[29].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly air passenger transport' + '  ' + '<br>' +
                            "(number of passengers)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>number of passengers</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly commercial air flights
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly commercial air flights'):
        mcaf = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly commercial air flights.csv')
        df1 = {}
        for i in range(0, 37):
            niz = mcaf.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[1][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly commercial air flights</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[2][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[3][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[4][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[4].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[5][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[6][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[7][2:35] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[8][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[9][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[10][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[11][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[12][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[13][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[14][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[15][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[16][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[17][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[19][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[18].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[20][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[21][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[22][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[23][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[24][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[25][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[26][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[27][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[28][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[29][2:35] if 'Iceland' in coun else [],
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[29].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[30][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[31][2:35] if 'Switzerland' in coun else [],
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMCAF.collect()[36][2:35] if 'Turkey' in coun else [],
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>' +
                                '<b>Monthly commercial air flights</b>: ' + df1[36].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly commercial air flights' + '  ' + '<br>' +
                            "(% change compared with same period of previous year)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Year</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>% changing</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly arrivals at tourist accommodation'):
        mata = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation.csv')

        df1 = {}
        for i in range(0, 29):
            niz = mata.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[0][1:20]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[1][1:20]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[2][1:20]] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[3][1:20]] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[5][1:20]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[6][1:20]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[7][1:20]] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[8][1:20]] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[9][1:20]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[10][1:20]] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[11][1:20]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[12][1:20]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[13][1:20]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[14][1:20]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[15][1:20]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[16][1:20]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[17][1:20]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[19][1:20]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[20][1:20]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[21][1:20]] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[22][1:20]] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[23][1:20]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[24][1:20]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[25][1:20]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[26][1:20]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMATA.collect()[27][1:20]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly arrivals at tourist accommodation</b>: ' + df1[27].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly arrivals at tourist accommodation establishments' + '  ' + '<br>' +
                            "(absolute numbers)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(title='<b>Refers to arrivals at hotels, holiday and other short-stay accommodation; camping grounds, recreational vehicle parks and trailer parks.</b>',
                           spikemode='toaxis+across',
                           spikedash='solid',
                           tick0=0,
                           dtick=1,
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                yaxis=dict(title='<b>Number of arrrivals</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly nights spent at tourist accommodation'):
        mnsta = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation.csv')
        df1 = {}
        for i in range(0, 32):
            niz = mnsta.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[0][1:25]] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'+
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[1][1:25]] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[2][1:25]] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[3][1:25]] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[4][1:25]] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[5][1:25]] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[6][1:25]] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[7][1:25]] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[8][1:25]] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[9][1:25]] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[10][1:25]] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[11][1:25]] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[12][1:25]] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[13][1:25]] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[14][1:25]] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[15][1:25]] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[16][1:25]] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[17][1:25]] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[19][1:25]] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[20][1:25]] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[21][1:25]] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[22][1:25]] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[23][1:25]] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[24][1:25]] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[25][1:25]] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[26][1:25]] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[27][1:25]] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[30][1:25]] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[30].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=[float(s.replace(',','')) for s in dataAETTMNSTA.collect()[29][1:25]] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly nights spent at tourist accommodation</b>: ' + df1[28].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly nights spent at tourist accommodation establishments' + '  ' + '<br>' +
                            "(absolute numbers)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(
                    title='<b>Refers to nights spent at hotels, holiday and other short-stay accommodation; camping grounds, recreational vehicle parks and trailer parks.</b>',
                    spikemode='toaxis+across',
                    spikedash='solid',
                    tick0=0,
                    dtick=1,
                    color='white',
                    showline=True,
                    showgrid=True,
                    showticklabels=True,
                    linecolor='white',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='white')),
                yaxis=dict(title='<b>Number of nights spent</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly electricity consumed by end-users'):
        mecbu = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users.csv')
        df1 = {}
        for i in range(0, 30):
            niz = mecbu.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[0][2:35] if 'European Union' in coun else [],
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[0].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[1][2:35] if 'Belgium' in coun else [],
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'+
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[1].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[2][2:35] if 'Bulgaria' in coun else [],
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[2].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[3][2:35] if 'Czechia' in coun else [],
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[3].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[4][2:35] if 'Denmark' in coun else [],
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[5][2:35] if 'Germany' in coun else [],
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[6][2:35] if 'Estonia' in coun else [],
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[6].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[7][2:35] if 'Ireland' in coun else [],
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[7].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[8][2:35] if 'Greece' in coun else [],
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[8].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[9][2:35] if 'Spain' in coun else [],
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[9].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[10][2:35] if 'France' in coun else [],
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[10].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[11][2:35] if 'Croatia' in coun else [],
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[11].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[12][2:35] if 'Italy' in coun else [],
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[12].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[13][2:35] if 'Cyprus' in coun else [],
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[13].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[14][2:35] if 'Latvia' in coun else [],
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[14].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[15][2:35] if 'Lithuania' in coun else [],
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[15].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[16][2:35] if 'Luxembourg' in coun else [],
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[16].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[17][2:35] if 'Hungary' in coun else [],
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[17].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[19][2:35] if 'Netherlands' in coun else [],
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[19].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[20][2:35] if 'Austria' in coun else [],
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[20].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[21][2:35] if 'Poland' in coun else [],
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[21].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[22][2:35] if 'Portugal' in coun else [],
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[22].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[23][2:35] if 'Romania' in coun else [],
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[23].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[24][2:35] if 'Slovenia' in coun else [],
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[24].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[25][2:35] if 'Slovakia' in coun else [],
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[26][2:35] if 'Finland' in coun else [],
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[26].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[27][2:35] if 'Sweden' in coun else [],
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[27].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[28][2:35] if 'United Kingdom' in coun else [],
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[28].astype(str) + '<br>'),
                     go.Scatter(x=labelsAETTMCAF,
                                y=dataAETTMEC.collect()[29][2:35] if 'Norway' in coun else [],
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Monthly electricity consumed by end-users</b>: ' + df1[29].astype(str) + '<br>'),],
            'layout': go.Layout(
                barmode='stack',
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                title={
                    'text': 'Monthly electricity consumed by end-users' + '  ' + '<br>' +
                            "(% change compared with same period of previous year)" + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                hovermode='closest',
                showlegend=True,
                xaxis=dict(
                    title='<b></b>',
                    spikemode='toaxis+across',
                    spikedash='solid',
                    tick0=0,
                    dtick=1,
                    color='white',
                    showline=True,
                    showgrid=True,
                    showticklabels=True,
                    linecolor='white',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='white')),
                yaxis=dict(title='<b>% change</b>',
                           color='white',
                           showline=True,
                           showgrid=True,
                           showticklabels=True,
                           linecolor='white',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='white')),
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),)
        }
    else:
        return dash.no_update
#################################################
#KRAJ LINE CHART GRAFIKA
#################################################
# Kreiranje pie chart grafika
#################################################
#Uzimanje podataka za Podgrupu Economy Inflation
def getPieArr(countryName):
    data2 = data.where(data.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(2, 31):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:GDP – quarterly growth rate
def getPieGDP(countryName):
    data2 = dataGDP.where(dataGDP.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(1, 19):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:Economy Monthly industrial production
def getPieMIP(countryName):
    data2 = dataMIN.where(dataMIN.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(1, 29):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:Economy montly volume
def getPieMV(countryName):
    data2 = dataMV.where(dataMV.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(1, 83):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:Economy Monthly production in contruction
def getPieMPIC(countryName):
    data2 = dataMPIC.where(dataMPIC.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(1, 29):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:Population and health-Monthly excess mortality
def getPieMEM(countryName):
    data2 = dataPAHMEM.where(dataPAHMEM.GEOLABEL == countryName).collect()
    suma = 0
    for i in range(1, 22):
        suma = suma + float(data2[0][i])
    return suma
#Uzimanje podataka za podgrupu:Population and health-Number of deaths by week
def getPieNOD(countryName):
    data2 = dataPAHDBW.where(dataPAHDBW.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    for i in range(1, 109):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 95):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 94):
        suma = suma + float(value3[i])
    return suma
#Uzimanje podataka za podgrupu:Population and health-Monthly first-time asylum
def getPieMFTA(countryName):
    data2 = dataPAHMFTA.where(dataPAHMFTA.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    for i in range(1, 146):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 145):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 144):
        suma = suma + float(value3[i])
    return suma
#############################################
#Krece grupa: Society and work
#####################################
#Uzimanje podataka za podgrupu:Society and work-Monthly unemployment rate
def getPieMUR(countryName):
    data2 = dataSAWMUR.where(dataSAWMUR.GEOLABEL == countryName).collect()
    suma = 0
    formsuma = 0
    for i in range(1, 35):
        suma = suma + float(data2[0][i])
    formsuma = suma / 35
    formsuma = f"{formsuma:.2f}"
    return formsuma
#Uzimanje podataka za podgrupu:Society and work-Monthly youth unemployment rate
def getPieMYUR(countryName):
    data2 = dataSAWMYUR.where(dataSAWMYUR.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 30):
        suma = suma + float(data2[0][i])
    formsuma=suma/30
    formsuma=f"{formsuma:.2f}"
    return formsuma
#Uzimanje podataka za podgrupu:Society and work-Quarterly employment
def getPieQE(countryName):
    data2 = dataSAWQE.where(dataSAWQE.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    formsuma = 0
    for i in range(1, 20):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 19):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 18):
        suma = suma + float(value3[i])
    formsuma = suma / 18
    formsuma = f"{formsuma:.2f}"
    return formsuma
#Uzimanje podataka za podgrupu:Society and work-Quarterly labour market slack
def getPieQLMS(countryName):
    data2 = dataSAWQLMS.where(dataSAWQLMS.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 10):
        suma = suma + float(data2[0][i])
    formsuma=suma/10
    formsuma=f"{formsuma:.2f}"
    return formsuma
#Uzimanje podataka za podgrupu:Society and work-Quarterly job vacancy rate
def getPieQJVR(countryName):
    data2 = dataSAWQJVR.where(dataSAWQJVR.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 19):
        suma = suma + float(data2[0][i])
    formsuma=suma/19
    formsuma=f"{formsuma:.2f}"
    return formsuma
#Uzimanje podataka za podgrupu:Society and work-Quarterly labour cost
def getPieQLC(countryName):
    data2 = dataSAWQLC.where(dataSAWQLC.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 19):
        suma = suma + float(data2[0][i])
    formsuma=suma/19
    formsuma=f"{formsuma:.2f}"
    return formsuma
##########################################
#UZIMANJE PODATAKA ZA GRUPU:AGRICULTURE, ENERGY, TRANSPORT & TOURISM
############################
#Uzimanje podataka za: AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly air passenger transport
def getPieMAPT(countryName):
    data2 = dataAETTMAPT.where(dataAETTMAPT.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    formsuma = 0
    for i in range(1, 30):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 29):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 28):
        suma = suma + float(value3[i])
    formsuma = suma / 28
    formsuma = f"{formsuma:.2f}"
    return formsuma
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly commercial air flights
def getPieMCAF(countryName):
    data2 = dataAETTMCAF.where(dataAETTMCAF.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 24):
        suma = suma + float(data2[0][i])
    formsuma=suma/24
    formsuma=f"{formsuma:.2f}"
    return formsuma
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly arrivals at tourist accommodation
def getPieMATA(countryName):
    data2 = dataAETTMATA.where(dataAETTMATA.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    formsuma = 0
    for i in range(1, 20):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 19):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 18):
        suma = suma + float(value3[i])
    formsuma = suma / 18
    formsuma = f"{formsuma:.2f}"
    return formsuma
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly nights spent at tourist accommodation
def getPieMNSTA(countryName):
    data2 = dataAETTMNSTA.where(dataAETTMNSTA.GEOLABEL == countryName).collect()
    value = []
    value1 = []
    value2 = []
    value3 = []
    formsuma = 0
    for i in range(1, 20):
        val = data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1, 19):
        val = float(value[i].replace(',', ''))
        value2.append(val)
    value3 = np.array(value2)
    suma = 0
    for i in range(0, 18):
        suma = suma + float(value3[i])
    formsuma = suma / 18
    formsuma = f"{formsuma:.2f}"
    return formsuma
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly electricity consumed by end-users
def getPieMEC(countryName):
    data2 = dataAETTMEC.where(dataAETTMEC.GEOLABEL == countryName).collect()
    suma = 0
    formsuma=0
    for i in range(1, 24):
        suma = suma + float(data2[0][i])
    formsuma=suma/24
    formsuma=f"{formsuma:.2f}"
    return formsuma

@app.callback(Output('pie', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')],
              [Input('w_countries2', 'value')])
def display_content(w_countries, w_countries1, country_chosen):
    coun = np.array(country_chosen)
    colors = ['#FF00FF', '#9C0C38', 'orange', 'lightblue']
    #Pocinje deo Economy- Inflation
    if (w_countries == 'Economy') & (w_countries1 == 'Inflation - annual growth rate'):
        valueEu1= ''
        if('European Union' in coun):
           valueEu1 = getPieArr('European Union')

        valueBel1= ''
        if('Belgium' in coun):
           valueBel1=getPieArr('Belgium')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieArr('Serbia')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieArr('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieArr('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieArr('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieArr('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieArr('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieArr('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieArr('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieArr('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieArr('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieArr('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieArr('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieArr('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieArr('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieArr('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieArr('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieArr('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieArr('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieArr('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieArr('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieArr('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieArr('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieArr('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieArr('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieArr('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieArr('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieArr('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieArr('United Kingdom')

        valueEea1 = ''
        if ('European Economic Area' in coun):
            valueEea1 = getPieArr('European Economic Area')

        valueIce1 = ''
        if ('Iceland' in coun):
            valueIce1 = getPieArr('Iceland')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieArr('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieArr('Switzerland')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieArr('North Macedonia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieArr('Turkey')

        valueUs1 = ''
        if ('United States' in coun):
            valueUs1 = getPieArr('United States')
        return {
            'data': [go.Pie(labels=['EU','BE','RS','BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','EEA','IS','NO','CH','MK','TR','US'],
                        values=[valueEu1, valueBel1,valueSer1,valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueEea1,valueIce1,valueNor1,valueSwi1,valueMake1,valueTur1,valueUs1],
                        marker=dict(colors=colors),
                        hoverinfo='label+value+percent',
                        textinfo='label+value',
                        textfont=dict(size=13))],
            'layout': go.Layout(
                    plot_bgcolor='#808080',
                    paper_bgcolor='#A8A8A8',
                    hovermode='closest',
                    title={
                        'text':'Average inflation' +'</br>',
                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont={
                        'color': 'white',
                        'size': 20},
                    legend={
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                    font=dict(
                        family="sans-serif",
                        size=12,
                        color='white')),
        }
    # Pocinje ECONOMY GDP
    elif (w_countries == 'Economy') & (w_countries1 == 'GDP – quarterly growth rate'):
        valueEu1 = ''
        if('European Union' in coun):
            valueEu1=getPieGDP('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieGDP('Belgium')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieGDP('Serbia')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieGDP('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieGDP('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieGDP('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieGDP('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieGDP('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieGDP('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieGDP('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieGDP('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieGDP('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieGDP('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieGDP('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieGDP('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieGDP('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieGDP('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieGDP('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieGDP('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieGDP('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieGDP('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieGDP('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieGDP('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieGDP('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieGDP('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieGDP('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieGDP('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieGDP('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieGDP('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieGDP('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieGDP('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieGDP('Switzerland')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieGDP('North Macedonia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieGDP('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR','BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','MK','TR'],
                            values=[valueEu1, valueBel1, valueSer1,valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueMake1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average GDP' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #
    # ECONOMY Monthly industrial production
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly industrial production'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMIP('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMIP('Belgium')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieMIP('Serbia')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMIP('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMIP('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMIP('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMIP('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMIP('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMIP('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMIP('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMIP('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMIP('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMIP('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMIP('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMIP('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMIP('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMIP('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMIP('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMIP('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMIP('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMIP('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMIP('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMIP('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMIP('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMIP('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMIP('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMIP('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMIP('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMIP('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMIP('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMIP('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMIP('Switzerland')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieMIP('North Macedonia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieMIP('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR','BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','MK','TR'],
                            values=[valueEu1, valueBel1, valueSer1,valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueMake1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average industrial production' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    # ECONOMY MV
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly volume of retail trade'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMV('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMV('Belgium')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieMV('Serbia')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMV('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMV('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMV('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMV('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMV('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMV('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMV('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMV('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMV('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMV('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMV('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMV('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMV('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMV('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMV('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMV('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMV('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMV('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMV('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMV('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMV('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMV('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMV('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMV('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMV('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMV('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMV('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMV('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMV('Switzerland')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieMV('North Macedonia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieMV('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR','BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','MK','TR'],
                            values=[valueEu1, valueBel1, valueSer1,valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueMake1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average monthly volume' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    # Pocinje Economy-Monthly production in construction
    #
    elif (w_countries == 'Economy') and (w_countries1 == 'Monthly production in construction'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMPIC('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMPIC('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMPIC('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMPIC('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMPIC('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMPIC('Germany')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMPIC('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMPIC('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMPIC('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMPIC('Italy')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMPIC('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMPIC('Hungary')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMPIC('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMPIC('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMPIC('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMPIC('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMPIC('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMPIC('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMPIC('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMPIC('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMPIC('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMPIC('United Kingdom')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieMPIC('North Macedonia')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE''ES','FR','HR','IT','LU','HU','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','MK'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueSpa1,valueFra1,valueCro1,valueIta1,valueLux1,valueHun1,
                                valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueMake1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average production in construction' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #Pocinje deo Population and health
    #########################################
    ###########################################
    # POPULATION AND HEALTH-Monthly excess mortality
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMEM('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMEM('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMEM('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMEM('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMEM('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMEM('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMEM('Estonia')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMEM('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMEM('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMEM('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMEM('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMEM('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMEM('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMEM('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMEM('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMEM('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMEM('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMEM('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMEM('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMEM('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMEM('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMEM('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMEM('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMEM('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMEM('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMEM('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMEM('Sweden')

        valueIce1 = ''
        if ('Iceland' in coun):
            valueIce1 = getPieMEM('Iceland')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMEM('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMEM('Switzerland')
        return {
            'data': [go.Pie(labels=['EU','BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','IS','NO','CH'],
                            values=[valueEu1,valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueIce1,valueNor1,valueSwi1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average excess mortality' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #
    # Pocinje Number of deaths by week
    elif (w_countries == 'Population and health') & (w_countries1 == 'Number of deaths by week'):
        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieNOD('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieNOD('Bulgaria')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieNOD('Serbia')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieNOD('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieNOD('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieNOD('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieNOD('Estonia')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieNOD('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieNOD('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieNOD('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieNOD('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieNOD('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieNOD('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieNOD('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieNOD('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieNOD('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieNOD('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieNOD('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieNOD('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieNOD('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieNOD('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieNOD('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieNOD('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieNOD('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieNOD('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieNOD('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieNOD('Sweden')

        valueIce1 = ''
        if ('Iceland' in coun):
            valueIce1 = getPieNOD('Iceland')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieNOD('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieNOD('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieNOD('Switzerland')
        return {
            'data': [go.Pie(labels=['BE', 'BG', 'SR','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH'],
                            values=[valueBel1, valueBul1, valueSer1,valueChe1,valueDen1,valueGer1,valueEst1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average deaths by week' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Pocinje Polulation and healht-Monthly first-time asylum applicants
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly first-time asylum applicants'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMFTA('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMFTA('Belgium')

        valueBul1 = ''
        if ('Belgium' in coun):
            valueBul1 = getPieMFTA('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMFTA('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMFTA('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMFTA('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMFTA('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMFTA('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMFTA('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMFTA('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMFTA('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMFTA('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMFTA('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMFTA('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMFTA('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMFTA('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMFTA('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMFTA('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMFTA('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMFTA('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMFTA('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMFTA('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMFTA('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMFTA('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMFTA('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMFTA('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMFTA('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMFTA('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMFTA('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMFTA('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMFTA('Switzerland')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average first-time asylum' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }

    #####################Pocinje Society and work-Monthly unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly unemployment rate'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMUR('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMUR('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMUR('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMUR('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMUR('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMUR('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMUR('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMUR('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMUR('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMUR('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMUR('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMUR('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMUR('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMUR('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMUR('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMUR('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMUR('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMUR('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMUR('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMUR('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMUR('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMUR('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMUR('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMUR('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMUR('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMUR('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMUR('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMUR('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMUR('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMUR('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMIP('Switzerland')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieMUR('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','TR'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average unemployment rate' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Society and work-Monthly youth unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly youth unemployment rate'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMYUR('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMYUR('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMYUR('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMYUR('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMYUR('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMYUR('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMYUR('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMYUR('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMYUR('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMYUR('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMYUR('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMYUR('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMYUR('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMYUR('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMYUR('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMYUR('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMYUR('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMYUR('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMYUR('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMYUR('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMYUR('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMYUR('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMYUR('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMYUR('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMYUR('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMYUR('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMYUR('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMYUR('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMYUR('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMYUR('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMYUR('Switzerland')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieMYUR('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','TR'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average youth unemployment rate' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Society and work-Quarterly employment
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly employment'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieQE('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieQE('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieQE('Bulgaria')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieQE('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieQE('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieQE('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieQE('Ireland')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieQE('Spain')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieQE('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieQE('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieQE('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieQE('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieQE('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieQE('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieQE('Hungary')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieQE('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieQE('Austria')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieQE('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieQE('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieQE('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieQE('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieQE('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieQE('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieQE('Norway')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','DK','DE','EE','IE','ES','HR','IT','CY','LV','LT','LU','HU','NL','AT','RO','SI','SK','FI','SE','UK','NO'],
                            values=[valueEu1, valueBel1, valueBul1,valueDen1,valueGer1,valueEst1,valueIre1,valueSpa1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueNet1,valueAus1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average quarterly employment'+ '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Society and work-Quarterly labour market slack
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour market slack'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieQLMS('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieQLMS('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieQLMS('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieQLMS('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieQLMS('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieQLMS('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieQLMS('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieQLMS('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieQLMS('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieQLMS('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieQLMS('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieQLMS('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieQLMS('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieQLMS('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieQLMS('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieQLMS('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieQLMS('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieQLMS('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieQLMS('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieQLMS('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieQLMS('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieQLMS('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieQLMS('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieQLMS('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieQLMS('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieQLMS('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieQLMS('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieQLMS('Sweden')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieQLMS('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieQLMS('Switzerland')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieQLMS('Serbia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieQLMS('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','NO','CH','RS','TR'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueNor1,valueSwi1,valueSer1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average labour market slack' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Society and work-Quarterly job vacancy rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly job vacancy rate'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieQJVR('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieQJVR('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieQJVR('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieQJVR('Czechia')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieQJVR('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieQJVR('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieQJVR('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieQJVR('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieQJVR('Spain')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieQJVR('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieQJVR('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieQJVR('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieQJVR('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieQJVR('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieQJVR('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieQJVR('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieQJVR('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieQJVR('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieQJVR('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieQJVR('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieQJVR('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieQJVR('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieQJVR('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieQJVR('Sweden')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieQJVR('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieQJVR('Switzerland')

        valueUk1 = ''
        if ('North Macedonia' in coun):
            valueUk1 = getPieQJVR('United Kingdom')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieQJVR('North Macedonia')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DE','EE','IE','El','ES','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','NO','CH','UK','NM'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueNor1,valueSwi1,valueUk1,valueMake1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average job vacancy rate' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    #####################Society and work-Quarterly labour cost
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour cost'):

        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieQLC('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieQLC('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieQLC('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieQLC('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieQLC('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieQLC('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieQLC('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieQLC('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieQLC('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieQLC('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieQLC('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieQLC('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieQLC('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieQLC('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieQLC('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieQLC('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieQLC('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieQLC('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieQLC('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieQLC('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieQLC('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieQLC('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieQLC('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieQLC('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieQLC('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieQLC('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieQLC('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieQLC('Sweden')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieQLC('Norway')

        valueUk1 = ''
        if ('North Macedonia' in coun):
            valueUk1 = getPieQLC('United Kingdom')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieQLC('Serbia')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','NO','UK','RS'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueNor1,valueUk1,valueSer1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average labour cost' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')
            ),
        }
    #Krece oblast AGRICULTURE, ENERGY, TRANSPORT & TOURISM
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly air passenger transport
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly air passenger transport'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMAPT('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMAPT('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMAPT('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMAPT('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMAPT('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMAPT('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMAPT('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMAPT('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMAPT('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMAPT('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMAPT('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMAPT('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMAPT('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMAPT('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMAPT('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMAPT('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMAPT('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMAPT('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMAPT('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMAPT('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMAPT('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMAPT('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMAPT('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMAPT('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMAPT('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMAPT('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMAPT('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMAPT('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMAPT('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMAPT('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMAPT('Switzerland')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)
                            )],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average air passenger transport' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),
        }
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly commercial air flights
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly commercial air flights'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMCAF('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMCAF('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMCAF('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMCAF('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMCAF('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMCAF('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMCAF('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMCAF('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMCAF('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMCAF('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMCAF('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMCAF('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMCAF('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMCAF('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMCAF('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMCAF('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMCAF('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMCAF('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMCAF('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMCAF('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMCAF('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMCAF('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMCAF('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMCAF('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMCAF('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMCAF('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMCAF('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMCAF('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMCAF('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMCAF('Norway')

        valueSwi1 = ''
        if ('Switzerland' in coun):
            valueSwi1 = getPieMCAF('Switzerland')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieMCAF('North Macedonia')

        valueSer1 = ''
        if ('Serbia' in coun):
            valueSer1 = getPieMCAF('Serbia')

        valueTur1 = ''
        if ('Turkey' in coun):
            valueTur1 = getPieMCAF('Turkey')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH','MK','RS','TR'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueSwi1,valueMake1,valueSer1,valueTur1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Average commercial air flights' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),}
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly arrivals at tourist accommodation'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMATA('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMATA('Belgium')

        valueBul1 = ''
        if ('Belgium' in coun):
            valueBul1 = getPieMATA('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMATA('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMATA('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMATA('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMATA('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMATA('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMATA('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMATA('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMATA('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMATA('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMATA('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMATA('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMATA('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMATA('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMATA('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMATA('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMATA('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMATA('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMATA('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMATA('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMATA('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMATA('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMATA('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMATA('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMATA('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMATA('Sweden')

        valueIce1 = ''
        if ('Iceland' in coun):
            valueIce1 = getPieMATA('Iceland')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','IS'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueIce1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Arrivals at tourist accommodation'+ '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),}
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly nights spent at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly nights spent at tourist accommodation'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMNSTA('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMNSTA('Belgium')

        valueBul1 = ''
        if ('Bulgaria' in coun):
            valueBul1 = getPieMNSTA('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMNSTA('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMNSTA('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMNSTA('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMNSTA('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMNSTA('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMNSTA('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMNSTA('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMNSTA('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMNSTA('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMNSTA('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMNSTA('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMNSTA('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMNSTA('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMNSTA('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMNSTA('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMNSTA('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMNSTA('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMNSTA('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMNSTA('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMNSTA('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMNSTA('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMNSTA('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMNSTA('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMNSTA('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMNSTA('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMNSTA('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMNSTA('Norway')

        valueMake1 = ''
        if ('North Macedonia' in coun):
            valueMake1 = getPieMNSTA('North Macedonia')

        valueIce1 = ''
        if ('Iceland' in coun):
            valueIce1 = getPieMNSTA('Iceland')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','MK','IS'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1,valueMake1,valueIce1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Nights spent at tourist accommodation' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),}
######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly electricity consumed by end-users
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly electricity consumed by end-users'):
        valueEu1 = ''
        if ('European Union' in coun):
            valueEu1 = getPieMEC('European Union')

        valueBel1 = ''
        if ('Belgium' in coun):
            valueBel1 = getPieMEC('Belgium')

        valueBul1 = ''
        if ('Belgium' in coun):
            valueBul1 = getPieMEC('Bulgaria')

        valueChe1 = ''
        if ('Czechia' in coun):
            valueChe1 = getPieMEC('Czechia')

        valueDen1 = ''
        if ('Denmark' in coun):
            valueDen1 = getPieMEC('Denmark')

        valueGer1 = ''
        if ('Germany' in coun):
            valueGer1 = getPieMEC('Germany')

        valueEst1 = ''
        if ('Estonia' in coun):
            valueEst1 = getPieMEC('Estonia')

        valueIre1 = ''
        if ('Ireland' in coun):
            valueIre1 = getPieMEC('Ireland')

        valueGree1 = ''
        if ('Greece' in coun):
            valueGree1 = getPieMEC('Greece')

        valueSpa1 = ''
        if ('Spain' in coun):
            valueSpa1 = getPieMEC('Spain')

        valueFra1 = ''
        if ('France' in coun):
            valueFra1 = getPieMEC('France')

        valueCro1 = ''
        if ('Croatia' in coun):
            valueCro1 = getPieMEC('Croatia')

        valueIta1 = ''
        if ('Italy' in coun):
            valueIta1 = getPieMEC('Italy')

        valueCyp1 = ''
        if ('Cyprus' in coun):
            valueCyp1 = getPieMEC('Cyprus')

        valueLat1 = ''
        if ('Latvia' in coun):
            valueLat1 = getPieMEC('Latvia')

        valueLith1 = ''
        if ('Lithuania' in coun):
            valueLith1 = getPieMEC('Lithuania')

        valueLux1 = ''
        if ('Luxembourg' in coun):
            valueLux1 = getPieMEC('Luxembourg')

        valueHun1 = ''
        if ('Hungary' in coun):
            valueHun1 = getPieMEC('Hungary')

        valueMal1 = ''
        if ('Malta' in coun):
            valueMal1 = getPieMEC('Malta')

        valueNet1 = ''
        if ('Netherlands' in coun):
            valueNet1 = getPieMEC('Netherlands')

        valueAus1 = ''
        if ('Austria' in coun):
            valueAus1 = getPieMEC('Austria')

        valuePol1 = ''
        if ('Poland' in coun):
            valuePol1 = getPieMEC('Poland')

        valuePor1 = ''
        if ('Portugal' in coun):
            valuePor1 = getPieMEC('Portugal')

        valueRom1 = ''
        if ('Romania' in coun):
            valueRom1 = getPieMEC('Romania')

        valueSlo1 = ''
        if ('Slovenia' in coun):
            valueSlo1 = getPieMEC('Slovenia')

        valueSlovak1 = ''
        if ('Slovakia' in coun):
            valueSlovak1 = getPieMEC('Slovakia')

        valueFin1 = ''
        if ('Finland' in coun):
            valueFin1 = getPieMEC('Finland')

        valueSwe1 = ''
        if ('Sweden' in coun):
            valueSwe1 = getPieMEC('Sweden')

        valueUk1 = ''
        if ('United Kingdom' in coun):
            valueUk1 = getPieMEC('United Kingdom')

        valueNor1 = ''
        if ('Norway' in coun):
            valueNor1 = getPieMEC('Norway')
        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG','CZ','DK','DE','EE','IE','El','ES','FR','HR','IT','CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO'],
                            values=[valueEu1, valueBel1, valueBul1,valueChe1,valueDen1,valueGer1,valueEst1,valueIre1,valueGree1,valueSpa1,valueFra1,valueCro1,valueIta1,valueCyp1,valueLat1,valueLith1,valueLux1,valueHun1,
                                valueMal1,valueNet1,valueAus1,valuePol1,valuePor1,valueRom1,valueSlo1,valueSlovak1,valueFin1,valueSwe1,valueUk1,valueNor1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13))],
            'layout': go.Layout(
                plot_bgcolor='#808080',
                paper_bgcolor='#A8A8A8',
                hovermode='closest',
                title={
                    'text': 'Electricity consumed by end-users' + '</br>',
                    'y': 0.93,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                titlefont={
                    'color': 'white',
                    'size': 20},
                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.07},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white')),}
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
