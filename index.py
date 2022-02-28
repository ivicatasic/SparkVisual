import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import numpy as np
from gunicorn.app.wsgiapp import run

app = dash.Dash(__name__, )
app.title = 'Spark Visual Data'

terr2 = pd.read_csv('data/products.csv')
location1 = terr2[['subproducts']]
list_locations = location1.set_index('subproducts').T.to_dict('dict')
region = terr2['products'].unique()

conf = SparkConf().setAppName('Spark visual')
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
labels = ["6-2019", "7-2019", "8-2019", "9-2019", "10-2019", "11-2019", "12-2019", "1-2020", "2-2020", "3-2020",
          "4-2020", "5-2020", "6-2020", "7-2020", "8-2020", "9-2020", "10-2020", "11-2020", "12-2020", "1-2021",
          "2-2021", "3-2021", "4-2021", "5-2021", "6-2021", "7-2021", "8-2021", "9-2021", "10-2021", "11-2021", ]
lab = np.array(labels)
# labels for Economy GDP
labelsGDP = ["Q1-2017", "Q2-2017", "Q3-2017", "Q4-2017", "Q1-2018", "Q2-2018", "Q3-2018", "Q4-2018", "Q1-2019",
             "Q2-2019", "Q3-2019", "Q4-2019", "Q1-2020", "Q2-2020", "Q3-2020", "Q4-2020", "Q1-2021", "Q2-2021",
             "Q3-2021", "Q4-2021"]
labGDP = np.array(labelsGDP)
# labels for Economy Montly volume
labelsMV = ["1-2015", "2-2015", "3-2015", "4-2015", "5-2015", "6-2015", "7-2015", "8-2015", "9-2015", "10-2015",
            "11-2015", "12-2015", "1-2016", "2-2016", "3-2016", "4-2016", "5-2016", "6-2016", "7-2016", "8-2016",
            "9-2016", "10-2016", "11-2016", "12-2016", "1-2017", "2-2017", "3-2017", "4-2017", "5-2017", "6-2017",
            "7-2017", "8-2017", "9-2017", "10-2017", "11-2017", "12-2017", "1-2018", "2-2018", "3-2018", "4-2018",
            "5-2018", "6-2018", "7-2018", "8-2018", "9-2018", "10-2018", "11-2018", "12-2018", "1-2019", "2-2019",
            "3-2019", "4-2019", "5-2019",
            "6-2019", "7-2019", "8-2019", "9-2019", "10-2019", "11-2019", "12-2019", "1-2020", "2-2020", "3-2020",
            "4-2020", "5-2020", "6-2020", "7-2020", "8-2020", "9-2020", "10-2020", "11-2020", "12-2020", "1-2021",
            "2-2021", "3-2021", "4-2021", "5-2021", "6-2021", "7-2021", "8-2021", "9-2021", "10-2021", "11-2021", ]
labMV = np.array(labelsMV)
#######################################
#######################################
# LABELS FOR POPULATION AND HEALTH
# labels population and health-montly excess mortality
labelsPAHMEM = ["1-2020", "2-2020", "3-2020",
                "4-2020", "5-2020", "6-2020", "7-2020", "8-2020", "9-2020", "10-2020", "11-2020", "12-2020", "1-2021",
                "2-2021", "3-2021", "4-2021", "5-2021", "6-2021", "7-2021", "8-2021", "9-2021", "10-2021"]
labPAHMEM = np.array(labelsPAHMEM)
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
csv_SAWMUR2 = 'data/Society and work-Monthly unemployment rate.csv'
dataSAWMUR2 = sqlContext.read.format("csv").options(header='false').load(csv_SAWMUR2)
def getLabelSAWMUR2():
    value = []
    value1 = []
    data2 = dataSAWMUR2.first()
    for i in range(1, 35):
        value.append(data2[i])
    value1 = np.array(value)
    return value1
labelSAWMUR=["01-2019","02-2019","03-2019","04-2019","05-2019","06-2019","07-2019","08-2019","09-2019","10-2019","11-2019","12-2019",
"01-2020","02-2020","03-2020","04-2020","05-2020","06-2020","07-2020","08-2020","09-2020","10-2020","11-2020","12-2020","01-2021",
"02-2021","03-2021","04-2021","05-2021","06-2021","07-2021","08-2021","09-2021","10-2021","11-2021","12-2021"]
#Labele za Society work-Monthly youth unemployment rate
labelSAWMYUR = ["01-2019", "02-2019", "03-2019", "04-2019", "05-2019", "06-2019", "07-2019", "08-2019", "09-2019",
"10-2019", "11-2019", "12-2019", "01-2020", "02-2020", "03-2020", "04-2020", "05-2020", "06-2020", "07-2020",
"08-2020", "09-2020","10-2020", "11-2020", "12-2020", "01-2021","02-2021", "03-2021", "04-2021", "05-2021", "06-2021",
               "07-2021", "08-2021", "09-2021", "10-2021","11-2021", "12-2021"]
#Labele za Society and work-Quarterly employment
labelSAWQE=["Q1-2017","Q2-2017","Q3-2017","Q4-2017","Q1-2018","Q2-2018","Q3-2018","Q4-2018","Q1-2019","Q2-2019","Q3-2019","Q4-2019",
            "Q1-2020","Q2-2020","Q3-2020","Q4-2020","Q1-2021","Q2-2021","Q3-2021","Q3-2021"]
#Labele za Society and work-Quarterly labour market slack
labelSAWQLMS=["Q2-2019","Q3-2019","Q4-2019","Q1-2020","Q2-2020","Q3-2020","Q4-2020","Q1-2021","Q2-2021","Q3-2021","Q3-2021"]
############################
############################
##############
#Labele za Agriculture,energy, transport & tourism
###############
#Labele za Agriculture,energy, transport & tourism-Monthly air passenger transport
labelsAETTMAPT = ["1-2019","2-2019","3-2019","4-2019","5-2019","6-2019", "7-2019", "8-2019", "9-2019", "10-2019", "11-2019", "12-2019", "1-2020", "2-2020", "3-2020",
          "4-2020", "5-2020", "6-2020", "7-2020", "8-2020", "9-2020", "10-2020", "11-2020", "12-2020", "1-2021",
          "2-2021", "3-2021", "4-2021", "5-2021"]
#LAbele za: Agriculture,energy, transport & tourism-Monthly commercial air flights
labelsAETTMCAF=["1-2020","2-2020","3-2020","4-2020","5-2020","6-2020","7-2020","8-2020","9-2020","10-2020","11-2020","12-2020",
                "1-2021","2-2021", "3-2021", "4-2021", "5-2021","6-2021","7-2021","8-2021","9-2021","10-2021","11-2021","12-2021"]
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
# get Data for Economy Inflation
def getArr(countryName):
    value = []
    value1 = []
    data2 = data.where(data.GEOLABEL == countryName).collect()
    for i in range(2, 31):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
# get Data for Economy GDP
def getGDP(countryName):
    value = []
    value1 = []
    data2 = dataGDP.where(dataGDP.GEOLABEL == countryName).collect()
    for i in range(1, 19):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
# get Data for Economy Monthly industrial production
def getMIN(countryName):
    value = []
    value1 = []
    data2 = dataMIN.where(dataMIN.GEOLABEL == countryName).collect()
    for i in range(1, 29):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
# get Data for Economy Monthly volume
def getMV(countryName):
    value = []
    value1 = []
    data2 = dataMV.where(dataMV.GEOLABEL == countryName).collect()
    for i in range(1, 83):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
# get Data for Economy Monthly production in contruction
def getMPIC(countryName):
    value = []
    value1 = []
    data2 = dataMPIC.where(dataMPIC.GEOLABEL == countryName).collect()
    for i in range(1, 29):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
######################################
# Get data for population and health
def getPAHMEM(countryName):
    value = []
    value1 = []
    data2 = dataPAHMEM.where(dataPAHMEM.GEOLABEL == countryName).collect()
    for i in range(1, 22):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
###Uzimanje podataka iz fajla Population and health-number of deaths by week
def getPAHDBW(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataPAHDBW.where(dataPAHDBW.GEOLABEL == countryName).collect()
    for i in range(1, 109):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1,95):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
##########################
###Uzimanje podataka iz fajle: POpulation and health-Monthly first-time asylum
def getPAHMFTA(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataPAHMFTA.where(dataPAHMFTA.GEOLABEL == countryName).collect()
    for i in range(1, 160):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(1,150):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
############################################
#############################################
#UZIMANJE PODATAKA ZA GRUPU: SOCIETY AND WOKR
#Society and work-Monthly unemployment rate
def getSAWMUR(countryName):
    value = []
    value1 = []
    data2 = dataSAWMUR.where(dataSAWMUR.GEOLABEL == countryName).collect()
    for i in range(1, 36):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
#Society and work-Monthly youth unemployment rate
def getSAWMYUR(countryName):
    value = []
    value1 = []
    data2 = dataSAWMYUR.where(dataSAWMYUR.GEOLABEL == countryName).collect()
    for i in range(1, 33):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
#Society and work-Quarterly employment
def getSAWQE(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataSAWQE.where(dataSAWQE.GEOLABEL == countryName).collect()
    for i in range(1, 20):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(0,18):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
#Society and work-Quarterly labour market slack
def getSAWQLMS(countryName):
    value = []
    value1 = []
    data2 = dataSAWQLMS.where(dataSAWQLMS.GEOLABEL == countryName).collect()
    for i in range(1, 10):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
#Society and work-Quarterly job vacancy rate
def getSAWQJVR(countryName):
    value = []
    value1 = []
    data2 = dataSAWQJVR.where(dataSAWQJVR.GEOLABEL == countryName).collect()
    for i in range(1, 19):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
#Society and work-Quarterly labour cost
def getSAWQLC(countryName):
    value = []
    value1 = []
    data2 = dataSAWQLC.where(dataSAWQLC.GEOLABEL == countryName).collect()
    for i in range(1, 19):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
##########################################
##########################################
#UZIMANJE PODATAKA ZA GRUPU:AGRICULTURE, ENERGY, TRANSPORT & TOURISM
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly air passenger transport
def getAETTMAPT(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataAETTMAPT.where(dataAETTMAPT.GEOLABEL == countryName).collect()
    for i in range(1, 30):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(0,28):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly commercial air flights
def getAETTMCAF(countryName):
    value = []
    value1 = []
    data2 = dataAETTMCAF.where(dataAETTMCAF.GEOLABEL == countryName).collect()
    for i in range(1, 24):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly arrivals at tourist accommodation
def getAETTMATA(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataAETTMATA.where(dataAETTMATA.GEOLABEL == countryName).collect()
    for i in range(1, 20):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(0,18):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly nights spent at tourist accommodation
def getAETTMNSTA(countryName):
    value = []
    value1 = []
    value2 = []
    value3 = []
    data2 = dataAETTMNSTA.where(dataAETTMNSTA.GEOLABEL == countryName).collect()
    for i in range(1, 20):
        val=data2[0][i]
        value.append(val)
    value1 = np.array(value)
    for i in range(0,18):
        val=float(value[i].replace(',',''))
        value2.append(val)
    value3=np.array(value2)
    return value3
#AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly electricity consumed by end-users
def getAETTMEC(countryName):
    value = []
    value1 = []
    data2 = dataAETTMEC.where(dataAETTMEC.GEOLABEL == countryName).collect()
    for i in range(1, 24):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getArr('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getArr('Malta')

        valueSer1 = []
        if ('Serbia' in coun):
            valueSer1 = getArr('Serbia')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getArr('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getArr('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getArr('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getArr('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getArr('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getArr('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getArr('Estonia')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getArr('Ireland')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getArr('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getArr('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getArr('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getArr('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getArr('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getArr('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getArr('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getArr('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getArr('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getArr('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getArr('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getArr('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getArr('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getArr('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getArr('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getArr('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getArr('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getArr('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getArr('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getArr('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getArr('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getArr('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getArr('Switzerland')

        valueMake1 = []
        if ('North Macedonia' in coun):
            valueMake1 = getArr('North Macedonia')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getArr('Turkey')

        valueUs1 = []
        if ('United States' in coun):
            valueUs1 = getArr('United States')

        ei = pd.read_csv('data/Economy-inflation.csv')
        df1 = {}
        for i in range(0, 38):
            niz = ei.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueSer1,
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
                                y=valueMal1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueMake1,
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
                                y=valueTur1,
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
                                y=valueUs1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getGDP('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getGDP('Malta')

        valueSer1 = []
        if ('Serbia' in coun):
            valueSer1 = getGDP('Serbia')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getGDP('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getGDP('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getGDP('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getGDP('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getGDP('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getGDP('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getGDP('Estonia')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getGDP('Ireland')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getGDP('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getGDP('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getGDP('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getGDP('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getGDP('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getGDP('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getGDP('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getGDP('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getGDP('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getGDP('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getGDP('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getGDP('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getGDP('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getGDP('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getGDP('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getGDP('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getGDP('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getGDP('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getGDP('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getGDP('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getGDP('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getGDP('Switzerland')

        valueMake1 = []
        if ('North Macedonia' in coun):
            valueMake1 = getGDP('North Macedonia')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getGDP('Turkey')

        gdp = pd.read_csv('data/Economy-GDP.csv')
        df1 = {}
        for i in range(0, 35):
            niz = gdp.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labGDP,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueSer1,
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
                                y=valueMal1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueMake1,
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
                                y=valueTur1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getMIN('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getMIN('Malta')

        valueSer1 = []
        if ('Serbia' in coun):
            valueSer1 = getMIN('Serbia')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getMIN('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getMIN('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getMIN('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getMIN('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getMIN('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getMIN('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getMIN('Estonia')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getMIN('Ireland')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getMIN('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getMIN('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getMIN('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getMIN('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getMIN('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getMIN('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getMIN('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getMIN('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getMIN('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getMIN('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getMIN('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getMIN('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getMIN('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getMIN('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getMIN('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getMIN('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getMIN('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getMIN('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getMIN('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getMIN('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getMIN('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getMIN('Switzerland')

        valueMake1 = []
        if ('North Macedonia' in coun):
            valueMake1 = getMIN('North Macedonia')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getMIN('Turkey')

        eMip = pd.read_csv('data/Economy-Monthly industrial production.csv')
        df1 = {}
        for i in range(0, 36):
            niz = eMip.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueSer1,
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
                                y=valueMal1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueMake1,
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
                                y=valueTur1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getMV('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getMV('Malta')

        valueSer1 = []
        if ('Serbia' in coun):
            valueSer1 = getMV('Serbia')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getMV('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getMV('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getMV('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getMV('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getMV('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getMV('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getMV('Estonia')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getMV('Ireland')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getMV('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getMV('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getMV('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getMV('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getMV('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getMV('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getMV('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getMV('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getMV('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getMV('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getMV('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getMV('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getMV('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getMV('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getMV('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getMV('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getMV('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getMV('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getMV('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getMV('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getMV('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getMV('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getMV('Switzerland')

        valueMake1 = []
        if ('North Macedonia' in coun):
            valueMake1 = getMV('North Macedonia')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getMV('Turkey')

        valueUs1 = []
        if ('United States' in coun):
            valueUs1 = getMV('United States')

        mv = pd.read_csv('data/Economy-Monthly volume.csv')
        df1 = {}
        for i in range(0, 38):
            niz = mv.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labMV,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueSer1,
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
                                y=valueMal1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[31].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueMake1,
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
                                y=valueTur1,
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[36].astype(str) + '<br>'),
                     go.Scatter(x=labMV,
                                y=valueUs1,
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2) ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United States' + '<br>'+
                                '<b>Monthly volume of retail trade</b>: ' + df1[37].astype(str) + '<br>' ),],
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getMPIC('European Union')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getMPIC('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getMPIC('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getMPIC('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getMPIC('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getMPIC('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getMPIC('Germany')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getMPIC('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getMPIC('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getMPIC('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getMPIC('Italy')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getMPIC('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getMPIC('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getMPIC('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getMPIC('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getMPIC('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getArr('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getMPIC('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getMPIC('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getMPIC('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getMPIC('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getMPIC('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getMPIC('United Kingdom')

        valueMake1 = []
        if ('North Macedonia' in coun):
            valueMake1 = getMPIC('North Macedonia')

        mpic = pd.read_csv('data/Economy-Monthly production in construction.csv')
        df1 = {}
        for i in range(0, 24):
            niz = mpic.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueMake1,
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
    # Start POPULATION AND HEALTH
    #
    # POPULATION AND HEALTH-Monthly excess mortality
    #
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getPAHMEM('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getPAHMEM('Malta')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getPAHMEM('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getPAHMEM('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getPAHMEM('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getPAHMEM('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getPAHMEM('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getPAHMEM('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getPAHMEM('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getPAHMEM('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getPAHMEM('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getPAHMEM('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getPAHMEM('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getPAHMEM('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getPAHMEM('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getPAHMEM('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getPAHMEM('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getPAHMEM('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getPAHMEM('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getPAHMEM('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getPAHMEM('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getPAHMEM('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getPAHMEM('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getPAHMEM('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getPAHMEM('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getPAHMEM('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getPAHMEM('Sweden')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getPAHMEM('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getPAHMEM('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getPAHMEM('Switzerland')

        pop = pd.read_csv('data/Population and health-Monthly excess mortality.csv')
        df1 = {}
        for i in range(0, 31):
            niz = pop.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labPAHMEM,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueMal1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
            valueBul1 = []
            if ('Bulgaria' in coun):
                valueBul1 = getPAHDBW('Bulgaria')

            valueBel1 = []
            if ('Belgium' in coun):
                valueBel1 = getPAHDBW('Belgium')

            valueMal1 = []
            if ('Malta' in coun):
                valueMal1 = getPAHDBW('Malta')

            labelDbw = getLabelPAHDBW2()

            valueSer1 = []
            if ('Serbia' in coun):
                valueSer1 = getPAHDBW('Serbia')

            valueChe1 = []
            if ('Czechia' in coun):
                valueChe1 = getPAHDBW('Czechia')

            valueDen1 = []
            if ('Denmark' in coun):
                valueDen1 = getPAHDBW('Denmark')

            valueGer1 = []
            if ('Germany' in coun):
                valueGer1 = getPAHDBW('Germany')

            valueEst1 = []
            if ('Estonia' in coun):
                valueEst1 = getPAHDBW('Estonia')

            valueGree1 = []
            if ('Greece' in coun):
                valueGree1 = getPAHDBW('Greece')

            valueSpa1 = []
            if ('Spain' in coun):
                valueSpa1 = getPAHDBW('Spain')

            valueFra1 = []
            if ('France' in coun):
                valueFra1 = getPAHDBW('France')

            valueCro1 = []
            if ('Croatia' in coun):
                valueCro1 = getPAHDBW('Croatia')

            valueIta1 = []
            if ('Italy' in coun):
                valueIta1 = getPAHDBW('Italy')

            valueCyp1 = []
            if ('Cyprus' in coun):
                valueCyp1 = getPAHDBW('Cyprus')

            valueLat1 = []
            if ('Latvia' in coun):
                valueLat1 = getPAHDBW('Latvia')

            valueLith1 = []
            if ('Lithuania' in coun):
                valueLith1 = getPAHDBW('Lithuania')

            valueLux1 = []
            if ('Luxembourg' in coun):
                valueLux1 = getPAHDBW('Luxembourg')

            valueHun1 = []
            if ('Hungary' in coun):
                valueHun1 = getPAHDBW('Hungary')

            valueNet1 = []
            if ('Netherlands' in coun):
                valueNet1 = getPAHDBW('Netherlands')

            valueAus1 = []
            if ('Austria' in coun):
                valueAus1 = getPAHDBW('Austria')

            valuePol1 = []
            if ('Poland' in coun):
                valuePol1 = getPAHDBW('Poland')

            valuePor1 = []
            if ('Portugal' in coun):
                valuePor1 = getPAHDBW('Portugal')

            valueRom1 = []
            if ('Romania' in coun):
                valueRom1 = getPAHDBW('Romania')

            valueSlo1 = []
            if ('Slovenia' in coun):
                valueSlo1 = getPAHDBW('Slovenia')

            valueSlovak1 = []
            if ('Slovakia' in coun):
                valueSlovak1 = getPAHDBW('Slovakia')

            valueFin1 = []
            if ('Finland' in coun):
                valueFin1 = getPAHDBW('Finland')

            valueSwe1 = []
            if ('Sweden' in coun):
                valueSwe1 = getPAHDBW('Sweden')

            valueUk1 = []
            if ('United Kingdom' in coun):
                valueUk1 = getPAHDBW('United Kingdom')

            valueIce1 = []
            if ('Iceland' in coun):
                valueIce1 = getPAHDBW('Iceland')

            valueNor1 = []
            if ('Norway' in coun):
                valueNor1 = getPAHDBW('Norway')

            valueSwi1 = []
            if ('Switzerland' in coun):
                valueSwi1 = getPAHDBW('Switzerland')

            dbw = pd.read_csv('data/Population and health-Number of deaths by week.csv')
            df1 = {}
            for i in range(0, 34):
                niz = dbw.loc[i, :]
                df1[i] = niz
            return {
                'data': [go.Scatter(x=labelDbw,
                                    y=valueBul1,
                                    mode='lines+markers',
                                    name='Bulgarua',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#E6D1D1', width=2)),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Bulgaria' + '<br>'+
                                    '<b>Number of deaths by week</b>: ' + df1[1].astype(str) + '<br>'),
                         go.Scatter(x=labelDbw,
                                    y=valueBel1,
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
                                    y=valueSer1,
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
                                    y=valueMal1,
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
                                    y=valueChe1,
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
                                    y=valueDen1,
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
                                    y=valueGer1,
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
                                    y=valueEst1,
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
                                    y=valueGree1,
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
                                    y=valueSpa1,
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
                                    y=valueFra1,
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
                                    y=valueCro1,
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
                                    y=valueIta1,
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
                                    y=valueCyp1,
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
                                    y=valueLat1,
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
                                    y=valueLith1,
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
                                    y=valueLux1,
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
                                    y=valueHun1,
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
                                    y=valueNet1,
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
                                    y=valueAus1,
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
                                    y=valuePol1,
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
                                    y=valuePor1,
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
                                    y=valueRom1,
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
                                    y=valueSlo1,
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
                                    y=valueSlovak1,
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
                                    y=valueFin1,
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
                                    y=valueSwe1,
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
                                    y=valueUk1,
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
                                    y=valueIce1,
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
                                    y=valueNor1,
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
                                    y=valueSwi1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getPAHMFTA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getPAHMFTA('Belgium')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getPAHMFTA('Malta')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getPAHMFTA('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getPAHMFTA('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getPAHMFTA('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getPAHMFTA('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getPAHMFTA('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getPAHMFTA('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getPAHMFTA('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getPAHMFTA('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getPAHMFTA('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getPAHMFTA('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getPAHMFTA('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getPAHMFTA('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getPAHMFTA('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getPAHMFTA('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getPAHMFTA('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getPAHMFTA('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getPAHMFTA('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getPAHMFTA('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getPAHMFTA('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getPAHMFTA('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getPAHMFTA('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getPAHMFTA('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getPAHMFTA('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getPAHMFTA('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getPAHMFTA('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getPAHMFTA('Switzerland')

        mfta = pd.read_csv('data/Population and health-Monthly first-time asylum applicants.csv')
        df1 = {}
        for i in range(0, 31):
            niz = mfta.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelPAHMFTA2,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueMal1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWMUR('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWMUR('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWMUR('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getSAWMUR('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getSAWMUR('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWMUR('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWMUR('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getSAWMUR('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWMUR('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getSAWMUR('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getSAWMUR('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getSAWMUR('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWMUR('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWMUR('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWMUR('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWMUR('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWMUR('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWMUR('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWMUR('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getSAWMUR('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getSAWMUR('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWMUR('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWMUR('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWMUR('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWMUR('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWMUR('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getSAWMUR('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getSAWMUR('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWMUR('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getSAWMUR('Switzerland')

        valueUs1 = []
        if ('United States' in coun):
            valueUs1 = getSAWMUR('United States')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWMUR('Euro area')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getSAWMUR('Turkey')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWMUR('Ireland')

        mur = pd.read_csv('data/Society and work-Monthly unemployment rate.csv')
        df1 = {}
        for i in range(0, 36):
            niz = mur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWMUR,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueTur1,
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
                                y=valueUs1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWMYUR('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWMYUR('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWMYUR('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getSAWMYUR('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getSAWMYUR('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWMYUR('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWMYUR('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getSAWMYUR('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWMYUR('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getSAWMYUR('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getSAWMYUR('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getSAWMYUR('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWMYUR('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWMYUR('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWMYUR('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWMYUR('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWMYUR('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWMYUR('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWMYUR('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getSAWMYUR('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getSAWMYUR('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWMYUR('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWMYUR('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWMYUR('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWMYUR('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWMYUR('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getSAWMYUR('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getSAWMYUR('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWMYUR('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getSAWMYUR('Switzerland')

        valueUs1 = []
        if ('United States' in coun):
            valueUs1 = getSAWMYUR('United States')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWMYUR('Euro area')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getSAWMYUR('Turkey')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWMYUR('Ireland')

        myur = pd.read_csv('data/Society and work-Monthly youth unemployment rate.csv')
        df1 = {}
        for i in range(0, 36):
            niz = myur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWMYUR,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueTur1,
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
                                y=valueUs1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQE('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQE('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWQE('Bulgaria')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getSAWQE('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWQE('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWQE('Estonia')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWQE('Spain')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getSAWQE('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getSAWQE('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWQE('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWQE('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWQE('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWQE('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWQE('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWQE('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWQE('Austria')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWQE('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWQE('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWQE('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWQE('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWQE('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getSAWQE('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getSAWQE('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWQE('Norway')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWQE('Euro area')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWQE('Ireland')

        qe = pd.read_csv('data/Society and work-Quarterly employment.csv')
        df1 = {}
        for i in range(0, 26):
            niz = qe.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueSpa1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQLMS('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQLMS('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWQLMS('Bulgaria')

        valueSer1 = []
        if ('Serbia' in coun):
            valueSer1 = getSAWQLMS('Serbia')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getSAWQLMS('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getSAWQLMS('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWQLMS('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWQLMS('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getSAWQLMS('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWQLMS('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getSAWQLMS('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getSAWQLMS('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getSAWQLMS('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWQLMS('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWQLMS('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWQLMS('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWQLMS('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWQLMS('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWQLMS('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWQLMS('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getSAWQLMS('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getSAWQLMS('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWQLMS('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWQLMS('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWQLMS('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWQLMS('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWQLMS('Sweden')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getSAWQLMS('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWQLMS('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getSAWQLMS('Switzerland')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWQLMS('Euro area')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getSAWQLMS('Turkey')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWQLMS('Ireland')

        qlms = pd.read_csv('data/Society and work-Quarterly labour market slack.csv')
        df1 = {}
        for i in range(0, 36):
            niz = qlms.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQLMS,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueTur1,
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
                                y=valueSer1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQJVR('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQJVR('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWQJVR('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getSAWQJVR('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getSAWQJVR('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWQJVR('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWQJVR('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getSAWQJVR('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWQJVR('Spain')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWQJVR('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWQJVR('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWQJVR('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWQJVR('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWQJVR('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWQJVR('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWQJVR('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getSAWQJVR('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getSAWQJVR('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWQJVR('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWQJVR('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWQJVR('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWQJVR('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWQJVR('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getSAWQJVR('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWQJVR('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getSAWQJVR('Switzerland')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWQJVR('Euro area')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWQJVR('Ireland')

        mur = pd.read_csv('data/Society and work-Quarterly job vacancy rate.csv')
        df1 = {}
        for i in range(0, 29):
            niz = mur.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>' +
                                '<b>Quarterly job vacancy rate</b>: ' + df1[5].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQLC('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQLC('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getSAWQLC('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getSAWQLC('Czechia')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getSAWQLC('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getSAWQLC('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getSAWQLC('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getSAWQLC('Spain')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getSAWQLC('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getSAWQLC('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getSAWQLC('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getSAWQLC('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getSAWQLC('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getSAWQLC('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getSAWQLC('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getSAWQLC('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getSAWQLC('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getSAWQLC('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getSAWQLC('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getSAWQLC('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getSAWQLC('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getSAWQLC('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getSAWQLC('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getSAWQLC('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getSAWQLC('Switzerland')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getSAWQLC('Euro area')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getSAWQLC('Ireland')

        qlc = pd.read_csv('data/Society and work-Quarterly labour cost.csv')
        df1 = {}
        for i in range(0, 29):
            niz = qlc.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueEa1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[25].astype(str) + '<br>'),
                     go.Scatter(x=labelSAWQE,
                                y=valueSwi1,
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>' +
                                '<b>Quarterly labour cost</b>: ' + df1[26].astype(str) + '<br>'),],
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMAPT('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMAPT('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getAETTMAPT('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getAETTMAPT('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getAETTMAPT('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getAETTMAPT('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getAETTMAPT('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getAETTMAPT('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getAETTMAPT('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getAETTMAPT('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getAETTMAPT('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getAETTMAPT('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getAETTMAPT('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getAETTMAPT('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getAETTMAPT('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getAETTMAPT('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getAETTMAPT('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getAETTMAPT('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getAETTMAPT('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getAETTMAPT('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getAETTMAPT('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getAETTMAPT('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getAETTMAPT('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getAETTMAPT('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getAETTMAPT('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getAETTMAPT('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getAETTMAPT('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getAETTMAPT('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getAETTMAPT('Switzerland')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getAETTMAPT('Ireland')

        mapt = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly air passenger transport.csv')
        df1 = {}
        for i in range(0, 31):
            niz = mapt.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMAPT,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMCAF('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMCAF('Belgium')

        valueBul1=[]
        if ('Bulgaria' in coun):
            valueBul1 = getAETTMCAF('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getAETTMCAF('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getAETTMCAF('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getAETTMCAF('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getAETTMCAF('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getAETTMCAF('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getAETTMCAF('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getAETTMCAF('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getAETTMCAF('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getAETTMCAF('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getAETTMCAF('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getAETTMCAF('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getAETTMCAF('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getAETTMCAF('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getAETTMCAF('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getAETTMCAF('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getAETTMCAF('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getAETTMCAF('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getAETTMCAF('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getAETTMCAF('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getAETTMCAF('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getAETTMCAF('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getAETTMCAF('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getAETTMCAF('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getAETTMCAF('United Kingdom')

        valueIce1 = []
        if ('Iceland' in coun):
            valueIce1 = getAETTMCAF('Iceland')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getAETTMCAF('Norway')

        valueSwi1 = []
        if ('Switzerland' in coun):
            valueSwi1 = getAETTMCAF('Switzerland')

        valueTur1 = []
        if ('Turkey' in coun):
            valueTur1 = getAETTMCAF('Turkey')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getAETTMCAF('Ireland')

        mcaf = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly commercial air flights.csv')
        df1 = {}
        for i in range(0, 37):
            niz = mcaf.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueIce1,
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
                                y=valueNor1,
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
                                y=valueSwi1,
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
                                y=valueTur1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMATA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMATA('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getAETTMATA('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getAETTMATA('Czechia')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getAETTMATA('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getAETTMATA('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getAETTMATA('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getAETTMATA('Spain')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getAETTMATA('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getAETTMATA('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getAETTMATA('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getAETTMATA('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getAETTMATA('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getAETTMATA('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getAETTMATA('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getAETTMATA('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getAETTMATA('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getAETTMATA('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getAETTMATA('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getAETTMATA('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getAETTMATA('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getAETTMATA('Sweden')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getAETTMATA('Ireland')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getAETTMATA('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getAETTMATA('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getAETTMATA('Italy')

        mata = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation.csv')

        df1 = {}
        for i in range(0, 29):
            niz = mata.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMNSTA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMNSTA('Belgium')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getAETTMNSTA('Bulgaria')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getAETTMNSTA('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getAETTMNSTA('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getAETTMNSTA('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getAETTMNSTA('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getAETTMNSTA('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getAETTMNSTA('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getAETTMNSTA('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getAETTMNSTA('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getAETTMNSTA('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getAETTMNSTA('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getAETTMNSTA('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getAETTMNSTA('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getAETTMNSTA('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getAETTMNSTA('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getAETTMNSTA('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getAETTMNSTA('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getAETTMNSTA('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getAETTMNSTA('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getAETTMNSTA('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getAETTMNSTA('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getAETTMNSTA('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getAETTMNSTA('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getAETTMNSTA('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getAETTMNSTA('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getAETTMNSTA('Norway')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getAETTMNSTA('Ireland')

        mnsta = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation.csv')
        df1 = {}
        for i in range(0, 32):
            niz = mnsta.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMEC('European Union')

        valueBul1 = []
        if ('Bulgaria' in coun):
            valueBul1 = getAETTMEC('Bulgaria')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMEC('Belgium')

        valueChe1 = []
        if ('Czechia' in coun):
            valueChe1 = getAETTMEC('Czechia')

        valueDen1 = []
        if ('Denmark' in coun):
            valueDen1 = getAETTMEC('Denmark')

        valueGer1 = []
        if ('Germany' in coun):
            valueGer1 = getAETTMEC('Germany')

        valueEst1 = []
        if ('Estonia' in coun):
            valueEst1 = getAETTMEC('Estonia')

        valueGree1 = []
        if ('Greece' in coun):
            valueGree1 = getAETTMEC('Greece')

        valueSpa1 = []
        if ('Spain' in coun):
            valueSpa1 = getAETTMEC('Spain')

        valueFra1 = []
        if ('France' in coun):
            valueFra1 = getAETTMEC('France')

        valueCro1 = []
        if ('Croatia' in coun):
            valueCro1 = getAETTMEC('Croatia')

        valueIta1 = []
        if ('Italy' in coun):
            valueIta1 = getAETTMEC('Italy')

        valueCyp1 = []
        if ('Cyprus' in coun):
            valueCyp1 = getAETTMEC('Cyprus')

        valueLat1 = []
        if ('Latvia' in coun):
            valueLat1 = getAETTMEC('Latvia')

        valueLith1 = []
        if ('Lithuania' in coun):
            valueLith1 = getAETTMEC('Lithuania')

        valueLux1 = []
        if ('Luxembourg' in coun):
            valueLux1 = getAETTMEC('Luxembourg')

        valueHun1 = []
        if ('Hungary' in coun):
            valueHun1 = getAETTMEC('Hungary')

        valueNet1 = []
        if ('Netherlands' in coun):
            valueNet1 = getAETTMEC('Netherlands')

        valueAus1 = []
        if ('Austria' in coun):
            valueAus1 = getAETTMEC('Austria')

        valuePol1 = []
        if ('Poland' in coun):
            valuePol1 = getAETTMEC('Poland')

        valuePor1 = []
        if ('Portugal' in coun):
            valuePor1 = getAETTMEC('Portugal')

        valueRom1 = []
        if ('Romania' in coun):
            valueRom1 = getAETTMEC('Romania')

        valueSlo1 = []
        if ('Slovenia' in coun):
            valueSlo1 = getAETTMEC('Slovenia')

        valueSlovak1 = []
        if ('Slovakia' in coun):
            valueSlovak1 = getAETTMEC('Slovakia')

        valueFin1 = []
        if ('Finland' in coun):
            valueFin1 = getAETTMEC('Finland')

        valueSwe1 = []
        if ('Sweden' in coun):
            valueSwe1 = getAETTMEC('Sweden')

        valueUk1 = []
        if ('United Kingdom' in coun):
            valueUk1 = getAETTMEC('United Kingdom')

        valueNor1 = []
        if ('Norway' in coun):
            valueNor1 = getAETTMEC('Norway')

        valueIre1 = []
        if ('Ireland' in coun):
            valueIre1 = getAETTMEC('Ireland')

        mecbu = pd.read_csv('data/Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users.csv')
        df1 = {}
        for i in range(0, 30):
            niz = mecbu.loc[i, :]
            df1[i] = niz
        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
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
                                y=valueBel1,
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
                                y=valueBul1,
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
                                y=valueChe1,
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
                                y=valueDen1,
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
                                y=valueGer1,
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
                                y=valueEst1,
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
                                y=valueIre1,
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
                                y=valueGree1,
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
                                y=valueSpa1,
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
                                y=valueFra1,
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
                                y=valueCro1,
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
                                y=valueIta1,
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
                                y=valueCyp1,
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
                                y=valueLat1,
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
                                y=valueLith1,
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
                                y=valueLux1,
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
                                y=valueHun1,
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
                                y=valueNet1,
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
                                y=valueAus1,
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
                                y=valuePol1,
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
                                y=valuePor1,
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
                                y=valueRom1,
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
                                y=valueSlo1,
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
                                y=valueSlovak1,
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
                                y=valueFin1,
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
                                y=valueSwe1,
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
                                y=valueUk1,
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
                                y=valueNor1,
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
