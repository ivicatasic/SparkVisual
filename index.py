import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import numpy as np

app = dash.Dash(__name__, )
app.title = 'Spark Visual Data'

terr2 = pd.read_csv('C:/Users/Korisnik/Desktop/SparkVisual/data/products.csv')
location1 = terr2[['subproducts']]
list_locations = location1.set_index('subproducts').T.to_dict('dict')
region = terr2['products'].unique()

conf = SparkConf().setAppName('Spark visual')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# upload Economy inflation file
csv_EI = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-inflation.csv'
data = sqlContext.read.format("csv").options(header='true').load(csv_EI)

country = pd.read_csv(csv_EI)
countries = country['GEOLABEL'].unique()

# upload Economy GDP file
csv_EGDP = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-GDP.csv'
dataGDP = sqlContext.read.format("csv").options(header='true').load(csv_EGDP)

# upload Economy Monthly industrial production
csv_EMIN = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly industrial production.csv'
dataMIN = sqlContext.read.format("csv").options(header='true').load(csv_EMIN)

# upload Economy montly volume
csv_EMV = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly volume.csv'
dataMV = sqlContext.read.format("csv").options(header='true').load(csv_EMV)

# upload Economy montly production in construction
csv_EMPIC = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly production in construction.csv'
dataMPIC = sqlContext.read.format("csv").options(header='true').load(csv_EMPIC)

####################################
####################################
# Upload Population and Health

#Ucitavanje fajla: Population and health-Monthly excess mortality
csv_PAHMEM = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Monthly excess mortality.csv'
dataPAHMEM = sqlContext.read.format("csv").options(header='true').load(csv_PAHMEM)

# Ucitavanje fajla: Population and health-Number of deaths by week
csv_PAHDBW = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Number of deaths by week.csv'
dataPAHDBW = sqlContext.read.format("csv").options(header='true').load(csv_PAHDBW)

#Ucitavanje fajla: Population and health-Monthly first-time asylum applicants
csv_PAHMFTA = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Monthly first-time asylum applicants.csv'
dataPAHMFTA = sqlContext.read.format("csv").options(header='true').load(csv_PAHMFTA)

##############################################
#############################################
#Krece grupa: Society and work

#Ucitavanje fajla Monthly unemployment rate
csv_SAWMUR = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Monthly unemployment rate.csv'
dataSAWMUR = sqlContext.read.format("csv").options(header='true').load(csv_SAWMUR)

#Ucitavanje fajla: Society and work-Monthly youth unemployment rate
csv_SAWMYUR = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Monthly youth unemployment rate.csv'
dataSAWMYUR = sqlContext.read.format("csv").options(header='true').load(csv_SAWMYUR)

#Ucitavanje fajla: Society and work-Quarterly employment
csv_SAWQE = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Quarterly employment.csv'
dataSAWQE = sqlContext.read.format("csv").options(header='true').load(csv_SAWQE)

#Ucitavanje fajla: Society and work-Quarterly labour market slack
csv_SAWQLMS = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Quarterly labour market slack.csv'
dataSAWQLMS = sqlContext.read.format("csv").options(header='true').load(csv_SAWQLMS)

#Ucitavanje fajla: Society and work-Quarterly job vacancy rate
csv_SAWQJVR = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Quarterly job vacancy rate.csv'
dataSAWQJVR = sqlContext.read.format("csv").options(header='true').load(csv_SAWQJVR)

#Ucitavanje fajla:Society and work-Quarterly labour cost
csv_SAWQLC = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Quarterly labour cost.csv'
dataSAWQLC = sqlContext.read.format("csv").options(header='true').load(csv_SAWQLC)
######################################
######################################

#KRECE OBLAST AGRICULTURE, ENERGY, TRANSPORT & TOURISM

#Ucitavanje fajla Agriculture, energy, transport & tourism-Monthly air passenger transport
csv_AETTMAPT = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Agriculture, energy, transport & tourism-Monthly air passenger transport.csv'
dataAETTMAPT = sqlContext.read.format("csv").options(header='true').load(csv_AETTMAPT)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly commercial air flights
csv_AETTMCAF = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Agriculture, energy, transport & tourism-Monthly commercial air flights.csv'
dataAETTMCAF = sqlContext.read.format("csv").options(header='true').load(csv_AETTMCAF)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation
csv_AETTMATA = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation.csv'
dataAETTMATA = sqlContext.read.format("csv").options(header='true').load(csv_AETTMATA)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation
csv_AETTMNSTA = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Agriculture, energy, transport & tourism-Monthly nights spent at tourist accommodation.csv'
dataAETTMNSTA = sqlContext.read.format("csv").options(header='true').load(csv_AETTMNSTA)

#Ucitavanje fajla: Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users
csv_AETTMEC = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users.csv'
dataAETTMEC = sqlContext.read.format("csv").options(header='true').load(csv_AETTMEC)


############################################
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
csv_PAHDBW2 = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Number of deaths by week.csv'
dataPAHDBW2 = sqlContext.read.format("csv").options(header='false').load(csv_PAHDBW)

#Labela za population and health-Monthly first-time asylum applicants
csv_PAHMFTA2 = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Monthly first-time asylum applicants.csv'
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
csv_SAWMUR2 = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Society and work-Monthly unemployment rate.csv'
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


# Create scattermapbox chart
@app.callback(Output('map_1', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')])
def update_graph(w_countries, w_countries1):
    select_years=[]
    terr3 = terr2.groupby(['region_txt', 'country_txt', 'provstate', 'city', 'iyear', 'latitude', 'longitude'])[
        ['nkill', 'nwound']].sum().reset_index()
    terr4 = terr3[(terr3['region_txt'] == w_countries) & (terr3['country_txt'] == w_countries1) & (
                terr3['iyear'] >= select_years[0]) & (terr3['iyear'] <= select_years[1])]

    if w_countries1:
        zoom = 3
        zoom_lat = list_locations[w_countries1]['latitude']
        zoom_lon = list_locations[w_countries1]['longitude']

    return {
        'data': [go.Scattermapbox(
            lon=terr4['longitude'],
            lat=terr4['latitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=terr4['nwound'],
                color=terr4['nwound'],
                colorscale='hsv',
                showscale=False,
                sizemode='area'),

            hoverinfo='text',
            hovertext=
            '<b>Region</b>: ' + terr4['region_txt'].astype(str) + '<br>' +
            '<b>Country</b>: ' + terr4['country_txt'].astype(str) + '<br>' +
            '<b>Province/State</b>: ' + terr4['provstate'].astype(str) + '<br>' +
            '<b>City</b>: ' + terr4['city'].astype(str) + '<br>' +
            '<b>Longitude</b>: ' + terr4['longitude'].astype(str) + '<br>' +
            '<b>Latitude</b>: ' + terr4['latitude'].astype(str) + '<br>' +
            '<b>Killed</b>: ' + [f'{x:,.0f}' for x in terr4['nkill']] + '<br>' +
            '<b>Wounded</b>: ' + [f'{x:,.0f}' for x in terr4['nwound']] + '<br>' +
            '<b>Year</b>: ' + terr4['iyear'].astype(str) + '<br>'

        )],

        'layout': go.Layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            hovermode='closest',
            mapbox=dict(
                accesstoken='pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',
                # Use mapbox token here
                center=go.layout.mapbox.Center(lat=zoom_lat, lon=zoom_lon),
                # style='open-street-map',
                style='dark',
                zoom=zoom
            ),
            autosize=True,

        )

    }


####################################
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


#
# get Data for Economy GDP
def getGDP(countryName):
    value = []
    value1 = []
    data2 = dataGDP.where(dataGDP.GEOLABEL == countryName).collect()
    for i in range(1, 19):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1


#
# get Data for Economy Monthly industrial production
def getMIN(countryName):
    value = []
    value1 = []
    data2 = dataMIN.where(dataMIN.GEOLABEL == countryName).collect()
    for i in range(1, 29):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1


#
# get Data for Economy Monthly volume
def getMV(countryName):
    value = []
    value1 = []
    data2 = dataMV.where(dataMV.GEOLABEL == countryName).collect()
    for i in range(1, 83):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1


#
# get Data for Economy Monthly production in contruction
def getMPIC(countryName):
    value = []
    value1 = []
    data2 = dataMPIC.where(dataMPIC.GEOLABEL == countryName).collect()
    for i in range(1, 29):
        value.append(float(data2[0][i]))
    value1 = np.array(value)
    return value1


#
######################################
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


#
###################
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
    # Data for line

    mon = np.array(terr2)
    mon1 = mon[5][1]

    coun = []
    coun = np.array(country_chosen)
    #
    # ECONOMY INFLATION
    #
    wctr1 = w_countries1
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

        valueEea1 = []
        if ('European Economic Area' in coun):
            valueEea1 = getArr('European Economic Area')

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

        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'

                                ),

                     go.Scatter(x=lab,
                                y=valueBel1,
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'

                                ),
                     go.Scatter(x=lab,
                                y=valueSer1,
                                mode='lines+markers',
                                name='RS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Serbia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueMal1,
                                mode='lines+markers',
                                name='MT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF00FF'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF00FF', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Malta' + '<br>'

                                ),
                     go.Scatter(x=lab,
                                y=valueEa1,
                                mode='lines+markers',
                                name='EA',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#472727'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#472727', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Euro area' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueBul1,
                                mode='lines+markers',
                                name='BG',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#353131'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#353131', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Bulgaria' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueChe1,
                                mode='lines+markers',
                                name='CZ',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#49AF30'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#49AF30', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Czechia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueDen1,
                                mode='lines+markers',
                                name='DK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2A4623'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2A4623', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Denmark' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueGer1,
                                mode='lines+markers',
                                name='DE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#7B7D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#7B7D7B', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Germany' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueEst1,
                                mode='lines+markers',
                                name='EE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#C4C048'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#C4C048', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Estonia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueIre1,
                                mode='lines+markers',
                                name='IE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#9E9D7B'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#9E9D7B', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Ireland' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueGree1,
                                mode='lines+markers',
                                name='EL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#1A46C0'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#1A46C0', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Greece' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueSpa1,
                                mode='lines+markers',
                                name='ES',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#2E063A'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#2E063A', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Spain' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueFra1,
                                mode='lines+markers',
                                name='FR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#39313C'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#39313C', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'France' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueCro1,
                                mode='lines+markers',
                                name='HR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#189F96'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#189F96', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Croatia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueIta1,
                                mode='lines+markers',
                                name='IT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#94A4A3'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#94A4A3', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Italy' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueCyp1,
                                mode='lines+markers',
                                name='CY',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3399', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Cyprus' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueLat1,
                                mode='lines+markers',
                                name='LV',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#aaaa55'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#aaaa55', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Latvia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueLith1,
                                mode='lines+markers',
                                name='LT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffff00'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffff00', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Lithuania' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueLux1,
                                mode='lines+markers',
                                name='LU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#007399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#007399', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Luxembourg' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueHun1,
                                mode='lines+markers',
                                name='HU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff3300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff3300', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Hungary' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueNet1,
                                mode='lines+markers',
                                name='NL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00e600', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Netherlands' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueAus1,
                                mode='lines+markers',
                                name='AT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dff4d'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dff4d', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Austria' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valuePol1,
                                mode='lines+markers',
                                name='PL',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#003300'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#003300', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Poland' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valuePor1,
                                mode='lines+markers',
                                name='PT',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#662900'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#662900', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Portugal' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueRom1,
                                mode='lines+markers',
                                name='RO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#993399'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#993399', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Romania' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueSlo1,
                                mode='lines+markers',
                                name='SI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#d98cd9'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#d98cd9', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovenia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueSlovak1,
                                mode='lines+markers',
                                name='SK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#0033cc'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#0033cc', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Slovakia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueFin1,
                                mode='lines+markers',
                                name='FI',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#99b3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#99b3ff', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Finland' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueSwe1,
                                mode='lines+markers',
                                name='SE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#001a66'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#001a66', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Sweden' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueUk1,
                                mode='lines+markers',
                                name='UK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#00cc99'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#00cc99', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United Kingdom' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueIce1,
                                mode='lines+markers',
                                name='IS',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#336600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#336600', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Iceland' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueNor1,
                                mode='lines+markers',
                                name='NO',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#73e600'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#73e600', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Norway' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueSwi1,
                                mode='lines+markers',
                                name='CH',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#bfff80'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#bfff80', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Switzerland' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueMake1,
                                mode='lines+markers',
                                name='MK',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ffa366'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ffa366', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'North Macedonia' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueTur1,
                                mode='lines+markers',
                                name='TR',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#ff4da6'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#ff4da6', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Turkey' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueUs1,
                                mode='lines+markers',
                                name='US',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#4dc3ff'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#4dc3ff', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'United States' + '<br>'
                                ),

                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    #
    #Kraj ECONOMY INFLATION
    #############################
    #Pocinje ECONOMY GDP
    #
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

        valueEea1 = []
        if ('European Economic Area' in coun):
            valueEea1 = getGDP('European Economic Area')

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

        return {
            'data': [go.Scatter(x=labGDP,
                                y=valueEu1,
                                mode='lines+markers',
                                name='EU',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labGDP,
                                y=valueBel1,
                                mode='lines+markers',
                                name='BE',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }


    #
    # ECONOMY GDP
    #
    #
    # ECONOMY MIN
    #
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

        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    #
    # ECONOMY MIN
    #
    # ECONOMY MV
    #
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

        return {
            'data': [go.Scatter(x=labMV,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labMV,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    #
    # End ECONOMY MV
    #
    # Strart Economy-Monthly production in construction
    #
    elif (w_countries == 'Economy') and (w_countries1 == 'Monthly production in construction'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getMPIC('European Union')

        valueMal1 = []
        if ('Malta' in coun):
            valueMal1 = getMPIC('Malta')

        # valueSer1=[]
        # if ('Serbia' in coun):
        #    valueSer1=getMPIC('Serbia')

        valueEa1 = []
        if ('Euro area' in coun):
            valueEa1 = getMPIC('Euro area')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getMPIC('Belgium')

        return {
            'data': [go.Scatter(x=lab,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=lab,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    #
    # END Economy-Monthly production in construction
    #
    # END of ECONOMY
    #
    #
    # Start POPULATION AND HEALTH
    #
    #
    # POPULATION AND HEALTH-Monthly excess mortality
    #
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):

        # valueMal1=[]
        # if ('Malta' in coun):
        #   valueMal1=getPAHMEM('Malta')

        # valueSer1=[]
        # if ('Serbia' in coun):
        #    valueSer1=getMPIC('Serbia')

        # valueEa1=[]
        # if ('Euro area' in coun):
        #   valueEa1=getPAHMEM('Euro area')
        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getPAHMEM('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getPAHMEM('Belgium')

        return {
            'data': [go.Scatter(x=labPAHMEM,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labPAHMEM,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }


    #
    # Zavrsava se POPULATION AND HEALTH-Monthly excess mortality
    #
    #Pocinje Number of deaths by week
    elif (w_countries == 'Population and health') &  (w_countries1 == 'Number of deaths by week'):

            valueBul1 = []
            if ('Bulgaria' in coun):
                valueBul1 = getPAHDBW('Bulgaria')

            valueBel1 = []
            if ('Belgium' in coun):
                valueBel1 = getPAHDBW('Belgium')

            labelDbw = getLabelPAHDBW2()

            return {
                'data': [go.Scatter(x=labelDbw,
                                    y=valueBul1,
                                    mode='lines+markers',
                                    name='Bulgarua',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#E6D1D1', width=2)
                                                ),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Bulgaria' + '<br>'
                                    ),
                         go.Scatter(x=labelDbw,
                                    y=valueBel1,
                                    mode='lines+markers',
                                    name='Belgium',
                                    line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                    marker=dict(size=5, symbol='circle', color='lightblue',
                                                line=dict(color='#FF0000', width=2)
                                                ),
                                    hoverinfo='text',
                                    hovertext=
                                    '<b>Country</b>: ' + 'Belgium' + '<br>'
                                    )
                         ],

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
                                   color='white'
                               )

                               ),

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
                                   color='white'
                               )

                               ),

                    legend={
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font=dict(
                        family="sans-serif",
                        size=12,
                        color='white'),

                )

            }

    ##################################
    ##################################Pocinje Polulation and healht-Monthly first-time asylum applicants

    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly first-time asylum applicants'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getPAHMFTA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getPAHMFTA('Belgium')

        labelDbw = getLabelPAHMFTA2()

        return {
            'data': [go.Scatter(x=labelPAHMFTA2,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelPAHMFTA2,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
#####################################################
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

        labelDbw = getLabelSAWMUR2()

        return {
            'data': [go.Scatter(x=labelSAWMUR,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWMUR,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly youth unemployment rate'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWMYUR('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWMYUR('Belgium')



        return {
            'data': [go.Scatter(x=labelSAWMYUR,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWMYUR,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    #KRece podgrupa: Society and work-Quarterly employment
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly employment'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQE('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQE('Belgium')



        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWQE,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    #Krece podgrupa Society and work-Quarterly labour market slack
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour market slack'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQLMS('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQLMS('Belgium')



        return {
            'data': [go.Scatter(x=labelSAWQLMS,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWQLMS,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    #KRece podgrupa Society and work-Quarterly job vacancy rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly job vacancy rate'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQJVR('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQJVR('Belgium')



        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWQE,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

        # KRece podgrupa Society and work-Quarterly labour cost
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour cost'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getSAWQLC('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getSAWQLC('Belgium')

        return {
            'data': [go.Scatter(x=labelSAWQE,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelSAWQE,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
#######################################################
    ###################################################
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

        return {
            'data': [go.Scatter(x=labelsAETTMAPT,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelsAETTMAPT,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    # Krece oblast Agriculture, energy, transport & tourism-Monthly commercial air flights
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly commercial air flights'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMCAF('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMCAF('Belgium')

        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelsAETTMCAF,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly arrivals at tourist accommodation'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMATA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMATA('Belgium')

        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelsAETTMCAF,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                               color='white'
                           )

                           ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    # Krece oblast Agriculture, energy, transport & tourism-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly nights spent at tourist accommodation'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMNSTA('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMNSTA('Belgium')

        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelsAETTMCAF,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                        color='white'
                    )

                    ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }
    # Krece oblast Agriculture, energy, transport & tourism-Monthly electricity consumed by end-users
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly electricity consumed by end-users'):

        valueEu1 = []
        if ('European Union' in coun):
            valueEu1 = getAETTMEC('European Union')

        valueBel1 = []
        if ('Belgium' in coun):
            valueBel1 = getAETTMEC('Belgium')

        return {
            'data': [go.Scatter(x=labelsAETTMCAF,
                                y=valueEu1,
                                mode='lines+markers',
                                name='European Union',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#E6D1D1'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#E6D1D1', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'European Union' + '<br>'
                                ),
                     go.Scatter(x=labelsAETTMCAF,
                                y=valueBel1,
                                mode='lines+markers',
                                name='Belgium',
                                line=dict(shape="spline", smoothing=1.3, width=3, color='#FF0000'),
                                marker=dict(size=5, symbol='circle', color='lightblue',
                                            line=dict(color='#FF0000', width=2)
                                            ),
                                hoverinfo='text',
                                hovertext=
                                '<b>Country</b>: ' + 'Belgium' + '<br>'
                                )
                     ],

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
                        color='white'
                    )

                ),

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
                               color='white'
                           )

                           ),

                legend={
                    'orientation': 'h',
                    'bgcolor': '#010915',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                font=dict(
                    family="sans-serif",
                    size=12,
                    color='white'),

            )

        }

    else:
        return dash.no_update

#################################################
#KRAJ LINE CHART GRAFIKA
#################################################
#################################################

#################################################
#################################################
# Kreiranje pie chart grafika
#################################################
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
##############################################
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

    coun = []
    coun = np.array(country_chosen)
    colors = ['#FF00FF', '#9C0C38', 'orange', 'lightblue']

    #Pocinje deo Economy- Inflation
    if (w_countries == 'Economy') & (w_countries1 == 'Inflation - annual growth rate'):
        valueEu1=0
        if('European Union' in coun):
           valueEu1 = getPieArr('European Union')

        valueBel1=0
        if('Belgium' in coun):
           valueBel1=getPieArr('Belgium')

        valueSer1 = 0
        if ('Serbia' in coun):
            valueSer1 = getPieArr('Serbia')

        return {
            'data': [go.Pie(labels=['EU','BE','SR'],
                        values=[valueEu1, valueBel1,valueSer1],
                        marker=dict(colors=colors),
                        hoverinfo='label+value+percent',
                        textinfo='label+value',
                        textfont=dict(size=13)

                        )],

            'layout': go.Layout(
                    plot_bgcolor='#010915',
                    paper_bgcolor='#010915',
                    hovermode='closest',
                    title={
                        'text': 'Economy: ' +(w_countries1) +'</br>',

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
    ##
    # Pocinje ECONOMY GDP
    ##
    elif (w_countries == 'Economy') & (w_countries1 == 'GDP – quarterly growth rate'):

        valueEu1=0
        if('European Union' in coun):
            valueEu1=getPieGDP('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieGDP('Belgium')

        valueSer1 = 0
        if ('Serbia' in coun):
            valueSer1 = getPieGDP('Serbia')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR'],
                            values=[valueEu1, valueBel1, valueSer1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Economy: ' + (w_countries1) + '</br>',

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
    #
    # ECONOMY Monthly industrial production
    #
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly industrial production'):
        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMIP('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMIP('Belgium')

        valueSer1 = 0
        if ('Serbia' in coun):
            valueSer1 = getPieMIP('Serbia')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR'],
                            values=[valueEu1, valueBel1, valueSer1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Economy: ' + (w_countries1) + '</br>',

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
    # ECONOMY MV
    #
    elif (w_countries == 'Economy') & (w_countries1 == 'Monthly volume of retail trade'):
        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMV('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMV('Belgium')

        valueSer1 = 0
        if ('Serbia' in coun):
            valueSer1 = getPieMV('Serbia')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'SR'],
                            values=[valueEu1, valueBel1, valueSer1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Economy: ' + (w_countries1) + '</br>',

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
    # Pocinje Economy-Monthly production in construction
    #
    elif (w_countries == 'Economy') and (w_countries1 == 'Monthly production in construction'):
        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMPIC('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMPIC('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMPIC('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Economy: ' + (w_countries1) + '</br>',

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
    #######################################
###############################################
###############################################
    #Pocinje deo Population and health
    #########################################
    ###########################################
    # POPULATION AND HEALTH-Monthly excess mortality
    #
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly excess mortality'):

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieNOD('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieNOD('Bulgaria')

        return {
            'data': [go.Pie(labels=['BE', 'BG'],
                            values=[valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Population: ' + (w_countries1) + '</br>',

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
    #
    # Pocinje Number of deaths by week
    elif (w_countries == 'Population and health') & (w_countries1 == 'Number of deaths by week'):

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieNOD('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieNOD('Bulgaria')

        return {
            'data': [go.Pie(labels=['BE', 'BG'],
                            values=[valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Population: ' + (w_countries1) + '</br>',

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
    #####################Pocinje Polulation and healht-Monthly first-time asylum applicants
    elif (w_countries == 'Population and health') & (w_countries1 == 'Monthly first-time asylum applicants'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMFTA('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMFTA('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMFTA('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Population: ' + (w_countries1) + '</br>',

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

    #####################Pocinje Society and work-Monthly unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly unemployment rate'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMUR('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMUR('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMUR('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #####################Society and work-Monthly youth unemployment rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Monthly youth unemployment rate'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMYUR('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMYUR('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMYUR('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #####################Society and work-Quarterly employment
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly employment'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieQE('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieQE('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieQE('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #####################Society and work-Quarterly labour market slack
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour market slack'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieQLMS('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieQLMS('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieQLMS('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #####################Society and work-Quarterly job vacancy rate
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly job vacancy rate'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieQJVR('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieQJVR('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieQJVR('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #####################Society and work-Quarterly labour cost
    elif (w_countries == 'Society and work') & (w_countries1 == 'Quarterly labour cost'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieQLC('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieQLC('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieQLC('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Society: Average ' + (w_countries1) + '</br>',

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
    #KRece oblast AGRICULTURE, ENERGY, TRANSPORT & TOURISM
    ######################################################
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly air passenger transport
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly air passenger transport'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMAPT('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMAPT('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMAPT('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Agriculture: Average ' + (w_countries1) + '</br>',

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
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly commercial air flights
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly commercial air flights'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMCAF('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMCAF('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMCAF('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Agriculture: Average ' + (w_countries1) + '</br>',

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
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly arrivals at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly arrivals at tourist accommodation'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMATA('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMATA('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMATA('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Agriculture: Average ' + (w_countries1) + '</br>',

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
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly nights spent at tourist accommodation
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly nights spent at tourist accommodation'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMNSTA('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMNSTA('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMNSTA('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Agriculture: Average ' + (w_countries1) + '</br>',

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
    ######################AGRICULTURE, ENERGY, TRANSPORT & TOURISM-Monthly electricity consumed by end-users
    elif (w_countries == 'Agriculture, energy, transport & tourism') & (w_countries1 == 'Monthly electricity consumed by end-users'):

        valueEu1 = 0
        if ('European Union' in coun):
            valueEu1 = getPieMEC('European Union')

        valueBel1 = 0
        if ('Belgium' in coun):
            valueBel1 = getPieMEC('Belgium')

        valueBul1 = 0
        if ('Belgium' in coun):
            valueBul1 = getPieMEC('Bulgaria')

        return {
            'data': [go.Pie(labels=['EU', 'BE', 'BG'],
                            values=[valueEu1, valueBel1, valueBul1],
                            marker=dict(colors=colors),
                            hoverinfo='label+value+percent',
                            textinfo='label+value',
                            textfont=dict(size=13)

                            )],

            'layout': go.Layout(
                plot_bgcolor='#010915',
                paper_bgcolor='#010915',
                hovermode='closest',
                title={
                    'text': 'Agriculture: Average ' + (w_countries1) + '</br>',

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

    else:
        return dash.no_update



if __name__ == '__main__':
    app.run_server(debug=True)
