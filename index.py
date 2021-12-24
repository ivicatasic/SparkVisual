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
app.title='Spark Visual Data'

terr2 = pd.read_csv('C:/Users/Korisnik/Desktop/SparkVisual/data/products.csv')
location1 = terr2[['subproducts']]
list_locations = location1.set_index('subproducts').T.to_dict('dict')
region = terr2['products'].unique()

conf = SparkConf().setAppName('Spark visual')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#upload Economy inflation file
csv_EI = 'C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-inflation.csv' 
data = sqlContext.read.format("csv").options(header='true').load(csv_EI) 

country=pd.read_csv(csv_EI)
countries=country['GEOLABEL'].unique()

#upload Economy GDP file
csv_EGDP='C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-GDP.csv'
dataGDP=sqlContext.read.format("csv").options(header='true').load(csv_EGDP)

#upload Economy Monthly industrial production
csv_EMIN='C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly industrial production.csv'
dataMIN=sqlContext.read.format("csv").options(header='true').load(csv_EMIN)

#upload Economy montly volume
csv_EMV='C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly volume.csv'
dataMV=sqlContext.read.format("csv").options(header='true').load(csv_EMV)

#upload Economy montly production in construction
csv_EMPIC='C:/Users/Korisnik/Desktop/SparkVisual/data/Economy-Monthly production in construction.csv'
dataMPIC=sqlContext.read.format("csv").options(header='true').load(csv_EMPIC)

####################################
####################################
# Upload Population and Health
csv_PAHMEM='C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Monthly excess mortality.csv'
dataPAHMEM=sqlContext.read.format("csv").options(header='true').load(csv_PAHMEM)

#Ucitavanje fajla: Population and health-Number of deaths by week
csv_PAHDBW='C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Number of deaths by week.csv'
dataPAHDBW=sqlContext.read.format("csv").options(header='true').load(csv_PAHDBW)




############################################
#labels for Economy inflation
labels=["6-2019","7-2019","8-2019","9-2019","10-2019","11-2019","12-2019","1-2020","2-2020","3-2020",
        "4-2020","5-2020","6-2020","7-2020","8-2020","9-2020","10-2020","11-2020","12-2020","1-2021",
        "2-2021","3-2021","4-2021","5-2021","6-2021","7-2021","8-2021","9-2021","10-2021","11-2021",]
lab=np.array(labels)

#labels for Economy GDP
labelsGDP=["Q1-2017","Q2-2017","Q3-2017","Q4-2017","Q1-2018","Q2-2018","Q3-2018","Q4-2018","Q1-2019",
           "Q2-2019","Q3-2019","Q4-2019","Q1-2020","Q2-2020","Q3-2020","Q4-2020","Q1-2021","Q2-2021",
           "Q3-2021","Q4-2021"]
labGDP=np.array(labelsGDP)

#labels for Economy Montly volume
labelsMV=["1-2015","2-2015","3-2015","4-2015","5-2015","6-2015","7-2015","8-2015","9-2015","10-2015",
        "11-2015","12-2015","1-2016","2-2016","3-2016","4-2016","5-2016","6-2016","7-2016","8-2016",
        "9-2016","10-2016","11-2016","12-2016","1-2017","2-2017","3-2017","4-2017","5-2017","6-2017",
        "7-2017","8-2017","9-2017","10-2017","11-2017","12-2017","1-2018","2-2018","3-2018","4-2018",
        "5-2018", "6-2018","7-2018","8-2018","9-2018","10-2018","11-2018","12-2018","1-2019","2-2019",
        "3-2019","4-2019","5-2019",
        "6-2019","7-2019","8-2019","9-2019","10-2019","11-2019","12-2019","1-2020","2-2020","3-2020",
        "4-2020","5-2020","6-2020","7-2020","8-2020","9-2020","10-2020","11-2020","12-2020","1-2021",
        "2-2021","3-2021","4-2021","5-2021","6-2021","7-2021","8-2021","9-2021","10-2021","11-2021",]
labMV=np.array(labelsMV)
#######################################
#######################################
#LABELS FOR POPULATION AND HEALTH

#labels population and health-montly excess mortality
labelsPAHMEM=["1-2020","2-2020","3-2020",
        "4-2020","5-2020","6-2020","7-2020","8-2020","9-2020","10-2020","11-2020","12-2020","1-2021",
        "2-2021","3-2021","4-2021","5-2021","6-2021","7-2021","8-2021","9-2021","10-2021"]
labPAHMEM=np.array(labelsPAHMEM)

#labela za population and health-Number of deaths by week
csv_PAHDBW2='C:/Users/Korisnik/Desktop/SparkVisual/data/Population and health-Number of deaths by week.csv'
dataPAHDBW2=sqlContext.read.format("csv").options(header='false').load(csv_PAHDBW)
def getLabelPAHDBW2():
    value=[]
    value1=[]
    data2=dataPAHDBW.first()
    for i in range(1,104):
        value.append(float(data[i]))
    value1= np.array(value)
    return value1


################################################
app.layout = html.Div([

    html.Div([
        html.Div([
            html.Div([
                html.H3('Spark Visual app', style = {"margin-bottom": "0px", 'color': 'white'}),
               

            ]),
        ], className = "six column", id = "title")

    ], id = "header", className = "row flex-display", style = {"margin-bottom": "25px"}),

    html.Div([
        html.Div([
            dcc.Graph(id = 'map_1',
                      config = {'displayModeBar': 'hover'}),

        ], className = "create_container 12 columns"),

    ], className = "row flex-display"),

    html.Div([
        html.Div([
            html.P('Izaberi kategoriju:', className = 'fix_label', style = {'color': 'white'}),
            dcc.Dropdown(id = 'w_countries',
                         multi = False,
                         clearable = True,
                         disabled = False,
                         style = {'display': True},
                         value = 'Economy',
                         placeholder = 'Select category',
                         options = [{'label': c, 'value': c}
                                    for c in region], className = 'dcc_compon'),

            html.P('Izaberi podkategoriju:', className = 'fix_label', style = {'color': 'white'}),
            dcc.Dropdown(id = 'w_countries1',
                         multi = False,
                         clearable = True,
                         disabled = False,
                         style = {'display': True},
                         placeholder = 'Select subcategory',
                         options = [], className = 'dcc_compon'),
            
            html.P('Izaberi zemlju:', className = 'fix_label', style = {'color': 'white'}),
            dcc.Dropdown(id = 'w_countries2',
                         multi = True,
                         clearable = True,
                         disabled = False,
                         style = {'display': True},
                         value = ['European Union','Belgium'],
                         placeholder = 'Select country',
                         options = [{'label': c, 'value': c}
                                    for c in countries], className = 'dcc_compon'),

            html.P('Izaberi godinu:', className = 'fix_label', style = {'color': 'white', 'margin-left': '1%'}),
            dcc.RangeSlider(id = 'select_years',
                            min = 2019,
                            max = 2021,
                            dots = False,
                            value = [2020, 2021]),

        ], className = "create_container three columns"),

        
 html.Div([
       
        dcc.Graph(id='bar_line_1', figure={}, clickData=None, hoverData=None, # I assigned None for tutorial purposes. By defualt, these are None, unless you specify otherwise.
                  config={
                      'staticPlot': False,     # True, False
                      'scrollZoom': True,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': False,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                        })
    ]),

        html.Div([
            dcc.Graph(id = 'pie',
                      config = {'displayModeBar': 'hover'}),

        ], className = "create_container three columns"),

    ], className = "row flex-display"),

], id = "mainContainer", style = {"display": "flex", "flex-direction": "column"})

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
              [Input('w_countries1', 'value')],
              [Input('select_years', 'value')])
def update_graph(w_countries, w_countries1, select_years):
    terr3 = terr2.groupby(['region_txt', 'country_txt', 'provstate', 'city', 'iyear', 'latitude', 'longitude'])[['nkill', 'nwound']].sum().reset_index()
    terr4 = terr3[(terr3['region_txt'] == w_countries) & (terr3['country_txt'] == w_countries1) & (terr3['iyear'] >= select_years[0]) & (terr3['iyear'] <= select_years[1])]

    if w_countries1:
        zoom = 3
        zoom_lat = list_locations[w_countries1]['latitude']
        zoom_lon = list_locations[w_countries1]['longitude']


    return {
        'data': [go.Scattermapbox(
            lon = terr4['longitude'],
            lat = terr4['latitude'],
            mode = 'markers',
            marker = go.scattermapbox.Marker(
                size = terr4['nwound'],
                color = terr4['nwound'],
                colorscale = 'hsv',
                showscale = False,
                sizemode = 'area'),

            hoverinfo = 'text',
            hovertext =
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
            margin = {"r": 0, "t": 0, "l": 0, "b": 0},
            hovermode = 'closest',
            mapbox = dict(
                accesstoken = 'pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw',  # Use mapbox token here
                center = go.layout.mapbox.Center(lat = zoom_lat, lon = zoom_lon),
                # style='open-street-map',
                style = 'dark',
                zoom = zoom
            ),
            autosize = True,

        )

    }

####################################
############# LINE CHART ###########
####################################
#get Data for Economy Inflation
def getArr(countryName):
    value=[]
    value1=[]
    data2=data.where(data.GEOLABEL==countryName).collect()
    for i in range(2,31):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#
#get Data for Economy GDP
def getGDP(countryName):
    value=[]
    value1=[]
    data2=dataGDP.where(dataGDP.GEOLABEL==countryName).collect()
    for i in range(1,19):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#
#get Data for Economy Monthly industrial production
def getMIN(countryName):
    value=[]
    value1=[]
    data2=dataMIN.where(dataMIN.GEOLABEL==countryName).collect()
    for i in range(1,29):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#
#get Data for Economy Monthly volume
def getMV(countryName):
    value=[]
    value1=[]
    data2=dataMV.where(dataMV.GEOLABEL==countryName).collect()
    for i in range(1,83):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#
#get Data for Economy Monthly production in contruction
def getMPIC(countryName):
    value=[]
    value1=[]
    data2=dataMPIC.where(dataMPIC.GEOLABEL==countryName).collect()
    for i in range(1,29):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#
######################################
######################################
#Get data for population and health
def getPAHMEM(countryName):
    value=[]
    value1=[]
    data2=dataPAHMEM.where(dataPAHMEM.GEOLABEL==countryName).collect()
    for i in range(1,22):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1
#

def getPAHDBW(countryName):
    value=[]
    value1=[]
    data2=dataPAHDBW.where(dataPAHDBW.GEOLABEL==countryName).collect()
    for i in range(1,104):
        value.append(float(data2[0][i]))
    value1= np.array(value)
    return value1


# Create line  chart 
@app.callback(Output('bar_line_1', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')],
              [Input('w_countries2', 'value')],
              [Input('select_years', 'value')])
def update_graph(w_countries, w_countries1,country_chosen, select_years):
    # Data for line
    
        coun=[]
        coun=np.array(country_chosen)
        #
        # ECONOMY INFLATION
        #
        if (w_countries=='Economy') & (w_countries1=='Inflation - annual growth rate'):
            valueEu1=[]
            if ('European Union' in coun):
                valueEu1=getArr('European Union')
            
            valueMal1=[]
            if ('Malta' in coun):
                valueMal1=getArr('Malta')
            
            valueSer1=[]
            if ('Serbia' in coun):
                valueSer1=getArr('Serbia')
            
            valueEa1=[]
            if ('Euro area' in coun):
                valueEa1=getArr('Euro area')
            
            valueBel1=[]
            if ('Belgium' in coun):
                valueBel1=getArr('Belgium')
        
            valueBul1=[]
            if ('Bulgaria' in coun):
                valueBul1=getArr('Bulgaria')
            
            valueChe1=[]
            if ('Czechia' in coun):
                valueChe1=getArr('Czechia')
            
            valueDen1=[]
            if ('Denmark' in coun):
                valueDen1=getArr('Denmark')
         
            valueGer1=[]
            if ('Germany' in coun):
                valueGer1=getArr('Germany')
         
            valueEst1=[]
            if ('Estonia' in coun):
                valueEst1=getArr('Estonia')
         
            valueIre1=[]
            if ('Ireland' in coun):
                valueIre1=getArr('Ireland')
        
            valueGree1=[]
            if ('Greece' in coun):
                valueGree1=getArr('Greece')
        
            valueSpa1=[]
            if ('Spain' in coun):
                valueSpa1=getArr('Spain')
        
            valueFra1=[]
            if ('France' in coun):
                valueFra1=getArr('France')
            
            valueCro1=[]
            if ('Croatia' in coun):
                valueCro1=getArr('Croatia')
        
            valueIta1=[]
            if ('Italy' in coun):
                valueIta1=getArr('Italy')
        
            valueCyp1=[]
            if ('Cyprus' in coun):
                valueCyp1=getArr('Cyprus')
        
            valueLat1=[]
            if ('Latvia' in coun):
                valueLat1=getArr('Latvia')
        
            valueLith1=[]
            if ('Lithuania' in coun):
                valueLith1=getArr('Lithuania')
        
            valueLux1=[]
            if ('Luxembourg' in coun):
                valueLux1=getArr('Luxembourg')
        
            valueHun1=[]
            if ('Hungary' in coun):
                valueHun1=getArr('Hungary')
        
            valueNet1=[]
            if ('Netherlands' in coun):
                valueNet1=getArr('Netherlands')
                    
            valueAus1=[]
            if ('Austria' in coun):
                valueAus1=getArr('Austria')
        
            valuePol1=[]
            if ('Poland' in coun):
                valuePol1=getArr('Poland')
            
            valuePor1=[]
            if ('Portugal' in coun):
                valuePor1=getArr('Portugal')
        
            valueRom1=[]
            if ('Romania' in coun):
                valueRom1=getArr('Romania')
        
            valueSlo1=[]
            if ('Slovenia' in coun):
                valueSlo1=getArr('Slovenia')
        
            valueSlovak1=[]
            if ('Slovakia' in coun):
                valueSlovak1=getArr('Slovakia')
            
            valueFin1=[]
            if ('Finland' in coun):
                valueFin1=getArr('Finland')
        
            valueSwe1=[]
            if ('Sweden' in coun):
                valueSwe1=getArr('Sweden')
        
            valueUk1=[]
            if ('United Kingdom' in coun):
                valueUk1=getArr('United Kingdom')
            
            valueEea1=[]
            if ('European Economic Area' in coun):
                valueEea1=getArr('European Economic Area')
        
            valueIce1=[]
            if ('Iceland' in coun):
                valueIce1=getArr('Iceland')
        
            valueNor1=[]
            if ('Norway' in coun):
                valueNor1=getArr('Norway')
        
            valueSwi1=[]
            if ('Switzerland' in coun):
                valueSwi1=getArr('Switzerland')
        
            valueMake1=[]
            if ('North Macedonia' in coun):
                valueMake1=getArr('North Macedonia')
        
            valueTur1=[]
            if ('Turkey' in coun):
                valueTur1=getArr('Turkey')
        
            valueUs1=[]
            if ('United States' in coun):
                valueUs1=getArr('United States')
        
            return {
                'data': [go.Scatter(x =lab,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    
                                    
                                    ),
                
                           go.Scatter(x =lab,
                                y = valueBel1,
                                mode = 'lines+markers',
                                name = 'Belgium',
                                line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF00FF'),
                                marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                              line = dict(color = '#FF00FF', width = 2)
                                              ),
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Country</b>: ' + 'Belgium'+'<br>'
                                
                                ),
                           go.Scatter(x =lab,
                                y = valueSer1,
                                mode = 'lines+markers',
                                name = 'Serbia',
                                line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                line = dict(color = '#FF0000', width = 2)
                                                       ),
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Country</b>: ' + 'Serbia'+'<br>'
                                )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'Inflation - annual growth rate'+ '  ' + '<br>' + 
                            "(change compared with same month of previous year)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Year</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>%</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                )

            }
        #
        #ECONOMY INFLATION
        #
        #ECONOMY GDP
        #
        elif (w_countries=='Economy') & (w_countries1=='GDP – quarterly growth rate'):
            
            valueEu1=[]
            if ('European Union' in coun):
                valueEu1=getGDP('European Union')
            
            valueMal1=[]
            if ('Malta' in coun):
                valueMal1=getGDP('Malta')
            
            valueSer1=[]
            if ('Serbia' in coun):
                valueSer1=getGDP('Serbia')
            
            valueEa1=[]
            if ('Euro area' in coun):
                valueEa1=getGDP('Euro area')
            
            valueBel1=[]
            if ('Belgium' in coun):
                valueBel1=getGDP('Belgium')
            
            
            return {
                'data': [go.Scatter(x =labGDP,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    ),
                         go.Scatter(x =labGDP,
                                    y = valueBel1,
                                    mode = 'lines+markers',
                                    name = 'Belgium',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                    line = dict(color = '#FF0000', width = 2)
                                                           ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'Belgium'+'<br>'
                                    )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'GDP – quarterly growth rate'+ '  ' + '<br>' + 
                            "(% change compared with previous quarter)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Quartal</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>%</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                )

            }
        #
        #ECONOMY GDP
        #
        #
        #ECONOMY MIN
        #
        elif (w_countries=='Economy') & (w_countries1=='Monthly industrial production'):
            
            valueEu1=[]
            if ('European Union' in coun):
                valueEu1=getMIN('European Union')
            
            valueMal1=[]
            if ('Malta' in coun):
                valueMal1=getMIN('Malta')
            
            valueSer1=[]
            if ('Serbia' in coun):
                valueSer1=getMIN('Serbia')
            
            valueEa1=[]
            if ('Euro area' in coun):
                valueEa1=getMIN('Euro area')
            
            valueBel1=[]
            if ('Belgium' in coun):
                valueBel1=getMIN('Belgium')
            
            
            return {
                'data': [go.Scatter(x =lab,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    ),
                         go.Scatter(x =lab,
                                    y = valueBel1,
                                    mode = 'lines+markers',
                                    name = 'Belgium',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                    line = dict(color = '#FF0000', width = 2)
                                                           ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'Belgium'+'<br>'
                                    )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'Monthly industrial production'+ '  ' + '<br>' + 
                            "(Index 2015=100)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Quartal</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>%</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                )

            }
        #
        #ECONOMY MIN
        #
        #ECONOMY MV
        #
        elif (w_countries=='Economy') & (w_countries1=='Monthly volume of retail trade'):
            
            valueEu1=[]
            if ('European Union' in coun):
                valueEu1=getMV('European Union')
            
            valueMal1=[]
            if ('Malta' in coun):
                valueMal1=getMV('Malta')
            
            valueSer1=[]
            if ('Serbia' in coun):
                valueSer1=getMV('Serbia')
            
            valueEa1=[]
            if ('Euro area' in coun):
                valueEa1=getMV('Euro area')
            
            valueBel1=[]
            if ('Belgium' in coun):
                valueBel1=getMV('Belgium')
            
            
            return {
                'data': [go.Scatter(x =labMV,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    ),
                         go.Scatter(x =labMV,
                                    y = valueBel1,
                                    mode = 'lines+markers',
                                    name = 'Belgium',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                    line = dict(color = '#FF0000', width = 2)
                                                           ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'Belgium'+'<br>'
                                    )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'Monthly volume of retail trade'+ '  ' + '<br>' + 
                            "(Index 2015=100)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Year</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>Index 2015=100</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                )

            }
        #
        #End ECONOMY MV
        #
        #Strart Economy-Monthly production in construction
        #
        elif (w_countries=='Economy') & (w_countries1=='Monthly production in construction'):
            
            valueEu1=[]
            if ('European Union' in coun):
                valueEu1=getMPIC('European Union')
            
            valueMal1=[]
            if ('Malta' in coun):
                valueMal1=getMPIC('Malta')
            
            #valueSer1=[]
            #if ('Serbia' in coun):
            #    valueSer1=getMPIC('Serbia')
            
            valueEa1=[]
            if ('Euro area' in coun):
                valueEa1=getMPIC('Euro area')
            
            valueBel1=[]
            if ('Belgium' in coun):
                valueBel1=getMPIC('Belgium')
            
            
            
            return {
                'data': [go.Scatter(x =lab,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    ),
                         go.Scatter(x =lab,
                                    y = valueBel1,
                                    mode = 'lines+markers',
                                    name = 'Belgium',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                    line = dict(color = '#FF0000', width = 2)
                                                           ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'Belgium'+'<br>'
                                    )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'Monthly production in construction'+ '  ' + '<br>' + 
                            "(Index 2015=100)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Year</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>Index 2015=100</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                )

            }
        #
        #END Economy-Monthly production in construction
        #
        #END of ECONOMY
        #
        #
        #Start POPULATION AND HEALTH
        #
        #
        #POPULATION AND HEALTH-Monthly excess mortality
        #
        elif (w_countries=='Population and health') & (w_countries1=='Monthly excess mortality'):
            
           
            #valueMal1=[]
            #if ('Malta' in coun):
            #   valueMal1=getPAHMEM('Malta')
            
            #valueSer1=[]
            #if ('Serbia' in coun):
            #    valueSer1=getMPIC('Serbia')
            
            #valueEa1=[]
            #if ('Euro area' in coun):
            #   valueEa1=getPAHMEM('Euro area')
                valueEu1=[]
                if ('European Union' in coun):
                    valueEu1=getPAHMEM('European Union')    
            
                valueBel1=[]
                if ('Belgium' in coun):
                    valueBel1=getPAHMEM('Belgium')
            
                str1=w_countries1
                return {
                        'data': [go.Scatter(x =labPAHMEM,
                                    y = valueEu1,
                                    mode = 'lines+markers',
                                    name = 'European Union',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#E6D1D1'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                                  line = dict(color = '#E6D1D1', width = 2)
                                                  ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + 'European Union'+'<br>'
                                    ),
                         go.Scatter(x =labPAHMEM,
                                    y = valueBel1,
                                    mode = 'lines+markers',
                                    name = 'Belgium',
                                    line = dict(shape = "spline", smoothing = 1.3, width = 3, color = '#FF0000'),
                                    marker = dict(size = 5, symbol = 'circle', color = 'lightblue',
                                    line = dict(color = '#FF0000', width = 2)
                                                           ),
                                    hoverinfo = 'text',
                                    hovertext =
                                    '<b>Country</b>: ' + str(str1)+'<br>'
                                    )
                         ],

                'layout': go.Layout(
                    barmode = 'stack',
                    plot_bgcolor = '#808080',
                    paper_bgcolor = '#A8A8A8',
                    title = {
                        'text': 'Monthly excess mortality'+ '  ' + '<br>' + 
                            "(% of additional deaths compared with average monthly deaths in 2016-2019)" + '</br>',

                        'y': 0.93,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    titlefont = {
                        'color': 'white',
                        'size': 20},

                    hovermode = 'closest',
                    showlegend = True,

                    xaxis = dict(title = '<b>Year</b>',
                                 spikemode  = 'toaxis+across',
                                 spikedash = 'solid',
                                 tick0 = 0,
                                 dtick = 1,
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    yaxis = dict(title = '<b>% of additional deaths</b>',
                                 color = 'white',
                                 showline = True,
                                 showgrid = True,
                                 showticklabels = True,
                                 linecolor = 'white',
                                 linewidth = 2,
                                 ticks = 'outside',
                                 tickfont = dict(
                                     family = 'Arial',
                                     size = 12,
                                     color = 'white'
                                 )

                                 ),

                    legend = {
                        'orientation': 'h',
                        'bgcolor': '#010915',
                        'xanchor': 'center', 'x': 0.5, 'y': -0.3},
                    font = dict(
                        family = "sans-serif",
                        size = 12,
                        color = 'white'),

                   )

               }
            
            #
            # END POPULATION AND HEALTH-Monthly excess mortality
            
              
                    
    




    
# Create pie chart (total casualties)
@app.callback(Output('pie', 'figure'),
              [Input('w_countries', 'value')],
              [Input('w_countries1', 'value')],
              [Input('select_years', 'value')])
def display_content(w_countries, w_countries1, select_years):
    terr9 = terr2.groupby(['region_txt', 'country_txt', 'iyear'])[
        ['nkill', 'nwound', 'attacktype1']].sum().reset_index()
    death = terr9[(terr9['region_txt'] == w_countries) & (terr9['country_txt'] == w_countries1) & (terr9['iyear'] >= select_years[0]) & (terr9['iyear'] <= select_years[1])]['nkill'].sum()
    wound = terr9[(terr9['region_txt'] == w_countries) & (terr9['country_txt'] == w_countries1) & (terr9['iyear'] >= select_years[0]) & (terr9['iyear'] <= select_years[1])]['nwound'].sum()
    attack = terr9[(terr9['region_txt'] == w_countries) & (terr9['country_txt'] == w_countries1) & (terr9['iyear'] >= select_years[0]) & (terr9['iyear'] <= select_years[1])]['attacktype1'].sum()
    colors = ['#FF00FF', '#9C0C38', 'orange']

    return {
        'data': [go.Pie(labels = ['Total Death', 'Total Wounded', 'Total Attack'],
                        values = [death, wound, attack],
                        marker = dict(colors = colors),
                        hoverinfo = 'label+value+percent',
                        textinfo = 'label+value',
                        textfont = dict(size = 13)
                        # hole=.7,
                        # rotation=45
                        # insidetextorientation='radial',

                        )],

        'layout': go.Layout(
            plot_bgcolor = '#010915',
            paper_bgcolor = '#010915',
            hovermode = 'closest',
            title = {
                'text': 'Total Casualties : ' + (w_countries1) + '  ' + '<br>' + ' - '.join(
                    [str(y) for y in select_years]) + '</br>',

                'y': 0.93,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            titlefont = {
                'color': 'white',
                'size': 20},
            legend = {
                'orientation': 'h',
                'bgcolor': '#010915',
                'xanchor': 'center', 'x': 0.5, 'y': -0.07},
            font = dict(
                family = "sans-serif",
                size = 12,
                color = 'white')
        ),

    }

if __name__ == '__main__':
    app.run_server(debug = True)
