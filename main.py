import dash
from dash.dependencies import Output, Input
from dash import html
from dash import dcc
import plotly.express as px
import pandas as pd
from datetime import datetime
from datetime import date
import plotly.graph_objects as go
import numpy as np
from dash import no_update
import dash_bootstrap_components as dbc
import ast

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Masukan data ke masing masing dataframe
data_2014 = './jumlah-wisatawan-tahun-2014.csv'
data_2015 = './jumlah-wisatawan-tahun-2015.csv'
data_2016 = './jumlah-wisatawan-tahun-2016.csv'
data_2017 = './jumlah-wisatawan-tahun-2017.csv'
data_2018 = './jumlah-wisatawan-tahun-2018.csv'
data_2019 = './jumlah_wisatawan_tahun_2019.csv'

df_2014 = pd.read_csv(data_2014)
df_2015 = pd.read_csv(data_2015)
df_2016 = pd.read_csv(data_2016)
df_2017 = pd.read_csv(data_2017)
df_2018 = pd.read_csv(data_2018)
df_2019 = pd.read_csv(data_2019)
df_2019.columns = df_2019.columns.str.lower()

df_list = [df_2014, df_2015, df_2016, df_2017, df_2018, df_2019]

# gabungkan dataframe menjadi satu
dataset = pd.concat(
    [df_2014, df_2015, df_2016, df_2017, df_2018, df_2019], axis=0)

# mengambil data dengan nilai wisatawan nusantara, serta melakukan perubahan dataframe
wisatawan_nusantara = dataset[(dataset.wisatawan == "Wisatawan Nusantara") | (
    dataset.wisatawan == "Wisatawan Nusantara ")]
wisatawan_nusantara.drop('wisatawan', axis=1, inplace=True)
wisatawan_nusantara.rename(
    columns={'jumlah': 'wisatawan_nusantara'}, inplace=True)

# mengambil data dengan nilai wisatawan mancanegara, serta melakukan perubahan dataframe
wisatawan_mancanegara = dataset[(dataset.wisatawan == "Wisatawan Mancanegara") | (
    dataset.wisatawan == "Wisatawan Mancanegara ")]
wisatawan_mancanegara.drop('wisatawan', axis=1, inplace=True)
wisatawan_mancanegara.rename(
    columns={'jumlah': 'wisatawan_mancanegara'}, inplace=True)

# menggabungkan dataframe menjadi dataframe final dan mengkelompokkan berdasarkan tahun
final_data = pd.merge(wisatawan_nusantara,
                      wisatawan_mancanegara, on=['bulan', 'tahun'])
final_data['bulan'] = final_data['bulan'].apply(lambda x: x.strip())

# membuat label encoder untuk bulan
le = LabelEncoder()
le.fit(final_data["bulan"])
final_data['bulan_enc'] = le.transform(final_data["bulan"])
# print(final_data.head())
# membuat scaler untuk wisatawan nusantara
scaler1 = StandardScaler()
scaler1.fit(final_data[['bulan_enc', 'tahun']])
X = scaler1.transform(final_data[['bulan_enc', 'tahun']])
y = final_data['wisatawan_nusantara']

# membuat polynomial Feature untuk wisatasan nusantara
poly_reg1 = PolynomialFeatures(degree=4)
X = poly_reg1.fit_transform(X)
poly_reg1.fit(X, y)

# membuat regresi untuk wisatasan nusantara
lr1 = LinearRegression()
lr1.fit(X, y)

# membuat scaler untuk wisatasan mancanegara
scaler2 = StandardScaler()
scaler2.fit(final_data[['bulan_enc', 'tahun']])
X = scaler2.transform(final_data[['bulan_enc', 'tahun']])
y = final_data['wisatawan_mancanegara']

# membuat polynomial Feature untuk wisatasan mancanegara
poly_reg2 = PolynomialFeatures(degree=6)
X = poly_reg2.fit_transform(X)
poly_reg2.fit(X, y)

# membuat regresi untuk wisatawan mancanegara
lr2 = LinearRegression()
lr2.fit(X, y)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True
                )

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Microcredential Data Science",
                    className='text-center'),
            html.H3("Jumlah Wisatawan Kota Banda Aceh",
                    className='text-center'),
            html.H6("sumber : https://data.go.id/pemerintah-kota-banda-aceh/jumlah-wisatawan-kota-banda-aceh",
                    className='text-center', style={'margin-bottom': 50}),
        ], width=12)
    ], justify='center', className='mt-2 mb-2'),

    dbc.Row([
        dbc.Col([
            html.H5('Wisatawan Mancanegara'),
            # dropdown bulan
            dcc.Dropdown(
                options=[
                    {'label': 'Semua', 'value': 9},
                    {'label': '2014', 'value': 0},
                    {'label': '2015', 'value': 1},
                    {'label': '2016', 'value': 2},
                    {'label': '2017', 'value': 3},
                    {'label': '2018', 'value': 4},
                    {'label': '2019', 'value': 5},
                ],
                id='tabelYear_manca',
                value=9,
            ),
            dcc.Graph(id='output_graph'),
            html.H5('Wisatawan Nusantara'),
            dcc.Dropdown(
                options=[
                    {'label': 'Semua', 'value': 9},
                    {'label': '2014', 'value': 0},
                    {'label': '2015', 'value': 1},
                    {'label': '2016', 'value': 2},
                    {'label': '2017', 'value': 3},
                    {'label': '2018', 'value': 4},
                    {'label': '2019', 'value': 5},
                ],
                id='tabelYear_nusan',
                value=9,
            ),
            dcc.Graph(id='output_graph2')
        ], width=9),

        dbc.Col([
            html.H5('Prediksi wisatawan'),
            html.H5('Bulan :'),
            dcc.Dropdown(
                options=[
                    {'label': 'januari', 'value': int(
                        le.transform(['januari']))},
                    {'label': 'februari', 'value': int(
                        le.transform(['februari']))},
                    {'label': 'maret', 'value': int(le.transform(['maret']))},
                    {'label': 'april', 'value': int(le.transform(['april']))},
                    {'label': 'mei', 'value': int(le.transform(['mei']))},
                    {'label': 'juni', 'value': int(le.transform(['juni']))},
                    {'label': 'juli', 'value': int(le.transform(['juli']))},
                    {'label': 'agustus', 'value': int(
                        le.transform(['agustus']))},
                    {'label': 'september', 'value': int(
                        le.transform(['september']))},
                    {'label': 'oktober', 'value': int(
                        le.transform(['oktober']))},
                    {'label': 'november', 'value': int(
                        le.transform(['november']))},
                    {'label': 'desember', 'value': int(
                        le.transform(['desember']))},
                ],
                id="input_month",
                value=int(le.transform(['januari'])),
            ),

            html.H5('Tahun :', style={'margin-top': 20}),
            dcc.Input(
                id="input_year",
                type='number',
                value=2020
            ),

            html.H5('Prediksi Wisatawan Mancanegara',
                    style={'margin-top': 20}),
            html.Div(id='output_manca'),

            html.H5('Prediksi Wisatawan Nusantara',
                    style={'margin-top': 20}),
            html.Div(id='output_nusan'),

        ], width=2)
    ], justify='center', className='mt-2 mb-2'),

], fluid=True)


@app.callback(
    Output(component_id='output_graph', component_property='figure'),
    Output(component_id='output_graph2', component_property='figure'),
    Output(component_id='output_manca', component_property='children'),
    Output(component_id='output_nusan', component_property='children'),

    Input(component_id='tabelYear_manca', component_property='value'),
    Input(component_id='tabelYear_nusan', component_property='value'),
    Input(component_id='input_month', component_property='value'),
    Input(component_id='input_year', component_property='value'),
)
def update_graph(tabelYear_manca, tabelYear_nusan, input_month, input_year,):
    # cetak tabel mancanegara sesuai input
    if(tabelYear_manca == 9):
        fig1 = px.line(final_data, x='tahun',
                       y='wisatawan_mancanegara', color='bulan')
    else:
        show_data = df_list[tabelYear_manca]
        show_data = show_data[(show_data.wisatawan == "Wisatawan Mancanegara") | (
            show_data.wisatawan == "Wisatawan Mancanegara ")]
        fig1 = px.bar(show_data, x='bulan',
                      y='jumlah', color='bulan')

    # cetak tabel nusantara sesuai input
    if(tabelYear_nusan == 9):
        fig2 = px.line(final_data, x='tahun',
                       y='wisatawan_nusantara', color='bulan')
    else:
        show_data2 = df_list[tabelYear_nusan]
        show_data2 = show_data2[(show_data2.wisatawan == "Wisatawan Nusantara") | (
            show_data2.wisatawan == "Wisatawan Nusantara ")]
        fig2 = px.bar(show_data2, x='bulan',
                      y='jumlah', color='bulan')

    predict_data1 = [[input_month, input_year]]
    predict_data1 = scaler1.transform(predict_data1)
    predict_data1 = poly_reg1.fit_transform(predict_data1)

    nusan = str(abs(int(lr1.predict(predict_data1))))

    predict_data2 = [[input_month, input_year]]
    predict_data2 = scaler2.transform(predict_data2)
    predict_data2 = poly_reg2.fit_transform(predict_data2)

    manca = str(abs(int(lr2.predict(predict_data2))))

    return fig1, fig2, manca, nusan


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
