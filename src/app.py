# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io
import copy
from ortools.sat.python import cp_model
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
# import SCHED_main_code as sched
from dash.dependencies import Input, Output, State
from scheduler import Scheduler, TimeVar
a = None
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


scheduler = Scheduler()
app.layout = html.Div([
    # html.H2(children='Hello You'),
    html.H1(children='Scheduler'),
    dcc.Upload(
            id='upload-data',
            children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select')
                                , ' your XML Files']),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
                    },
        # Allow multiple files to be uploaded
                multiple=True
                    ),
    html.Div(id='output-data-upload'),
    dash_table.DataTable(id='table-package'),
    dash_table.DataTable(id='table-worker'),
    dash_table.DataTable(id='table-location'),
    html.Button('Create Schedule', id='submit-val', n_clicks=0),
    html.Div(id='scheduler-output'),
            
 ])
#  +
#  [dcc.Input(
#             id=f"table-{i}",
#             type=j,
#             placeholder=f"input type {j}",) for i in ["package", "worker", "location"] for j in ["data", "column"]])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    # """
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sheet_name='Sheet_package')
            df2 = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sheet_name='Sheet_worker')  
            df3 = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sheet_name='Sheet_location')
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), sheet_name='Sheet_package')
            df2 = pd.read_excel(io.BytesIO(decoded), sheet_name='Sheet_worker')  
            df3 = pd.read_excel(io.BytesIO(decoded), sheet_name='Sheet_location')
    
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
                ])
    # """
#     return df, df2, df3

# def show_tables(df, df2, df3, filename, date):
#     path = "data/xl.xlsx"
#     df = pd.read_excel(open(path, 'rb'),
#                                        sheet_name='Sheet_package')
#     df2 = pd.read_excel(open(path, 'rb'),
#                                       sheet_name='Sheet_worker')
#     df3 = pd.read_excel(open(path, 'rb'),
#                                         sheet_name='Sheet_location')
    # data = sched.main_code(df,df2,df3,df4,df5,df6,df7,df8,df9) 
    # gantt_data = data[0]
    # gantt_figure = data[1]   
    num_vehicles = 4
    num_shifts = 11
    time_shifts = [TimeVar(6, 30) + TimeVar(0, 20*i)
                   for i in range(num_shifts)]  
    scheduler(df, df2, df3, time_shifts, num_vehicles)
    
    return html.Div([
                html.H5(f'Input filename: {filename}'),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.Div([
                    html.Div(
                        className='row',
                        children=[
                        html.H5('Location info'),
                        dash_table.DataTable(
                            id='table-location',
                            data=df3.to_dict('rows'),
                            columns=[{'name': i, 'id': i} for i in df3.columns],
                            editable=True
                                    ),
                         ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div(
                        className='row',
                        children=[
                        html.H5('Worker info'),
                        dash_table.DataTable(
                            id='table-worker',
                            data=df2.to_dict('rows'),
                            columns=[{'name': i, 'id': i} for i in df2.columns],
                            editable=True
                                    ),
                         ], style={'width': '50%', 'display': 'inline-block'}),
                    ], style={'width': '100%', 'display': 'inline-block'}),
                html.H5('Package info'),
                dash_table.DataTable(
                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    id='table-package',
                    data=scheduler.input_data_package_orig.to_dict('rows'),
                    columns=[{'name': i, 'id': i} for i in ['package', 'quantity', 'decay',
                                'location', 'vehicle', 'next', 'yesterday']],
                    editable=True
                            ),
                html.Hr(),
                #    dcc.Graph(gantt_figure),

                # # For debugging, display the raw contents provided by the web browser
                # html.Div('Raw Content'),
                # html.Pre(contents[0:200] + '...', style={
                #             'whiteSpace': 'pre-wrap',
                #             'wordBreak': 'break-all'
                #         }),
    ])
##################################

@app.callback(Output('scheduler-output', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    Input('table-package', 'data'),
    Input('table-package', 'columns'),
    Input('table-worker', 'data'),
    Input('table-worker', 'columns'),
    Input('table-location', 'data'),
    Input('table-location', 'columns'),
                # Input('output-data-upload', 'children'),
              )
def show_output_table(n_clicks, pr, pc, wr, wc, lr, lc):
# def show_output_table(n_clicks, pr, pc):
    # print(o)
    if scheduler.input_data_package is None: return

    df = pd.DataFrame(pr, columns=[c['name'] for c in pc])
    df2 = pd.DataFrame(wr, columns=[c['name'] for c in wc])
    df3 = pd.DataFrame(lr, columns=[c['name'] for c in lc])
    # df2 = scheduler.input_data_worker
    # df3 = scheduler.input_data_location

    num_vehicles = 4
    num_shifts = 11
    time_shifts = [TimeVar(6, 30) + TimeVar(0, 20*i)
                   for i in range(num_shifts)]  
    scheduler(df, df2, df3, time_shifts, num_vehicles)
    scheduler.run()
    if scheduler.status == cp_model.OPTIMAL:
        scheduler.output_data = scheduler.solution_printer()
        # scheduler.solution_writer()


        return html.Div([
                html.Hr(),  # horizontal line
                html.H5(f'Schedule'),
                # html.H6(datetime.datetime.fromtimestamp(date)),

                dash_table.DataTable(
                    data=scheduler.input_data_package_orig.to_dict('rows'),
                    columns=[{'name': i, 'id': i} for i in scheduler.input_data_package_orig.columns]
                            ),
                html.Hr(),
                #    dcc.Graph(gantt_figure),
                html.Hr(),  # horizontal line

                    ])
    else:
        return html.Div([
                html.H5(f'Schedule'),
                html.H6('No solutions found'),
                    ])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
