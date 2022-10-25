import pandas as pd
import re
from scipy.stats.mstats import trimmed_var
from EDA import make_plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from wrangle import wrangle
#Libraries for the Dash App
from jupyter_dash import JupyterDash
from dash import Input,Output,html,dcc



app = JupyterDash(__name__)

app.layout = html.Div(
    [
        #Application Title
        html.H1('Weekly Sales Transaction Simulation Clustering'),
        html.H2('Bar Chart Showing the Value for the Top N Highest Variance Feature'),
        #Barchart for the Var
        dcc.Graph(id='variance_bar_chart'),
        #
        html.H2('Slider for the Top N Variance Feature'),
        dcc.Slider(min=1,max=54,value= 5,step=1,id='n_var'),
        dcc.RadioItems(
            options=[
                {'label':'trimmed','value':True},
                {'label':'not trimmed','value':False},
            ],
            value = True,
            id = 'trim_button'
                    ),
         html.H2('Slider for the  K Cluster'),
         dcc.Slider(min=2,max=13,value= 3,step=1,id='k_slider'),
         html.H3('Metrics for the KMeans Cluster Model'),
         html.Div(id='metrics'),
         html.H2('A Scatter Plot showing the  Clustering of the Dimensionally Reduced Feature'),
         dcc.Graph(id='pca-scatter')
    ]
)

wrangled_df = wrangled_df = wrangle('Sales_Transactions_Dataset_Weekly.csv')

def high_var(n=5,trimmed=False,return_feat_name=False):
    var_feat = wrangled_df.var().sort_values(ascending=False).tail(n)
    if trimmed:
        var_feat = wrangled_df.apply(trimmed_var).sort_values(ascending=False).tail(n)
    if return_feat_name:
        return var_feat.index
    return var_feat


@app.callback(
    Output('variance_bar_chart','figure'), Input('n_var','value') ,Input('trim_button','value')
)
def serve_bar_chart(n=5,trimmed=False,return_feat_name=False):
    top_n_feature_var = high_var(n=n,trimmed=trimmed,return_feat_name=return_feat_name)
    fig = px.bar(top_n_feature_var,orientation='h',title=f'An Horizontal Barchart showing the Top {n} Highest Variance Features',
                labels={'index':'Features','value':'Variance'})
    return fig

def get_model(n =5,trimmed=False,k=3,return_metrics=False,return_feat_name=True):
    X = wrangled_df[high_var(return_feat_name=return_feat_name,trimmed=trimmed,n=n)]
    model = make_pipeline(StandardScaler(),KMeans(n_clusters=k))
    model.fit(X)
    inertia = model.named_steps['kmeans'].inertia_
    labels = model.named_steps['kmeans'].labels_
    ss_score = silhouette_score(X,labels)
    metrics = {'inertia':inertia,'ss_score':ss_score}
    if return_metrics:
        return metrics
    return model

@app.callback(
    Output('metrics','children'), Input('n_var','value') ,Input('trim_button','value'),Input('k_slider','value')
)
def serve_metrics(n =5,trimmed=False,k=3,return_metrics=True,return_feat_name=True):
    metrics = get_model(n =n,trimmed=trimmed,k=k,return_metrics=return_metrics,return_feat_name=return_feat_name)
    text = [
        html.H3(f'Inertia = {metrics["inertia"]}'),
        html.H3(f'Silhouette Score = {metrics["ss_score"]}')
    ]
    return text

def get_scatter(n =5,trimmed=False,k=3,return_metrics=False,return_feat_name=True):
    sub_set = wrangled_df[high_var(return_feat_name=return_feat_name,trimmed=trimmed,n=n)]
    pca_df = pd.DataFrame()
    pca = make_pipeline(StandardScaler(),PCA(n_components=2,random_state=42))
    pca_df[['pca1','pca2']] = pca.fit_transform(sub_set)
    model = get_model(n=n,trimmed=trimmed,k=k,return_metrics=return_metrics,return_feat_name=return_feat_name)
    labels = model.named_steps['kmeans'].labels_
    pca_df['label'] = labels
    return pca_df

@app.callback(
    Output('pca-scatter','figure'), Input('n_var','value') ,Input('trim_button','value'),Input('k_slider','value')
)
def serve_scatter(n =5,trimmed=False,k=3,return_metrics=False,return_feat_name=True):
    data = get_scatter(n =n,trimmed=trimmed,k=k,return_metrics=return_metrics,return_feat_name=return_feat_name)
    fig = px.scatter(data_frame=data,x='pca1',y='pca2',color='label')
    return fig

def run_simulation():
    app.run_server(host='0.0.0.0',mode='external')