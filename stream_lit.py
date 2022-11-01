import streamlit as st
from wrangle import wrangle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from dash_app import run_simulation

st.title('SALES TRANSACTION CLUSTERING PLAYGROUND')
df = wrangle('Sales_Transactions_Dataset_Weekly.csv')
st.write(df.head())
#st.write('Select how many feature to cluster')
col_list = df.columns
option = st.radio(label='Select Number of Feature to Cluster',options=['None',2,3,'More','All'],index=0)
if option is 'None':
    st.stop()
elif option is 'More':
    option = st.number_input('Select Number of Feature ',min_value=4,max_value=len(df.columns),value=4)
    feature = st.multiselect('Select Features to Cluster',options=col_list,max_selections=option)
elif option is 'All':
    feature = col_list
    option =  len(col_list)
else:
    feature = st.multiselect('Select Features to Cluster',options=col_list,max_selections=option)
new_df = df[feature]

trimmed = st.checkbox('Trimmed',value=True)
if trimmed:
    name = 'trimmed'
    top_10_feature_var = new_df.apply(trimmed_var).sort_values(ascending=False).tail(10)
else:
    name= ''
    top_10_feature_var = new_df.var().sort_values(ascending=False).tail(10)
if option >10:
    option = 10
if len(feature) < option:
    st.stop()
else:
    st.write('\n\n')
    st.subheader(f'HEATMAP OF THE TOP {option} FEATURES')
    new_df = df[top_10_feature_var.index]
    corr = new_df.corr()
    heat_fig, heat_ax = plt.subplots()
    heat_ax = sns.heatmap(corr,annot=True)
    st.pyplot(heat_fig)
    st.write('\n\n')
    st.subheader(f'BAR CHART OF VARIANCE OF THE TOP {option} FEATURES')
    fig = px.bar(top_10_feature_var,orientation='h',title=f'An Horizontal Barchart showing the Top {option} Highest {name} Variance Features',
                    labels={'index':'Features','value':f'{name} Variance'})
    st.plotly_chart(fig)
train = st.checkbox('Train Kmeans Clustering Model',value=False)
if train:
    cluster = st.number_input('Select Number of Clusters ',min_value=2,max_value=13,value=3)
    model = make_pipeline(StandardScaler(),KMeans(n_clusters=cluster))
    model.fit(new_df)
    labels = model.named_steps['kmeans'].labels_
    inertia = model.named_steps['kmeans'].inertia_
    ss_score = silhouette_score(new_df,labels)
    new_df['label'] = labels
    st.write(f'Inertia Score : {inertia}')
    st.write(f'Silhouette Score : {ss_score}')
else:
    st.stop()
plot = st.checkbox('Plot Clustered Scatter Plot',value=False)
if plot:
    if option >2:
        pca_df = pd.DataFrame()
        pca = make_pipeline(StandardScaler(),PCA(n_components=2,random_state=42))
        pca_df[['pca1','pca2']] = pca.fit_transform(new_df.iloc[:,:-1])
        pca_df['label'] = new_df['label']
        pca_fig = px.scatter(pca_df,x = 'pca1',y='pca2',color='label')
        st.plotly_chart(pca_fig)
    else:
        fig = px.scatter(new_df,x=new_df.iloc[:,0],y=new_df.iloc[:,1],color='label',labels={'x':new_df.iloc[:,0].name,'y':new_df.iloc[:,1].name})
        st.plotly_chart(fig)
else:
    st.stop()

<<<<<<< HEAD
    
st.write(' Want to run simulation using dash app ? ')
run = st.checkbox('Run',value = False)
if run:
    st.write('OK')
    st.write(run_simulation())
=======
st.write('Check the [Source Code](https://github.com/rashadmin/Weekly-Sales-Transaction-Clustering/blob/25a8d877458faca5db482dbc7cc9f354c9076054/stream_lit.py) for the app')
st.write('Check out the [notebook](https://github.com/rashadmin/Weekly-Sales-Transaction-Clustering/blob/25a8d877458faca5db482dbc7cc9f354c9076054/Sales%20Transaction%20Clustering.ipynb)	')
>>>>>>> dca301f (Fixed Bugs)
