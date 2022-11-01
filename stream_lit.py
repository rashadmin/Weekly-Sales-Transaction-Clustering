# Import Libraries
import streamlit as st
from wrangle import wrangle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
st.set_page_config(layout='wide')
st.title('SALES TRANSACTION CLUSTERING PLAYGROUND')

st.write('[Source Code](https://github.com/rashadmin/Weekly-Sales-Transaction-Clustering/blob/25a8d877458faca5db482dbc7cc9f354c9076054/stream_lit.py)')
st.write('[Jupyter Notebook](https://github.com/rashadmin/Weekly-Sales-Transaction-Clustering/blob/25a8d877458faca5db482dbc7cc9f354c9076054/Sales%20Transaction%20Clustering.ipynb)')

df = wrangle('Sales_Transactions_Dataset_Weekly.csv')
st.write(df.head())
#st.write('Select how many feature to cluster')
col_list = df.columns
st.header('Select Number of Feature to Train KMeans Clustering Model')
option = st.radio(label='',options=['None',2,3,'or More','All'],index=0,horizontal=True)
graph_color = st.sidebar.color_picker('Graph Colors')
if option == 'None':
    st.stop()
elif option == 'or More':
    option = st.number_input('Select Number of Feature ',min_value=4,max_value=len(df.columns),value=4)
    feature = st.sidebar.multiselect('Select Features to Cluster',options=col_list,max_selections=option)
elif option == 'All':
    feature = col_list
    option =  len(col_list)
else:
    feature = st.sidebar.multiselect('Select Features to Cluster',options=col_list,max_selections=option)
new_df = df[feature]
dist = st.checkbox('Show Distribution Plots ',value=False)
if (len(feature) < option) or not dist:
    heat_variance = st.checkbox('Show Variance and Heatmap Plots ',value=False)
else:
    if option >2:
        col1,col2 = st.columns(2)
        with col1:
            feat_1 = st.selectbox('Select Features' , options = feature)
            fig,ax= plt.subplots()
            ax = sns.histplot(df[feat_1],color=graph_color)
            plt.xlabel(f'{feat_1}')
            plt.ylabel('Values')
            plt.title(f'A Histogram showing the Distribution of the {feat_1} Feature')
            st.pyplot(fig)
        with col2:
            second_option = df[feature].drop(feat_1,axis=1).columns
            feat_2 = st.selectbox('Select Features' , options = second_option)
            fig,ax= plt.subplots()
            ax = sns.histplot(df[feat_2],color=graph_color)
            plt.xlabel(f'{feat_2}')
            plt.ylabel('Values')
            plt.title(f'A Histogram showing the Distribution of the {feat_2} Feature')
            st.pyplot(fig)
    else:
        col1,col2 = st.columns(2)
        with col1:
            fig,ax= plt.subplots()
            ax = sns.histplot(df[feature[0]],color=graph_color)
            plt.xlabel(f'{feature[0]}')
            plt.ylabel('Values')
            plt.title(f'A Histogram showing the Distribution of the {feature[0]} Feature')
            st.pyplot(fig)
        with col2:
            fig,ax= plt.subplots()
            ax = sns.histplot(df[feature[1]],color=graph_color)
            plt.xlabel(f'{feature[1]}')
            plt.ylabel('Values')
            plt.title(f'A Histogram showing the Distribution of the {feature[1]} Feature')
            st.pyplot(fig)









if heat_variance:
    trimmed = st.checkbox('Trimmed',value=True)
    col1,col2 = st.columns(2)
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
        
        with col1:
            #st.write('\n\n')
            st.header(f'HEATMAP OF THE TOP {option} FEATURES')
            new_df = df[top_10_feature_var.index]
            corr = new_df.corr()
            heat_fig, heat_ax = plt.subplots()
            heat_ax = sns.heatmap(corr,annot=True)
            st.pyplot(heat_fig)
        with col2:
            #st.write('\n\n')
            st.header(f'BAR CHART OF VARIANCE OF THE TOP {option} FEATURES')
            fig,ax= plt.subplots()
            ax = sns.barplot(x = top_10_feature_var.values,y = top_10_feature_var.index,orientation='horizontal',color=graph_color)
            plt.xlabel(f'{name} Variance')
            plt.ylabel('Features')
            plt.title(f'An Horizontal Barchart showing the Top {option} Highest {name} Variance Features')
            st.pyplot(fig)
train = st.checkbox('Train Kmeans Clustering Model',value=False)
if train:
    st.header('Select Number of Clusters ')
    cluster = st.slider(' ',min_value=2,max_value=13,value=3)
    model = make_pipeline(StandardScaler(),KMeans(n_clusters=cluster))
    model.fit(new_df)
    labels = model.named_steps['kmeans'].labels_
    inertia = model.named_steps['kmeans'].inertia_
    ss_score = silhouette_score(new_df,labels)
    new_df['label'] = labels
    st.subheader(f'Inertia Score : {inertia}')
    st.subheader(f'Silhouette Score : {ss_score}')
else:
    st.stop()
    
plot = st.checkbox(label = 'Plot Clustered Scatter Plot',value=False)
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

