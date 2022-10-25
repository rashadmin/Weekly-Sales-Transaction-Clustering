from ipywidgets import Dropdown,interact,IntSlider
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def make_plot(df):
    data = df
    print('SELECT FEATURE TO EXPLORE BELOW: ')
    def make_hist_box_plot(feature):
        fig,ax = plt.subplots(ncols=2,figsize=(25,9))
        sns.boxplot(data= data,x =feature,ax=ax[0])
        sns.histplot(data[feature],ax=ax[1])
        box_string = f'A BoxPlot for Distribution of {feature} for each Target Class'
        hist_string = f'A Histogram for Distribution of {feature} Feature'
        ax[0].set_title(box_string)
        ax[1].set_title(hist_string)
        plt.tight_layout()
    thresh_widget = Dropdown(options=data.columns,value = data.columns[0])
    interact(make_hist_box_plot,feature=thresh_widget)
def make(df):
    data = df
    print('SELECT FEATURE TO EXPLORE BELOW: ')
    # A function that returns the selected feature in the first dropdown box
    #A function that returns both features
    def both_feat(feat_1,feat_2):
        #return feat_1,feat_2
        fig = px.scatter(data[feat_1],data[feat_2])
        fig.show()
    first_feature = Dropdown(options=data.columns,value = data.columns[0])
    second_feature = Dropdown(options=data.columns,value = data.columns[0])
    interact(both_feat,feat_1=first_feature,feat_2=second_feature)
def make_plots(df):
    #data = df.select_dtypes(exclude='object')
    plot_list = ['Scatter','Histogram & Boxplot']
    print('SELECT PLOT TO SHOW BELOW: ')
    def plots(plot_type):
        if plot_type == 'Scatter':
            make(df)
        elif plot_type == 'Histogram & Boxplot':
            make_plot(df)
    thresh_widget = Dropdown(options=plot_list,value = plot_list[1])
    interact(plots,plot_type=thresh_widget)