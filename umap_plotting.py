
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import os

def UMAP_plot(umap_df, savedir, image_id, dataset, plot_show = False, dpi = 100):
    colors, hue_order = return_color_hue_order_jpg(dataset)
    sns.scatterplot(
        data = umap_df,
        x = 'UMAP_dim1', y = 'UMAP_dim2',
        hue = 'label', palette=colors, hue_order=hue_order, s = 1        
    )

    # Get the handles and labels of the plot's legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Filter out the handles and labels that are not present in the plot
    present_labels = set(umap_df['label'])
    filtered_handles = [h for h, l in zip(handles, labels) if l in present_labels]
    filtered_labels = [l for l in labels if l in present_labels]

    # make the labels more readable, instead of 0/1/2/3, we give them meaningful names via the following  
    final_labels = list(map(make_label_readable,[dataset]*len(filtered_labels), filtered_labels))
        
    # Create a new legend with the filtered handles and labels
    plt.legend(filtered_handles, final_labels, title='GT Labels', prop={'size': 10}, bbox_to_anchor=(1, 1))
    
    plt.xlabel("UMAP dim 1")
    plt.ylabel("UMAP dim 2")
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.gcf().set_size_inches(6, 6) 
    plt.savefig(os.path.join(savedir, str(image_id)+"_UMAP.jpg"),bbox_inches = 'tight',dpi=dpi)
    if plot_show:
        plt.show()
    plt.close()

def UMAP_interactive_plot(umap_df, savedir, image_id, dataset):
    groupby = ['label'] ### With this grouping, there should only be four groups
    fig = go.Figure()

    groups = umap_df.groupby(groupby)
    groups_list = list(groups)
    for name,group in reversed(groups_list):
        x = group["UMAP_dim1"]
        y = group["UMAP_dim2"]
        t = group["xy"]
        ##
        size = 5
        group_color = return_color_html(dataset, name)
        fig.add_trace(
            go.Scattergl(
                name = make_label_readable(dataset,name),
                mode = 'markers',
                x = x,
                y = y,
                hovertext = t,
                hoverinfo = 'text',
                marker = dict(color = group_color, size = size)
            )
        )
    fig.update_layout(xaxis_title="UMAP dim 1",yaxis_title="UMAP dim 2" )
    fig.write_html(os.path.join(savedir, str(image_id)+"_UMAP.html"))

def make_label_readable(dataset,input):
    """ This function makes the legend of UMAP plot more readable. 
    """
    if input == "0":
        return "bg" 
    elif "dClosure" in dataset:
        if input == "1": return "boundary"
        elif input == "2": return "square"
    elif "dContinuity" in dataset:
        if input == "1": return "circle(filled)"
        if input == "2": return "circle(painted)"
    elif "dIlluOcclusion" in dataset:
        if input == "1": return "bg_stripe"
        elif input == "2": return "square"
        elif input == "3": return "square_stripe"
    elif "dKanizsa" in dataset:
        if input == "1": return "circle"
        elif input == "2": return "square"
    elif "dGradOcclusion" in dataset:
        if input == "1": return "rectangle(tall)"
        elif input == "2": return "rectangle(wide)"
        elif input == "3": return "overlap"
    elif "dProximity" in dataset:
        if input == "1": return "group 1"
        elif input == "2": return "group 2"
        elif input == "3": return "group 3"
    elif "dCeleba" in dataset:
        atts = ['bg','skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        return atts[int(input)] # although bg will never be returned here... If will be returned in the first if-branch
    else:
        raise NotImplementedError
    

def A_in_B(A,B):
    """A simple helper function"""
    return A in B 

def return_color_hue_order_jpg(dataset):
    """
    return the color dictionary for the jpg plot of umap.
    Different color settings suit different dataset.
    """
    dark_green = '#008000'
    blue = '#0000ff'
    red = '#ff0000'
    yellow = '#ffdf00'

    list1 = ["dClosure", "dContinuity", "dKanizsa", ]
    list2 = ["dIlluOcclusion", "dGradOcclusion","dProximity","dCeleba"]
    if np.array(list(map(A_in_B,list1,[dataset]*len(list1)))).any():
        return {"0":dark_green, "1":red, "2":yellow}, ["0","1","2"]
    elif np.array(list(map(A_in_B,list2,[dataset]*len(list2)))).any():
        colors =  {"0":dark_green,
                "1": blue,
                "2":yellow,
                "3":red,
                "4":"#ED5CAC",
                "5":"#D5ED64",
                "6":"#E3C25F",
                "7":"#64EDE6",
                "8":"#5F64E3",
                "9":"#ED7664",
                "10":"#64EDB2",
                "11":"#625FE3",
                "12":"#5AC2ED",
                "13":"#E3C8CD",
                "14":"#40A2C9",
                "15":"#DB7F5E",
                "16":"#C76CA7",
                "17":"#6EFFFC",
                "18": "#0A665A"}
        hue_order = [str(x) for x in np.arange(19)]
        return colors, hue_order
    else:
        raise NotImplementedError


def return_color_html(dataset,name):
    dark_green = '#008000'
    blue = '#0000ff'
    red = '#ff0000'
    yellow = '#ffdf00'
    def color_func_1(name_1):
        if name_1== "0": group_color = dark_green
        elif name_1== "1": group_color = yellow 
        elif name_1== "2": group_color = red
        return group_color
    def color_func_2(name_1):
        if name_1== "0": group_color = dark_green
        elif name_1== "1": group_color = blue 
        elif name_1== "2": group_color = yellow 
        elif name_1== "3": group_color = red
        elif name_1== "4":group_color ="#ED5CAC"
        elif name_1== "5":group_color ="#D5ED64"
        elif name_1== "6":group_color ="#E3C25F"
        elif name_1== "7":group_color ="#64EDE6"
        elif name_1== "8":group_color ="#5F64E3"
        elif name_1== "9":group_color ="#ED7664"
        elif name_1== "10":group_color ="#64EDB2"
        elif name_1== "11":group_color ="#625FE3"
        elif name_1== "12":group_color ="#5AC2ED"
        elif name_1== "13":group_color ="#E3C8CD"
        elif name_1== "14":group_color ="#40A2C9"
        elif name_1== "15":group_color ="#DB7F5E"
        elif name_1== "16":group_color ="#C76CA7"
        elif name_1== "17":group_color ="#6EFFFC"
        elif name_1== "18":group_color = "#0A665A"
        return group_color
    list1 = ["ShapeClosure", "LineClosure", "Kanizsa", ]
    list2 = ["SwissFlag", "GradOcclusion","Proximity","Celeb"]
    if np.array(list(map(A_in_B,list1,[dataset]*len(list1)))).any():
        return color_func_1(name)
    elif np.array(list(map(A_in_B,list2,[dataset]*len(list2)))).any():
        return color_func_2(name)
    else:
        raise NotImplementedError

    
    

        
    
