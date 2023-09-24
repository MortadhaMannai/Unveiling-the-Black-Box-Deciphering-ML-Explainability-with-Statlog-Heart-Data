from IPython.display import HTML, display
import io
import base64
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels=['0', '1'], title='Confusion Matrix', ax=None):
    import seaborn as sns

    if not ax:
        ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax=ax, fmt='.0f')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])


def plot_coefficients(estimator, feature_names, figsize=(15, 5)):
    coef = estimator.coef_.ravel()

    plt.figure(figsize=figsize)
    colors = ['red' if c < 0 else 'blue' for c in coef]
    plt.bar(np.arange(len(coef)), coef, color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, len(coef)), feature_names, rotation=60, ha='right')
    plt.show()


#
# Found on https://stackoverflow.com/a/49566213
#
class FlowLayout(object):
    ''' A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml =  """
        <style>
        .floating-box {
        display: inline-block;
        margin: 0px;
        border: none;  
        }
        </style>
        """

    def add_plot(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio = io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml+= (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def add_figure(self, oFigure):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio = io.BytesIO() # bytes buffer for the plot
        oFigure.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml+= (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))