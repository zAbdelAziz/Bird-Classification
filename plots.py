import os
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_multiple(f_path, cols, titles, plot_mode='lines', title='', x_title='', y_title='', log_y=False, log_x=False):
    df = pd.read_csv(f_path)
    print(df)
    fig = go.Figure()
    for i in range(len(cols)):
        fig.add_trace(go.Scatter(x=titles, y=df[cols[i]],
                                 mode=plot_mode,
                                 name=titles[i],
                                 marker_size=50))
    fig.update_layout(title=title,
                      xaxis_title=x_title, yaxis_title=y_title,
                      template='plotly_white',
                      font=dict(family="Helvetica", size=24),
                      plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                      legend=dict(orientation="h")
                      )
    fig.update_yaxes(showgrid=True, type="log" if log_y else "linear", gridwidth=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, type="log" if log_x else "category", gridcolor='black')
    fig.show()



def plot_confusion(f_path):
    confusions = pd.read_csv(f_path)
    confusions /= confusions.sum(axis=0)
    confusions = confusions.iloc[:,1:]
    # confusions = (df-df.min())/(df.max()-df.min())
    trace = go.Heatmap(z=confusions, x=list(range(len(confusions.columns))), y=list(range(len(confusions.columns))),
                       colorscale='Blues')
    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
        template='plotly_white',
        font=dict(family="Helvetica", size=24),
        plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()



output_dir = os.path.join('output', 'reports')


model_name = 'savings'
f_path = os.path.join(output_dir, f'{model_name}.csv')
plot_multiple(os.path.join(output_dir, f'{model_name}.csv'), ['result'],
              ['Non Weighted', 'Model A', 'Augmented',
               'Model B',  'Model C', '1D + Binary', 'Ensemble',  'No Sampling',  'Guided Heuristic', 'Proposed'],
              plot_mode='markers', x_title='Model classes', y_title='Total Savings',
              title='Comparison of Total Savings between Different Methods',)
# plot_confusion(os.path.join(output_dir, f'best_valid_conf_{model_name}.csv'))
# plot_multiple(os.path.join(output_dir, f'{model_name}.csv'), ['2_0', '1_0', '1_200',  '1_12', '1_101'], ['No Sampling', 'Over Sampling', 'Under Sampling',  'Over + Under Sampling', 'Non-Weighted'], title='Comparison of Average Recall between Different Sampling Techniques', x_title='Epochs', y_title='Average Recall')
# plot_multiple(os.path.join(output_dir, f'{model_name}.csv'), ['1_12', '2_1003'], ['Single Model', 'Ensemble'], title='Comparison of Average Recall between Single Model and Majority Voting of Ensemble', x_title='Epochs', y_title='Average Recall')

