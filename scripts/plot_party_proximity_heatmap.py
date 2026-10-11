from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def get_scree_plot(explained_variance, output_folder, year):
    # scree plot
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    ax.plot(range(1, len(explained_variance) + 1), explained_variance,
        marker='o', linestyle='-', color='black')
    ax.set_xlabel('Principal component'); ax.set_ylabel('Explained variance')  
    fig.savefig(f"{output_folder}/scree_plot_{year}.png", dpi=300, bbox_inches='tight' )

    cumulative_variance = np.cumsum(explained_variance)
    at_least_80pc = (cumulative_variance<=0.8).sum()
    print(f"""
        At least 80% of variance is explained by {at_least_80pc+1} components. 
        Check the scree plot in the output folder.
        Then go to config.yaml to specify the number of components to keep for proximity analysis.
        """)

def plot_proximity_heatmap(df_pca, reference_party, n_components=3):
    # compute pairwise euclidean distances
    X = df_pca.iloc[:, :n_components].to_numpy()
    distances = cdist(X, X, metric='euclidean')
    labels = df_pca.index.tolist()

    # sort relative to the reference party
    ref_idx = labels.index(reference_party)
    order = np.argsort(distances[ref_idx])
    distances = distances[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    
    # mask the diagonal and the upper triangle
    mask = np.triu(np.ones_like(distances, dtype=bool))
    masked_distances = distances.astype(float)
    masked_distances[mask] = np.nan

    # plot heatmap
    fig = go.Figure(data=go.Heatmap(
        z=masked_distances,
        x=sorted_labels, y=sorted_labels,
        colorscale="GnBu_r",
        colorbar=dict(title=f"Eiklīda distance"),
        hovertemplate=("%{z:.3f}<extra></extra>")
    ))
  
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(df_pca.index))),
            ticktext=sorted_labels,
            tickangle=90
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(df_pca.index))),
            ticktext=sorted_labels,
            autorange="reversed"
        ),
        width=60*df_pca.shape[0],
        height=50*df_pca.shape[0],
    )

    return fig