import plotly.graph_objects as go
import pandas as pd
import numpy as np

def pc_loading_heatmap(loadings, loading_colnames):
    # Transpose so PCs are rows, features are columns
    loadings_T = pd.DataFrame(
        loadings.T,
        index=[f'PC{i+1}' for i in range(loadings.shape[1])],
        columns=loading_colnames
    )
    loadings_T = loadings_T.loc[loadings_T.index[::-1],:]

    # Mask for "strong" loadings
    mask = (loadings_T > 0.2) | (loadings_T < -0.2)

    # Prepare hover text with optional stars
    hover_text = round(loadings_T.copy(),3).astype(str)
    for i in range(loadings_T.shape[0]):
        for j in range(loadings_T.shape[1]):
            if mask.iloc[i, j]:
                hover_text.iloc[i, j] += " ★"

    # Create interactive heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=loadings_T.values,
            x=loadings_T.columns,
            y=loadings_T.index,
            colorscale='RdBu_r',
            zmin=-np.max(np.abs(loadings_T.values)),
            zmax=np.max(np.abs(loadings_T.values)),
            hovertext=hover_text.values,
            hoverinfo="text",
            colorbar=dict(title="")
        )
    )

    fig.update_layout(
        width=1200,
        height=600,
        xaxis_title="Jautājuma tēma",
        yaxis_title="Galvenās komponentes"
    )

    fig.add_annotation(
        x=1.1,  # slightly to the right of the colorbar
        y=0.55,
        text="Jautājuma svars",
        showarrow=False,
        xref="paper",
        yref="paper",
        textangle=90,  # rotate vertically
        font=dict(size=14)
    )
    return fig
