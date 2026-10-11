import numpy as np
import plotly.graph_objects as go

def plot_pca_1d(
    pca_df,
    labels=None,
    colours=None,
    direction=(1, 0, 0),
    axis_title="PC1",
):
    """
    pca_df: array of shape (parties, N_dims), containing x, y, z coordinates
    labels: optional list of point labels
    direction: 3D vector defining the projection axis
    """
    points = np.asarray(pca_df, dtype=float)

    # normalise the projection direction
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # scalar coordinate of each point along the chosen axis
    projected = points @ direction

    # sort items by value (so that the staggered heights form a neat cascade)
    order = np.argsort(projected)
    projected = projected[order]
    points = points[order]

    # labels
    if labels is None:
        labels = [str(i + 1) for i in range(len(points))]
    else:
        labels = [labels[i] for i in order]

    # colours
    if colours is None:
        colours = ['teal' for i in range(len(points))]
    else:
        colours = [colours[i] for i in order]

    # stagger points vertically to prevent overlap
    heights = np.linspace(1, 0.15, len(points))

    fig = go.Figure()

    # projection lines from each point to the horizontal baseline
    line_x, line_y = [], []
    for x, y in zip(projected, heights):
        line_x.extend([x, x, None])
        line_y.extend([0, y, None])

    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="gray", width=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    # points above the axis
    fig.add_trace(go.Scatter(
        x=projected,
        y=heights,
        mode="markers+text",
        text=labels,
        textposition="middle right",
        marker=dict(
        size=13,
        color=colours,  # One colour per marker
        ),
        customdata=points,
        hovertemplate=(
            "Label: %{text}<br>"
            "Projection: %{x:.3f}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # horizontal projection axis
    xmin, xmax = projected.min(), projected.max()

    fig.update_layout(
        xaxis=dict(
            title=f"Partijas pozīcija uz {axis_title} ass",
            range=[xmin-0.5, xmax + (len(labels[-1])*0.25)],
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False, ticks="",
            showgrid=False,zeroline=False),
        margin=dict(l=30, r=80, t=60, b=60),
        template="plotly_white",
    )

    # add the horizontal axis line
    fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=0.05, y1=0.05,
            xref="paper", yref="paper",
            line=dict(color="black", width=1),
        )

    return fig