import plotly.express as px

def plot_pca_3d(pca_df, variance_ratios, clr_dict, index_labels, 
                camera_view=dict(x=0.5, y=2, z=1), output_html=False,
                output_png=False, PC_in_view=None):
    variance_ratios = variance_ratios
    fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color=index_labels,
        color_discrete_map=clr_dict,
        title='',
        labels={
        'PC1': f'PC1 ({variance_ratios[0]*100:.0f}%)',
        'PC2': f'PC2 ({variance_ratios[1]*100:.0f}%)',
        'PC3': f'PC3 ({variance_ratios[2]*100:.0f}%)',
        'color': 'Partija'
    }
    )

    fig.update_layout(
        scene_camera=dict(
                eye=camera_view
                ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
                domain=dict(x=[0, 1], y=[0.1, 1])
            ),
        legend=dict(
            x=0.8,   # move legend left/right (0=left, 1=right)
            y=0.5,   # move legend up/down (0=bottom, 1=top)
            xanchor='left',  # how x is interpreted (relative to left edge)
            bgcolor='rgba(255,255,255,0.6)',  # optional: semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        height=500
    )
    if output_html:
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        fig.write_html(config['outputs']['3D_PLOT_PATH'])

    if output_png:
        import yaml
        import os
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        filename = f"3d_saved_{PC_in_view}.png"
        fig.write_image(os.path.join(config['outputs']['OUTPUT_FOLDER_PATH'], filename), 
                        scale=2, width=600, height=400)

    return fig
