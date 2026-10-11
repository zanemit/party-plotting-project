import yaml
import pandas as pd
from pathlib import Path    
from scripts.preprocess_data import load_and_clean_data
from scripts.pca_analysis import perform_pca
from scripts.plot_pca_projection_3d import plot_pca_3d
from scripts.plot_pca_lollipop_1d import plot_pca_1d
from scripts.plot_pc_loading_heatmap import pc_loading_heatmap
from scripts.plot_party_proximity_heatmap import get_scree_plot, plot_proximity_heatmap
from scripts.texts import TEXTS

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

df_encoded_list = load_and_clean_data(config)

year_links = []
for year, df_encoded in zip(config['data']['YEARS'], df_encoded_list):
    # map questions to topics
    topping_mapping_path = Path(config['data']['TOPIC_MAPPING_PATH'])
    topping_mapping_path = topping_mapping_path.with_name(
        f"{topping_mapping_path.stem}_{year}{topping_mapping_path.suffix}"
        )
    topic_table = pd.read_excel(topping_mapping_path)
    if topic_table['tēma'].isna().sum()==0: # does not contain nans
        mapping = dict(zip(topic_table['jautājums'], topic_table['tēma']))
        df_encoded = df_encoded.rename(columns=mapping)

    # PCA
    pca_df, loadings, variance_ratios = perform_pca(df_encoded)
    pca_df_3d = pca_df.iloc[:, :3]
    loadings_3d = loadings.iloc[:, :3]

    # how many components explain at least 80% of variance? + scree_plot
    get_scree_plot(variance_ratios, config['outputs']['OUTPUT_FOLDER_PATH'], year)
    
    # 3D PCA projection
    fig = plot_pca_3d(pca_df_3d, variance_ratios, config['colours'], df_encoded.index)
    plotly_html_snippet = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # PC loading heatmap - sorted by virstēma
    df_encoded_sorted = df_encoded.copy()
    loadings_sorted = loadings_3d.copy()
    if topic_table['virstēma'].isna().sum()==0:
        topic_table_sorted = topic_table.sort_values('virstēma')
        if topic_table['tēma'].isna().sum()==0:
            df_encoded_sorted = df_encoded[topic_table_sorted['tēma'].tolist()]
            loadings_sorted = loadings_3d.loc[topic_table_sorted['tēma'].tolist(), :]
    fig_heatmap = pc_loading_heatmap(loadings_sorted, df_encoded_sorted.columns)
    heatmap_html_snippet = fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn')

    # 1D projections
    fig_PC1 = plot_pca_1d(pca_df_3d, labels=df_encoded.index, 
                          colours=[config['colours'][party] for party in df_encoded.index],
                          direction=(1, 0, 0), axis_title="PC1")
    plotly_html_snippet_PC1 = fig_PC1.to_html(full_html=False, include_plotlyjs='cdn')

    fig_PC2 = plot_pca_1d(pca_df_3d, labels=df_encoded.index, 
                              colours=[config['colours'][party] for party in df_encoded.index],
                              direction=(0, 1, 0), axis_title="PC2")
    plotly_html_snippet_PC2 = fig_PC2.to_html(full_html=False, include_plotlyjs='cdn')

    fig_PC3 = plot_pca_1d(pca_df_3d, labels=df_encoded.index, 
                              colours=[config['colours'][party] for party in df_encoded.index],
                              direction=(0, 0, 1), axis_title="PC3")
    plotly_html_snippet_PC3 = fig_PC3.to_html(full_html=False, include_plotlyjs='cdn')

    # proximity heatmap
    fig_proximity_heatmap = plot_proximity_heatmap(
        pca_df, 
        reference_party=config['proximity_plot'][year]['reference_party'] or 'jauna-vienotiba',
        n_components=config['proximity_plot'][year]['dimensions'] or 3
        )
    plotly_html_proximity = fig_proximity_heatmap.to_html(full_html=False, include_plotlyjs='cdn')

    if year not in TEXTS.keys():
        print("Populate the dict in texts.py with the new texts!")
        TEXTS[year] = {f"PC{i+1}": {"before-plot": "",  "after-plot": ""} for i in range(3)}
        TEXTS[year]['proximity'] = ""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="lv">
    <head>
        <meta charset="UTF-8">
        <style>body {{font-family: Arial, Helvetica, sans-serif;}}</style>
        <title>Partiju atbildes {year}.gada partiju šķirotavā</title>
    </head>
        <body>
        <div style="width: 60%; margin: auto;">
            <h1>Partiju atbildes {year}.gada partiju šķirotavā</h1>
            <p>No {config['metadata']['TOTAL_NUM_PARTIES'][year]} partijām, kas kandidēja {year}.gada vēlēšanās, uz domnīcas Providus un LSM.lv veidotās Partiju šķirotavas {df_encoded.shape[1]} jautājumiem atbildēja {df_encoded.shape[0]} partijas.<br><br>Zemāk atainota partiju atbilžu projekcija pirmo trīs galveno komponentu (<i>principal component</i>, "PC") asīs, kas kopīgi paskaidro {sum(variance_ratios[:3]):.0%} no variācijas partiju atbildēs.</p>
            
            <!-- Plotly 3D plot -->
            {plotly_html_snippet}   

            <p> Galveno komponentu analīze (<i>principal component analysis</i>) šajos {df_encoded.shape[1]}-dimensionālajos datos atrod savstarpēji perpendikulāras asis, pa kurām partiju atbilžu datos ir vislielākā dispersija. Katra ass atspoguļo <u>lineāru</u> {df_encoded.shape[1]}-dimensiju (jautājumu) kombināciju, kurā katram no šķirotavas jautājumiem ir savs svars. Šī analīzes metode <i>a priori</i> nezina neko par partiju ideoloģiju; tā balstās tikai uz līdzībām partiju atbildēs.</p>
            
            <p>Katru no šīm trim galveno komponentu asīm lielākā vai mazākā mērā ietekmē liels skaits šķirotavas jautājumu, tāpēc šo asu nozīmi nav viegli interpretēt.</p>
            
            <p>Zemāk atainots katra šķirotavas jautājuma korelācijas koeficients ar attiecīgo asi. Jo lielāks koeficients (pozitīvs vai negatīvs), jo lielāka ietekme uz konkrēto asi. Pozitīvs koeficients nozīmē, ka pozitīva atbilde ("pilnībā piekrītu" vai "daļēji piekrītu") uz jautājumu augstāk esošajā 3D grafikā (vai zemāk esošajos 1D grafikos) virza partijas projekciju uz attiecīgās ass pozitīvo skaitļu virzienā. Taču, tā kā katru asi ietekmē liels jautājumu skaits, tas, ka kādas partijas atbilžu projekcijai ir izteikti pozitīva vērtība, nenozīmē, ka partija uz *visiem* šo asi būtiski ietekmējošajiem jautājumiem ir bijusi piekrītoša.</p>

            <!-- Heatmap -->
            <div style="width: 80%; max-width: 1200px; margin-left: -30%;">
                {heatmap_html_snippet} 
            </div>
            
            <hr>
            <h3>PC1 ass</h3>
            <p>{TEXTS[year]['PC1']['before-plot']}</p>
            
            <!-- Plotly 1D plot -->
                {plotly_html_snippet_PC1} 

            <p>{TEXTS[year]['PC1']['after-plot']}</p> 

            <hr>
            <h3>PC2 ass</h3>
            <p>{TEXTS[year]['PC2']['before-plot']}</p>

            <!-- Plotly 1D plot -->
                {plotly_html_snippet_PC2}

            <p> {TEXTS[year]['PC2']['after-plot']} </p>

            <hr>
            <h3>PC3 ass</h3>
            <p>{TEXTS[year]['PC3']['before-plot']} </p>
            
            <!-- Plotly 1D plot -->
                {plotly_html_snippet_PC3}

            <p>{TEXTS[year]['PC3']['after-plot']}</p>

            <hr>
            <h3>Partiju atbilžu līdzība</h3>
            <p>Lai novērtētu, cik kopumā līdzīgi atbildējuši atsevišķi partiju pāri, aprēķinām Eiklīda distances starp partiju atbildēm N-dimensionālā telpā, kas kopā paskaidro vismaz 80% variācijas partiju atbildēs. Šajā gadījumā N={config['proximity_plot'][year]['dimensions'] or 3}, proti, apskatām pirmos {config['proximity_plot'][year]['dimensions'] or 3} galvenos komponentus. Mazākas distances vērtības ir atainotas tumšākos toņos un apzīmē kopumā līdzīgākas atbildes Šķirotavas jautājumos.
            <!-- Proximity heatmap -->
                {plotly_html_proximity}
            <p>{TEXTS[year]['proximity']}</p>
        
        </div>
        </body>
    </html>
    """

    with open(f"{config['outputs']['OUTPUT_FOLDER_PATH']}/index{year}.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    year_links.append(f'<li><a href="{config["outputs"]["OUTPUT_FOLDER_PATH"]}/index{year}.html">{year}</a></li>')

# create the main index.html
index_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Partiju šķirotavu PCA</title>
</head>
<body>
    <h1>Partiju šķirotavu PCA</h1>
    <ul>
        {"".join(year_links)}
    </ul>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(index_content)