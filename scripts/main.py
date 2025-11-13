import yaml
from scripts.preprocess_data import load_and_clean_data
from scripts.pca_analysis import perform_pca
from scripts.plot_pca_projection_3d import plot_pca_3d
from scripts.plot_pc_loading_heatmap import pc_loading_heatmap

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

df_encoded = load_and_clean_data(config)

# map questions to topics
import pandas as pd
topic_table = pd.read_excel(config['data']['TOPIC_MAPPING_PATH'])
if not topic_table['tēma'].isna().sum()==topic_table.shape[0]: # all are nans
    mapping = dict(zip(topic_table['jautājums'], topic_table['tēma']))
    df_encoded = df_encoded.rename(columns=mapping)

# PCA
pca_df, loadings, variance_ratios = perform_pca(df_encoded)

# 3D PCA projection
fig = plot_pca_3d(pca_df, variance_ratios, config['colours'], df_encoded.index)
plotly_html_snippet = fig.to_html(full_html=False, include_plotlyjs='cdn')

# PC loading heatmap
fig_heatmap = pc_loading_heatmap(loadings, df_encoded.columns)
heatmap_html_snippet = fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn')

# save images
_ = plot_pca_3d(pca_df, variance_ratios, config['colours'], df_encoded.index, output_png=True,
                camera_view=dict(x=-2, y=0, z=0), PC_in_view='PC2')
_ = plot_pca_3d(pca_df, variance_ratios, config['colours'], df_encoded.index, output_png=True,
                camera_view=dict(x=0, y=2, z=0), PC_in_view='PC1')
_ = plot_pca_3d(pca_df, variance_ratios, config['colours'], df_encoded.index, output_png=True,
                camera_view=dict(x=2, y=0, z=0.125), PC_in_view='PC3')

html_content = f"""
<!DOCTYPE html>
<html lang="lv">
<head>
    <meta charset="UTF-8">
    <title>Partiju atbildes 2022.gada partiju šķirotavā</title>
</head>
    <body>
    <div style="width: 60%; margin: auto;">
        <h1>Partiju atbildes 2022.gada partiju šķirotavā</h1>
        <p>No 19 partijām, kas kandidēja 2022.gada vēlēšanās, uz LSM.lv sagatavotajiem 58 jautājumiem atbildēja 13 partijas.<br>Zemāk atainota partiju atbilžu projekcija pirmo trīs galveno komponentu (<i>principal component</i>, "PC") asīs, kas kopīgi paskaidro 59% no variācijas partiju atbildēs.</p>
        
        <!-- Plotly 3D plot -->
        {plotly_html_snippet}   

        <p> Galveno komponentu analīze (<i>principal component analysis</i>) mūsu 58-dimensionālajos datos atrod savstarpēji perpendikulāras asis, pa kurām partiju atbilžu datos ir vislielākā dispersija. Katra ass atspoguļo <u>lineāru</u> 58-dimensiju (jautājumu) kombināciju, kurā katram no šķirotavas jautājumiem ir savs svars. Šī analīzes metode <i>a priori</i> nezina neko par partiju ideoloģiju; tā balstās tikai uz līdzībām partiju atbildēs.</p>
        
        <p>Katru no šīm trim galveno komponentu asīm lielākā vai mazākā mērā ietekmē liels skaits šķirotavas jautājumu, tāpēc šo asu nozīmi nav viegli interpretēt.</p>
        
        <p>Zemāk atainots katra šķirotavas jautājuma svars. Jo lielāks absolūtais svars (pozitīvs vai negatīvs), jo lielāka ietekme uz konkrēto asi. Pozitīvs svars nozīmē, ka pozitīva atbilde ("pilnībā piekrītu" vai "daļēji piekrītu") uz jautājumu augstāk esošajā 3D grafikā virza partijas projekciju uz attiecīgās ass pozitīvo skaitļu virzienā.</p>

         <!-- Heatmap -->
        <div style="width: 80%; max-width: 1200px; margin-left: -30%;">
            {heatmap_html_snippet} 
        </div>

        <p><b>PC1</b>: Partijas atrašanās vietu uz šīs ass visbūtiskāk ietekmē atbildes uz jautājumiem par (1) valsts atbalstu AirBaltic, (2) uzturēšanās atļaujām trešo valstu investoriem, (3) okupekļa nojaukšanu, (4) Latvijas lomu globālu problēmu risināšanā, kā arī (5) attieksmi pret bioloģisko un konvencionālo lauksaimniecību.</p>

        <p> Pozitīvo skaitļu (SV, ST, LPV, LKS, KK) virzienu visvairāk veicina šādas idejas:
            <ul>
                <li>Latvijas valdības līdzšinējais finansiālais atbalsts lidsabiedrībai "airBaltic" <b>ir</b> bijis pārmērīgs;</li>
                <li>Latvijai <b>ir</b> vajadzīga programma, kas ļauj ārvalstu pilsoņiem iegūt uzturēšanās atļauju Latvijā tad, ja viņi investē Latvijas ekonomikā;</li>
                <li><b>Nav</b> jānojauc piemineklis Rīgā, Uzvaras parkā;</li>
                <li>Latvijai <b>nav</b> vairāk jāiesaistās globālu (visu planētu skarošu) problēmu risināšanā;</li>
                <li>Bioloģiskajai (organiskajai) lauksaimniecībai <b>ir</b> jāsniedz lielāks atbalsts nekā konvencionālajai (parastajai) lauksaimniecībai.</li>
            </ul>
            Vispretējākos viedokļus šiem paudušas redzam arī JV, NA un Konservatīvie. Negatīvos skaitļos redzam arī AP, PRO, AS un ZZS.

            <img src="outputs/3d_saved_PC1.png" alt="View from PC1 side" 
                style="display:block; margin:auto; width:80%; max-width:800px;">
        </p>

        <p><b>PC2</b>: Šo asi visvairāk ietekmē jautājumi par cilvēktiesībām, tiesiskumu un attieksmi pret Eiropas Savienību, taču būtiski ir arī saimnieciski jautājumi, kā pašvaldību ienākuma  nodokļa piesaistīšana darba vietai.</p>

        <p> Pozitīvo skaitļu (LPV, Konservatīvo, ZZS) virzienu visvairāk veicina šādas idejas:
            <ul>
                <li>Heteroseksuālajiem pāriem <b>ir</b> jābūt plašākām tiesībām nekā viendzimuma pāriem;</li>
                <li>Latvijai <b>nav</b> jāratificē Stambulas konvencija (par vardarbības pret sievietēm un vardarbības ģimenē novēršanu un apkarošanu);</li>
                <li>Latvijai <b>nav</b> svarīgi spēcināt Eiropas Savienību ar jaunām funkcijām (pienākumiem) un lielāku budžetu;</li>      
                <li>Satversmes tiesas spriedumi <b>nav</b> jāpilda, ja deputāti un daļa sabiedrības tiem nepiekrīt;</li>
                <li>Latvijā <b>vajadzētu</b> krimināli vai administratīvi sodīt par marihuānas lietošanu;</li>
                <li>Par pašvaldības koplietošanas infrastruktūru <b>nav</b> jāmaksā no to iedzīvotāju nodokļiem, kuri strādā attiecīgajā pašvaldībā, ja viņi dzīvo citur.</li>
            </ul>
            Pozitīvos skaitļos redzam arī Gobzema partiju, NA un AS. Vistālāk negatīvo skaitļu virzienā ir PRO un LKS, bet negatīvi skaitļi ir arī ST, JV, AP. SV ir tuvu nullei.

            <img src="outputs/3d_saved_PC2.png" alt="View from PC2 side" 
                style="display:block; margin:auto; width:80%; max-width:800px;">
        </p>

        <p><b>PC3</b>: Šo asi ietekmē raibs jautājumu spektrs. Daļa svarīgo jautājumu skar nepilsoņu tiesības un valsts attieksmi pret krieviju, taču relatīvi liels svars ir arī jautājumiem par dzimumu līdztiesību, eitanāziju, marihuānas lietotāju sodāmību. </p>

        Pozitīvo skaitļu (Gobzema partija, LPV, AP) virzienu visvairāk veicina šādas idejas:
        <ul>
                <li>Latvijā <b>nav</b> vajadzīgas "sarkanās līnijas" valdības koalīciju veidošanā (partiju atteikšanās sadarboties ar kādām citām partijām);</li>
                <li>Valstij <b>ir</b> jāpublicē to iestāžu, uzņēmumu, organizāciju nosaukumi, kur sievietēm par līdzīgu darbu tiek maksāta ievērojami zemāka atlīdzība nekā vīriešiem;</li>
                <li>Latvijai <b>ir</b> jāsniedz patvērums cilvēkiem, kuri politisko iemeslu dēļ tiek vajāti Krievijā un Baltkrievijā;</li>
                <li>Pašvaldības ikdienas darbā (piemēram, iedzīvotāju konsultatīvajās padomēs, sabiedriskajās apspriedēs, līdzdalības budžeta veidošanā) <b>ir</b> jāiesaista visi vietējie iedzīvotāji, ne tikai Latvijas pilsoņi;</li>
                <li>Latvijā <b>ir</b> jāatļauj eitanāzija kā brīvprātīga izvēle neārstējami slimiem pacientiem;</li>
                <li>Latvijā <b>nevajadzētu</b> krimināli vai administratīvi sodīt par marihuānas lietošanu;</li>
                <li>Skolēniem skolās <b>ir</b> jāapgūst ieroču lietošanas prasmes;</li>
                <li><b>Nav</b> jāsamazina pret Krieviju un Baltkrieviju noteiktās sankcijas.</li>         
            </ul>
        Pozitīvos skaitļos redzam arī Progresīvos. Vistālāk negatīvo skaitļu virzienā ir ST un NA, bet negatīvi skaitļi ir arī Konservatīvajiem un SV. Tuvu nullei negatīvajā pusē ir ZZS un AS, savukārt tuvu nullei pozitīvajā pusē ir JV un LKS.</p>
    </div>
    </body>
</html>
"""
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)