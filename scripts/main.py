import yaml
from scripts.preprocess_data import load_and_clean_data, encode_answers
from scripts.pca_analysis import perform_pca
from scripts.plot_pca_projection_3d import plot_pca_3d
from scripts.plot_pc_loading_heatmap import pc_loading_heatmap

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

df_comb = load_and_clean_data(config['data']['DATA_FOLDER'])
df_encoded = encode_answers(df_comb)

# map questions to topics
import pandas as pd
topic_table = pd.read_excel(config['data']['TOPIC_MAPPING_PATH'])
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
        <p>No 19 partijām, kas kandidēja 2022.gada vēlēšanās, uz LSM.lv sagatavotajiem 58 jautājumiem atbildēja 13 partijas.<br>Zemāk atainota partiju atbilžu projekcija pirmo trīs galveno komponentu (<i>principal component</i>, "PC") asīs, kas kopīgi paskaidro 67% no variācijas partiju atbildēs.</p>
        
        <!-- Plotly 3D plot -->
        {plotly_html_snippet}   

        <p> Galveno komponentu analīze (<i>principal component analysis</i>) mūsu 58-dimensionālajos datos atrod savstarpēji perpendikulāras asis, pa kurām partiju atbilžu datos ir vislielākā dispersija. Katra ass atspoguļo <u>lineāru</u> 58-dimensiju (jautājumu) kombināciju, kurā katram no šķirotavas jautājumiem ir savs svars. Šī analīzes metode <i>a priori</i> nezina neko par partiju ideoloģiju; tā balstās tikai uz līdzībām partiju atbildēs.</p>
        
        <p>Katru no šīm trim galveno komponentu asīm lielākā vai mazākā mērā ietekmē liels skaits šķirotavas jautājumu, tāpēc šo asu nozīmi nav viegli interpretēt.</p>
        
        <p>Zemāk atainots katra šķirotavas jautājuma svars. Jo lielāks absolūtais svars (pozitīvs vai negatīvs), jo lielāka ietekme uz konkrēto asi. Pozitīvs svars nozīmē, ka pozitīva atbilde ("pilnībā piekrītu" vai "daļēji piekrītu") uz jautājumu augstāk esošajā 3D grafikā virza partijas projekciju uz attiecīgās ass pozitīvo skaitļu virzienā.</p>

         <!-- Heatmap -->
        <div style="width: 80%; max-width: 1200px; margin-left: -30%;">
            {heatmap_html_snippet} 
        </div>

        <p><b>PC1</b>: Partijas atrašanās vietu uz šīs ass visbūtiskāk ietekmē atbildes uz jautājumiem par (1) Latvijas lomu globālu problēmu risināšanā, (2) uzturēšanās atļaujām trešo valstu investoriem, (3) valsts atbalstu AirBaltic, (4) obligāto militāro dienestu, kā arī (5) attieksmi pret bioloģisko un konvencionālo lauksaimniecību.</p>

        <p> Pozitīvo skaitļu (JV, NA, AP, PRO) virzienu visvairāk veicina šādas idejas:
            <ul>
                <li>Latvijai <b>ir</b> vairāk jāiesaistās globālu (visu planētu skarošu) problēmu risināšanā;</li>
                <li>Latvijai <b>nav</b> vajadzīga programma, kas ļauj ārvalstu pilsoņiem iegūt uzturēšanās atļauju Latvijā tad, ja viņi investē Latvijas ekonomikā;</li>
                <li>Latvijas valdības līdzšinējais finansiālais atbalsts lidsabiedrībai "airBaltic" <b>nav</b> bijis pārmērīgs;</li>
                <li>Latvijā <b>ir</b> jāievieš obligātais militārais (valsts aizsardzības) dienests;</li>
                <li>Bioloģiskajai (organiskajai) lauksaimniecībai <b>ir</b> jāsniedz lielāks atbalsts nekā konvencionālajai (parastajai) lauksaimniecībai.</li>
            </ul>
            Pozitīvos skaitļos redzam arī NA, AS, ZZS. Vistālāk negatīvo skaitļu virzienā ir krieviju atbalstošākie spēki (ST, SV, LKS), bet negatīvi skaitļi ir arī LPV, toreizējai Gobzema partijai un partijai Konservatīvie.

            <img src="outputs/3d_saved_PC1.png" alt="View from PC1 side" 
                style="display:block; margin:auto; width:80%; max-width:800px;">
        </p>

        <p><b>PC2</b>: Šo asi ietekmē ļoti raibs jautājumu spektrs. Daļa nozīmīgo jautājumu ir saistīta ar valsts attieksmi pret krievijas karu Ukrainā un valsts aizsardzību, bet relatīvi liels svars ir arī jautājumiem par pirotehnikas lietošanu privātos pasākumos, mežu izciršanu un nepieciešamību ieviest obligātu vidusskolas eksāmenu dabaszinībās.</p>

         <p> Pozitīvo skaitļu (LPV, Konservatīvo, Gobzema partijas) virzienu visvairāk veicina šādas idejas:
            <ul>
                <li>Pirotehnikas lietošana privātos pasākumos <b>ir</b> jāaizliedz;</li>
                <li>Skolēniem skolās <b>ir</b> jāapgūst ieroču lietošanas prasmes;</li>
                <li>Par krievijas naftas sajaukšanu ar citas izcelsmes naftu, lai apietu pret krieviju noteiktās sankcijas, uzņēmumiem <b>ir</b> jānosaka bargāki sodi;</li>
                <li>Latvijā <b>ir</b> nepieciešams palielināt NATO spēku klātbūtni;</li>
                <li>Latvijai <b>ir</b> jānosoda nopietni cilvēktiesību un tiesiskuma pārkāpumi, kas notiek citās Eiropas Savienības valstīs;</li>
                <li>Lai tuvākajos gados veicinātu uzņēmējdarbību un veidotu jaunas darba vietas, plašāka Latvijas mežu izciršana <b>ir</b> attaisnojama;</li>
                <li>Pret Krieviju un Baltkrieviju noteiktās sankcijas <b>nav</b> jāsamazina;</li>
                <li>Obligāts vidusskolas eksāmens dabaszinātnēs <b>ir</b> jāievieš;</li>
                <li>Enerģijas cenu palielinājuma kompensēšanas pabalsti <b>būtu</b> jāizmaksā visiem Latvijas iedzīvotājiem, nevis tikai maznodrošinātajām grupām;</li>
                <li>Stambulas konvencija <b>nav</b> jāratificē.</li>
            </ul>
            Pozitīvos skaitļos redzam arī NA, AS, ZZS. Vistālāk negatīvo skaitļu virzienā ir ST un LKS, bet negatīvi skaitļi ir arī Pro un SV. JV un AP ir tuvu nullei.

            <img src="outputs/3d_saved_PC2.png" alt="View from PC2 side" 
                style="display:block; margin:auto; width:80%; max-width:800px;">
        </p>

        <p><b>PC3</b>: Arī šo asi ietekmē jautājumi par ļoti atšķirīgām tēmām. Daļa svarīgo jautājumu skar krievvalodīgu un nepilsoņu tiesības, taču relatīvi liels svars ir arī jautājumiem par mācību gada ilgumu, marihuānas sodāmību un to, vai ātra lēmumu pieņemšana atspēko procesa slepenību. </p>
        <ul>

        Pozitīvo skaitļu (PRO, AP) virzienu visvairāk veicina šādas idejas:
                <li>Mācību gads <b>ir</b> jāpagarina, saīsinot vasaras brīvlaiku;</li>
                <li>Lai informētu krievvalodīgos seniorus par vakcināciju pret Covid-19, valsts iestādēm <b>ir</b> pieļaujami viņiem aizsūtīt informāciju (vēstuli, e-pastu) krievu valodā;</li>
                <li>Labāk, lai valstij svarīgi lēmumi tiek pieņemti <b>lēni, bet atklāti</b>, nevis slepeni, bet ātri;</li>
                <li>Latvijā <b>nevajadzētu</b> krimināli vai administratīvi sodīt par marihuānas lietošanu;</li>
                <li>Pašvaldības ikdienas darbā (piemēram, iedzīvotāju konsultatīvajās padomēs, sabiedriskajās apspriedēs, līdzdalības budžeta veidošanā) <b>ir</b> jāiesaista visi vietējie iedzīvotāji, ne tikai Latvijas pilsoņi;</li>
                <li>Latvijai <b>ir</b> svarīgi spēcināt Eiropas Savienību ar jaunām funkcijām (pienākumiem) un lielāku budžetu;</li>
                <li>Arī cilvēks, kuram abi vecāki ir etniskie krievi, <b>drīkst</b> sevi uzskatīt par latvieti;</li>
                <li>Ar īpašiem atbalsta pasākumiem (piemēram, kvotām) <b>ir</b> jāveicina, lai Latvijas valsts pārvaldē strādātu vairāk mazākumtautību pārstāvju;</li>
                <li>Latvijā <b>ir</b> vajadzīga sabiedriskā televīzija arī krievu valodā;</li>
                <li>Katram ienākuma nodokļa maksātājam <b>ir</b> jābūt tiesībām 1% no sava iemaksātā iedzīvotāju ienākuma nodokļa novirzīt jebkurai sabiedriskā labuma organizācijai (biedrībām, nodibinājumiem, kuri veic sabiedriski svarīgus darbus).</li>
            </ul>
            Pozitīvos skaitļos redzam arī LKS, JV, LPV, Gobzema partiju un Konservatīvos. Vistālāk negatīvo skaitļu virzienā ir ZZS un NA, bet negatīvi skaitļi ir arī ST, AS un SV. Jāatzīmē, ka atšķirības LPV, KK un Konservatīvo skaitļos uz šīs ass ir ieviestas mākslīgi, lai šīs partijas varētu redzēt kā trīs atsevišķus punktus; visu triju rezultāts uz PC3 ass bija ~1.43.              </p>
    </div>
    </body>
</html>
"""
with open("app.html", "w", encoding="utf-8") as f:
    f.write(html_content)