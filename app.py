import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import combinations
from collections import Counter
import zipfile
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from streamlit_extras.stylable_container import stylable_container

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Dashboard d'Analyse des Ventes 360°",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Injection CSS pour un design froid et moderne ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
        * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        .main { background: #f4f6f8; color: #333; }
        /* Sidebar avec gradient froid (bleu foncé) */
        .stSidebar { background: linear-gradient(135deg, #002F6C, #00509E); color: white; }
        /* Header avec une ambiance américaine et froide */
        .banking-header {
            background: linear-gradient(135deg, #002F6C 0%, #00509E 100%);
            padding: 2rem; 
            border-radius: 0 0 25px 25px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .stPlotlyChart { border: none; }
        .dataframe { border-radius: 10px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- En-tête personnalisé ---
with stylable_container(key="header", css_styles=".banking-header { color: white !important; }"):
    st.markdown("""
        <div class='banking-header'>
            <h1 style='margin:0;'>Dashboard d'Analyse des Ventes 360°</h1>
            <p style='opacity:0.8;'>Interface interactive pour explorer vos données commerciales</p>
        </div>
    """, unsafe_allow_html=True)

# --- Fonction utilitaire pour créer des "metric cards" compactes ---
def metric_card(title, value, bg_color):
    html = f"""
    <div style="
        background-color: {bg_color};
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        ">
        <h4 style="margin: 0; font-weight: 600; font-size: 1rem;">{title}</h4>
        <p style="font-size: 1.5rem; margin: 5px 0 0; font-weight: bold;">{value}</p>
    </div>
    """
    return html

# --- Chargement du fichier ZIP ---
st.sidebar.header("Chargement des données")
uploaded_zip = st.sidebar.file_uploader("Charger le fichier ZIP (12 CSV)", type=["zip"])

@st.cache_data(show_spinner=True)
def load_and_merge_zip(uploaded_zip_bytes):
    with zipfile.ZipFile(io.BytesIO(uploaded_zip_bytes)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        df_list = []
        for file in csv_files:
            with z.open(file) as f:
                df = pd.read_csv(f)
                df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

if uploaded_zip:
    try:
        data = load_and_merge_zip(uploaded_zip.read())
        st.success("Données chargées et fusionnées avec succès !")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

    # --- Nettoyage & transformation ---
    data.dropna(inplace=True)
    data["Quantity Ordered"] = pd.to_numeric(data["Quantity Ordered"], errors="coerce")
    data["Price Each"] = pd.to_numeric(data["Price Each"], errors="coerce")
    data.dropna(inplace=True)
    data["Order Date"] = pd.to_datetime(data["Order Date"], format="%m/%d/%y %H:%M", errors="coerce")
    data.dropna(subset=["Order Date"], inplace=True)
    data["Month"] = data["Order Date"].dt.month
    data["Hour"] = data["Order Date"].dt.hour
    data["Sales"] = data["Quantity Ordered"] * data["Price Each"]

    # --- Filtres dans la barre latérale ---
    st.sidebar.header("🔎 Filtres Stratégiques")
    villes = st.sidebar.multiselect("Villes", options=sorted(data["Purchase Address"].unique()))
    mois = st.sidebar.multiselect("Mois", options=sorted(data["Month"].unique()), default=sorted(data["Month"].unique()))
    if villes:
        data = data[data["Purchase Address"].isin(villes)]
    data = data[data["Month"].isin(mois)]

    # --- Création des onglets ---
    tabs = st.tabs(["📊 Dashboard Ventes", "👥 Segmentation Clients", "🔄 Vision 360"])

    # Onglet Vision 360 (contenu à enrichir ultérieurement)
    if len(tabs) > 2:
        with tabs[2]:
            st.subheader("Vision 360")
            st.write("Contenu de l'onglet Vision 360 à définir.")
    else:
        st.error("L'onglet 'Vision 360' n'a pas été créé correctement.")

    # =========================
    # Onglet 1 : Dashboard Ventes
    # =========================
    with tabs[0]:
        st.subheader("Dashboard Ventes")
        # Calcul des KPI
        total_sales = data["Sales"].sum()
        total_orders = data["Order ID"].nunique()
        total_customers = data["Purchase Address"].nunique()

        # Affichage dans des metric cards compactes, avec des teintes froides
        col1, col2, col3 = st.columns(3)
        col1.markdown(metric_card("Chiffre d'Affaires Total", f"{total_sales:,.2f} €", "#2E8B57"), unsafe_allow_html=True)
        col2.markdown(metric_card("Nombre de Commandes", total_orders, "#1E90FF"), unsafe_allow_html=True)
        col3.markdown(metric_card("Nombre de Clients", total_customers, "#4682B4"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Ventes par Mois")
        monthly_sales = data.groupby("Month")["Sales"].sum().reset_index()
        fig_month = px.bar(monthly_sales, x="Month", y="Sales",
                           title="CA par Mois",
                           text_auto=True,
                           color="Sales",
                           color_continuous_scale=["#1E90FF", "#4682B4"],
                           template="plotly_white")
        fig_month.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        # Graphiques côte à côte : Ventes par Produit et Combinaisons de Produits
        left_chart, right_chart = st.columns(2)
        with left_chart:
            st.markdown("#### Ventes par Produit")
            if {"Product", "Quantity Ordered"}.issubset(data.columns):
                product_sales = data.groupby("Product")["Quantity Ordered"].sum().reset_index()
                fig_prod = px.bar(product_sales, x="Product", y="Quantity Ordered",
                                  title="Ventes par Produit",
                                  color="Quantity Ordered",
                                  color_continuous_scale=["#1E90FF", "#5F9EA0"],
                                  template="plotly_white")
                fig_prod.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_prod, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Impossible d'afficher les ventes par produit.")
        with right_chart:
            st.markdown("#### Combinaisons de Produits")
            if {"Order ID", "Product"}.issubset(data.columns):
                basket = data.groupby("Order ID")["Product"].apply(list).reset_index()
                combo_counter = Counter()
                for products in basket["Product"]:
                    combos = combinations(sorted(set(products)), 2)
                    combo_counter.update(combos)
                top_combos = combo_counter.most_common(5)
                if top_combos:
                    st.write("5 combinaisons de produits les plus fréquentes :")
                    for combo, count in top_combos:
                        st.write(f"- **{combo[0]}** & **{combo[1]}** : {count} fois")
                else:
                    st.info("Pas de combinaisons trouvées.")
            else:
                st.warning("Colonnes 'Order ID' et 'Product' manquantes.")

    # =========================
    # Onglet 2 : Segmentation Clients
    # =========================
    with tabs[1]:
        st.subheader("Segmentation Clients")
        cust = data.groupby("Purchase Address").agg({
            "Sales": "sum",
            "Order ID": pd.Series.nunique,
            "Quantity Ordered": "sum"
        }).reset_index()
        cust.rename(columns={"Order ID": "NbCmd"}, inplace=True)
        st.dataframe(cust.head(10), height=240)
        st.markdown("---")
        st.subheader("Détermination du Nombre Optimal de Clusters")
        feats = cust[["Sales", "NbCmd", "Quantity Ordered"]]
        X = StandardScaler().fit_transform(feats)
        seg_mode = st.radio("Mode de segmentation", ("Manuel", "Automatique (Silhouette)"))
        if seg_mode == "Automatique (Silhouette)":
            scores = {}
            for k in range(2, 11):
                preds = KMeans(n_clusters=k, random_state=42).fit_predict(X)
                scores[k] = silhouette_score(X, preds)
            best_k = max(scores, key=scores.get)
            st.success(f"Nombre optimal de clusters : {best_k}")
            n_clusters = best_k
        else:
            n_clusters = st.slider("Nombre de segments", 2, 10, 3)
        st.markdown("---")
        st.subheader("Application du Clustering")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        cust["Cluster"] = kmeans.labels_
        
        st.markdown("#### Répartition et CA par Segment")
        col_a, col_b = st.columns(2)
        with col_a:
            seg_count = cust["Cluster"].value_counts().reset_index()
            seg_count.columns = ["Cluster", "Clients"]
            fig_seg_count = px.bar(seg_count, x="Cluster", y="Clients",
                                   title="Clients par Segment",
                                   template="plotly_white",
                                   color="Clients",
                                   color_continuous_scale=["#1E90FF", "#5F9EA0"])
            fig_seg_count.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_seg_count, use_container_width=True, config={"displayModeBar": False})
        with col_b:
            seg_sales = cust.groupby("Cluster")["Sales"].sum().reset_index()
            seg_sales["Prop (%)"] = 100 * seg_sales["Sales"] / seg_sales["Sales"].sum()
            fig_seg_sales = px.pie(seg_sales, names="Cluster", values="Prop (%)",
                                   title="Répartition du CA par Segment",
                                   template="plotly_white",
                                   hole=0.4,
                                   color_discrete_sequence=["#5F9EA0", "#708090", "#1E90FF"])
            fig_seg_sales.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_seg_sales, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("---")
        st.subheader("Top Produits par Segment")
        # Affichage des top produits de chaque cluster côte à côte (groupés par 3)
        merged = pd.merge(data, cust[["Purchase Address", "Cluster"]], on="Purchase Address", how="left")
        clusters = sorted(cust["Cluster"].unique())
        for i in range(0, len(clusters), 3):
            cols = st.columns(min(3, len(clusters)-i))
            for j, c in enumerate(clusters[i:i+3]):
                with cols[j]:
                    st.markdown(f"**Segment {c}**")
                    cluster_data = merged[merged["Cluster"] == c]
                    if "Product" in cluster_data.columns:
                        p_counts = cluster_data.groupby("Product")["Quantity Ordered"].sum().reset_index().sort_values(by="Quantity Ordered", ascending=False)
                        st.dataframe(p_counts.head(5), height=180)
                    else:
                        st.info("Colonne 'Product' manquante.")
        st.markdown("---")
        st.subheader("Visualisation des Segments (PCA)")
        pca = PCA(n_components=2)
        pca_feats = pca.fit_transform(X)
        cust["PCA1"], cust["PCA2"] = pca_feats[:, 0], pca_feats[:, 1]
        fig_pca = px.scatter(cust, x="PCA1", y="PCA2", color="Cluster",
                             title="PCA - Segmentation Clients",
                             template="plotly_white",
                             color_continuous_scale=["#5F9EA0", "#1E90FF", "#708090"])
        fig_pca.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pca, use_container_width=True, config={"displayModeBar": False})

    # =========================
    # Onglet 3 : Vision 360
    # =========================
    with tabs[2]:
        st.subheader("Vision 360°")
        # KPI financiers (exemple)
        revenue = data["Sales"].sum()
        expenses = revenue * 0.2  # hypothèse de 20%
        gross_profit = revenue - expenses
        net_profit = gross_profit * 0.8
        
        # Affichage des KPI financiers dans 4 metric cards aux tons froids
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(metric_card("Revenu", f"${revenue/1_000_000:.2f}M", "#1E90FF"), unsafe_allow_html=True)
        col2.markdown(metric_card("Dépenses", f"${expenses/1_000_000:.2f}M", "#708090"), unsafe_allow_html=True)
        col3.markdown(metric_card("Marge Brute", f"${gross_profit/1_000_000:.2f}M", "#003366"), unsafe_allow_html=True)
        col4.markdown(metric_card("Marge Nette", f"${net_profit/1_000_000:.2f}M", "#2E8B57"), unsafe_allow_html=True)
        
        st.markdown("---")
        # Graphiques financiers mensuels : Barres et Ligne côte à côte
        monthly_df = data.groupby("Month")["Sales"].sum().reset_index().rename(columns={"Sales": "Revenue"})
        monthly_df["Expenses"] = monthly_df["Revenue"] * 0.2
        monthly_df["Net_Profit"] = monthly_df["Revenue"] - monthly_df["Expenses"]
        monthly_df["Growth"] = monthly_df["Revenue"].pct_change().fillna(0) * 100
        
        fin_left, fin_right = st.columns(2)

        with fin_left:
            st.markdown("#### Revenu, Dépenses et Bénéfice (Mensuel)")
            fig_fin = px.bar(
                monthly_df,
                x="Month",
                y=["Revenue", "Expenses", "Net_Profit"],
                barmode="stack",  # affichage en tirroir (stacked bars)
                text_auto=True,
                title="Analyse Mensuelle",
                template="plotly_white",
                # Palette froide sans vert
                color_discrete_sequence=["#003366", "#444444", "#2E8B57"]
            )
            # Réduction des gaps pour l'effet "tirroir"
            fig_fin.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                bargap=0.05
            )
            # Personnalisation des textes pour une meilleure lisibilité
            fig_fin.update_traces(textfont=dict(color="white"), cliponaxis=True)
            st.plotly_chart(fig_fin, use_container_width=True, config={"displayModeBar": False})
        
            with fin_right:
                st.markdown("#### Croissance du Revenu (%)")
                fig_growth = px.line(
                    monthly_df,
                    x="Month",
                    y="Growth",
                    markers=True,
                    title="Croissance Mensuelle",
                    template="plotly_white",
                    color_discrete_sequence=["#033366"]
                )
                # Augmentation de l'épaisseur de la ligne et des marqueurs, avec opacité fixée à 1
                fig_growth.update_traces(
                    line=dict(width=4, color="#033366"), 
                    marker=dict(size=10, opacity=1),
                    opacity=1
                )
                fig_growth.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_growth, use_container_width=True, config={"displayModeBar": False})



        st.markdown("---")
        # Donut chart pour la répartition par jour de la semaine
        if "Order Date" in data.columns:
            df_dow = data.copy()
            df_dow["Weekday"] = df_dow["Order Date"].dt.dayofweek
            day_sales = df_dow.groupby("Weekday")["Sales"].sum().reset_index()
            day_map = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
            day_sales["Jour"] = day_sales["Weekday"].map(day_map)
            st.markdown("#### Répartition du CA par Jour de la Semaine")
            fig_donut = px.pie(day_sales, names="Jour", values="Sales",
                               hole=0.4,
                               title="Distribution des Ventes par Jour",
                               template="plotly_white",
                               color="Jour",
                               color_discrete_sequence=["#1E90FF", "#708090", "#2E8B57", "#003366", "#5F9EA0"])
            fig_donut.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Aucune colonne 'Order Date' trouvée, impossible de calculer la répartition par jour.")

    st.markdown("---")
    st.markdown("Vous pouvez ajouter d’autres visuels ou indicateurs en bas pour enrichir encore la vue 360°.")

    if st.checkbox("Afficher un aperçu des données"):
        st.subheader("Aperçu du Dataset")
        st.dataframe(data.head(), height=300)

# --- Footer personnalisé ---
st.markdown("""
    <div class="footer">
      <p>
        Développé par <strong>IBRAHIMA GABAR DIOP</strong> – Développeur Data • 
        Retrouvez-moi sur <a href="https://www.linkedin.com/in/ibrahima-gabar-diop-730537237/" target="_blank">LinkedIn</a> • 
        Consultez mon <a href="https://portfolio-igd.onrender.com" target="_blank">Portfolio</a> • 
        Retrouvez mon code sur <a href="https://github.com/Gblack98" target="_blank">GitHub</a> • 
        Pour suggestions et questions : <a href="mailto:gabardiop1@outlook.com">gabardiop1@outlook.com</a>
      </p>
    </div>
""", unsafe_allow_html=True)

