import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Analyse Ventes & Contrats",
    # page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
) 

# CSS personnalisé pour une meilleure présentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        background-color: #2E86AB;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-container {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
        margin: 1rem 0;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E4057;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E1E5E9;
        padding-bottom: 0.5rem;
    }
    .kpi-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007BFF;
        margin: 1rem 0;
    }
    .alert-container {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
    }
    .danger-alert {
        background-color: #F8D7DA;
        border-left-color: #DC3545;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #E5E7EB;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .metric-delta-positive {
        color: #10B981;
    }
    .metric-delta-negative {
        color: #EF4444;
    }
    .date-start {
        background-color: #D4EDDA !important;
        color: #155724 !important;
    }
    .date-end-warning {
        background-color: #F8D7DA !important;
        color: #721C24 !important;
    }
    div[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    .fiscal-year-info {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_fiscal_year_bounds(fiscal_year):
    """
    Retourne les dates de début et fin d'une année fiscale.
    L'année fiscale N commence le 1er août de l'année N-1 et finit le 31 juillet de l'année N.
    """
    start_date = datetime(fiscal_year - 1, 8, 1)
    end_date = datetime(fiscal_year, 7, 31)
    return start_date, end_date

def get_fiscal_years_for_contract(start_date, end_date):
    """
    Retourne la liste des années fiscales couvertes par un contrat, mais seulement celles qui ont des données.
    """
    fiscal_years = []
    
    # Convertir en datetime si nécessaire
    if pd.notna(start_date):
        # Si pas de date de fin, utiliser la date actuelle + 1 an
        if pd.isna(end_date):
            end_date = datetime.now() + timedelta(days=365)
            
        # Déterminer l'année fiscale de début
        if start_date.month >= 8:
            start_fiscal_year = start_date.year + 1
        else:
            start_fiscal_year = start_date.year
        
        # Déterminer l'année fiscale de fin
        if end_date.month >= 8:
            end_fiscal_year = end_date.year + 1
        else:
            end_fiscal_year = end_date.year
        
        # Générer toutes les années fiscales entre début et fin
        for year in range(start_fiscal_year, end_fiscal_year + 1):
            fiscal_years.append(year)
    
    return fiscal_years

def get_fiscal_years_with_data(df_groupe):
    """
    Retourne uniquement les années fiscales qui contiennent des données de ventes.
    """
    fiscal_years_with_data = []
    
    if not df_groupe.empty:
        # Obtenir les dates du contrat
        date_debut = df_groupe['DATE DE DÉBUT'].iloc[0]
        date_fin = df_groupe['DATE DE FIN'].iloc[0]
        
        # Obtenir toutes les années fiscales possibles
        all_fiscal_years = get_fiscal_years_for_contract(date_debut, date_fin)
        
        # Vérifier quelles années fiscales ont des données
        for fiscal_year in all_fiscal_years:
            start_date, end_date = get_fiscal_year_bounds(fiscal_year)
            
            # Filtrer les données pour cette année fiscale
            mask = (
                (df_groupe['Posting Date'] >= start_date) & 
                (df_groupe['Posting Date'] <= end_date) &
                (df_groupe['Posting Date'] >= df_groupe['DATE DE DÉBUT'])
            )
            
            # Si pas de date de fin, ne pas filtrer par date de fin
            if pd.notna(df_groupe['DATE DE FIN'].iloc[0]):
                mask = mask & (df_groupe['Posting Date'] <= df_groupe['DATE DE FIN'])
            
            if df_groupe[mask]['Customer Sales'].sum() > 0:
                fiscal_years_with_data.append(fiscal_year)
    
    return fiscal_years_with_data

def filter_data_by_fiscal_year(df, fiscal_year):
    """
    Filtre les données pour une année fiscale spécifique.
    """
    start_date, end_date = get_fiscal_year_bounds(fiscal_year)
    
    # Obtenir les dates du contrat
    contract_start = df['DATE DE DÉBUT'].iloc[0] if not df.empty else None
    contract_end = df['DATE DE FIN'].iloc[0] if not df.empty else None
    
    if contract_start:
        # Ajuster les dates selon la logique demandée
        # La période effective est l'intersection entre l'année fiscale et le contrat
        effective_start = max(start_date, contract_start)
        
        # Si pas de date de fin de contrat, utiliser la fin de l'année fiscale
        if pd.isna(contract_end):
            effective_end = end_date
        else:
            effective_end = min(end_date, contract_end)
        
        # Filtrer les ventes dans cette période
        mask = (df['Posting Date'] >= effective_start) & (df['Posting Date'] <= effective_end)
        df_filtered = df[mask].copy()
        
        # Stocker les dates effectives pour l'affichage
        df_filtered.attrs['actual_start'] = effective_start
        df_filtered.attrs['actual_end'] = effective_end
        
        return df_filtered
    
    return df[mask].copy()

def filter_data_by_custom_dates(df, start_date, end_date):
    """
    Filtre les données pour une période personnalisée.
    """
    # Obtenir les dates du contrat
    contract_start = df['DATE DE DÉBUT'].iloc[0] if not df.empty else None
    contract_end = df['DATE DE FIN'].iloc[0] if not df.empty else None
    
    if contract_start:
        # La période effective est l'intersection entre la période choisie et le contrat
        effective_start = max(start_date, contract_start)
        
        # Si pas de date de fin de contrat, utiliser la date de fin choisie
        if pd.isna(contract_end):
            effective_end = end_date
        else:
            effective_end = min(end_date, contract_end)
        
        # Filtrer les ventes dans cette période
        mask = (df['Posting Date'] >= effective_start) & (df['Posting Date'] <= effective_end)
        df_filtered = df[mask].copy()
        
        # Stocker les dates effectives pour l'affichage
        df_filtered.attrs['actual_start'] = effective_start
        df_filtered.attrs['actual_end'] = effective_end
        
        return df_filtered
    
    # Si pas de contrat, utiliser les dates directement
    mask = (df['Posting Date'] >= start_date) & (df['Posting Date'] <= end_date)
    df_filtered = df[mask].copy()
    df_filtered.attrs['actual_start'] = start_date
    df_filtered.attrs['actual_end'] = end_date
    
    return df_filtered

def apply_global_date_filter(df):
    """
    Applique le filtre de date global basé sur les paramètres dans session_state
    """
    if 'global_date_filter_type' not in st.session_state:
        return df
    
    filter_type = st.session_state.global_date_filter_type
    
    if filter_type == "Toute la période":
        return df
    elif filter_type == "Année Fiscale" and 'global_fiscal_year' in st.session_state:
        fiscal_year = st.session_state.global_fiscal_year
        return filter_data_by_fiscal_year(df, fiscal_year)
    elif filter_type == "Dates Personnalisées" and 'global_custom_start' in st.session_state and 'global_custom_end' in st.session_state:
        start_date = st.session_state.global_custom_start
        end_date = st.session_state.global_custom_end
        return filter_data_by_custom_dates(df, start_date, end_date)
    
    return df

def create_monthly_sales_chart(df, fiscal_year):
    """
    Crée un graphique de l'évolution du CA par mois pour l'année fiscale.
    """
    if df.empty:
        return None
    
    # Créer une copie pour éviter les warnings
    df_copy = df.copy()
    
    # Extraire l'année et le mois pour un meilleur affichage
    df_copy['YearMonth'] = df_copy['Posting Date'].dt.strftime('%Y-%m')
    df_copy['MonthName'] = df_copy['Posting Date'].dt.strftime('%b %Y')
    
    # Grouper par mois
    monthly_sales = df_copy.groupby(['YearMonth', 'MonthName'])['Customer Sales'].sum().reset_index()
    monthly_sales = monthly_sales.sort_values('YearMonth')
    
    # Créer le graphique simple mais efficace
    fig = px.bar(
        monthly_sales, 
        x='MonthName', 
        y='Customer Sales',
        title=f"Évolution mensuelle du CA - {fiscal_year}",
        labels={'Customer Sales': 'Chiffre d\'affaires (€)', 'MonthName': ''},
        color='Customer Sales',
        color_continuous_scale='Blues'
    )
    
    # Personnaliser l'apparence
    fig.update_traces(
        texttemplate='',  # Supprimer le texte
        marker_line_color='darkblue',
        marker_line_width=1
    )
    
    # Mise en page simple
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickangle=-45),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            tickformat=',.2f'  # Format avec 2 décimales
        )
    )
    
    # Masquer la barre de couleur
    fig.update_coloraxes(showscale=False)
    
    return fig

def preprocess_and_merge_data(df_sales, df_clients):
    """
    Prétraite les données de ventes et clients, puis fusionne et filtre selon les dates de contrat.
    Retourne un DataFrame final avec les données valides.
    """
    try:
        # =========================
        # Prétraitement des ventes
        # =========================
        df_sales_clean = df_sales.copy()
        mask_valid_format = (
        df_sales_clean['SoldTo Managed Group'].notna() &  # Pas de valeurs nulles
        (df_sales_clean['SoldTo Managed Group'] != '-') &  # Pas juste un tiret
        df_sales_clean['SoldTo Managed Group'].str.contains('-', na=False)  # Contient au moins un tiret
    )
        df_sales_clean = df_sales_clean[mask_valid_format].copy()
        

        # Créer les colonnes Groupe et Interlocuteur
        # Pour gérer les cas où il n'y a pas de "-" dans le nom
        def split_soldto(value):
            if pd.isna(value) or value == '':
                return None, None
            value_str = str(value).strip()  # Convertir en string
            if '-' not in value_str:
                # Si pas de tiret, tout est le groupe, pas d'interlocuteur
                return value_str, ''
            else:
                parts = value_str.rsplit('-', 1)
                groupe = parts[0].strip()
                # Si après le tiret c'est vide ou juste des espaces
                interlocuteur = parts[1].strip() if len(parts) > 1 else ''
                return groupe, interlocuteur

        # Appliquer la fonction de split
        df_sales_clean[['Groupe', 'Interlocuteur']] = df_sales_clean['SoldTo Managed Group'].apply(
            lambda x: pd.Series(split_soldto(x))
        )

        # Supprimer les lignes où le Groupe est None
        df_sales_clean = df_sales_clean[df_sales_clean['Groupe'].notna()].copy()

        # Mettre en majuscules et nettoyer
        df_sales_clean['Groupe'] = df_sales_clean['Groupe'].astype(str).str.upper().str.strip()
        df_sales_clean['Interlocuteur'] = df_sales_clean['Interlocuteur'].fillna('').astype(str).str.upper().str.strip()

        # Convertir Posting Date de MM/DD/YYYY → datetime
        df_sales_clean['Posting Date'] = pd.to_datetime(df_sales_clean['Posting Date'], format='%m/%d/%Y', errors='coerce')

        # Supprimer les lignes avec dates invalides
        df_sales_clean = df_sales_clean.dropna(subset=['Posting Date'])
        
        # DEBUG: Afficher toutes les colonnes pour vérifier le nom exact
        print("Colonnes disponibles dans df_sales_clean:")
        print(df_sales_clean.columns.tolist())
        
        # Chercher la colonne de marge (peut avoir des espaces ou caractères invisibles)
        margin_column = None
        for col in df_sales_clean.columns:
            if 'margin' in col.lower() or 'marge' in col.lower():
                margin_column = col
                print(f"Colonne de marge trouvée: '{col}'")
                break
        
        # Conversion des marges
        if margin_column:
            # Multiplier par 100 toutes les valeurs non-nulles
            df_sales_clean[margin_column] = df_sales_clean[margin_column].apply(
                lambda x: x * 100 if pd.notna(x) else x
            )
            print(f"Marges converties - nouvelles valeurs: {df_sales_clean[margin_column].describe()}")
        else:
            print("ATTENTION: Aucune colonne de marge trouvée!")

        # ===========================
        # Prétraitement des clients
        # ===========================
        df_clients_clean = df_clients.copy()

        # Nettoyer les noms - gérer les valeurs manquantes avec chaîne vide
        df_clients_clean['Groupe'] = df_clients_clean['Groupe'].fillna('').astype(str).str.upper().str.strip()
        
        # Pour l'interlocuteur dans le fichier clients, gérer aussi les vides
        df_clients_clean['Interlocuteur'] = df_clients_clean['Interlocuteur'].fillna('').astype(str).str.upper().str.strip()
        
        # Appliquer fillna('').astype(str) à toutes les autres colonnes texte
        text_columns = ['Gestionnaire du compte', 'CONDITIONS D ACCORD', 'CONDITIONS DE PAIEMENT', 'REMISE DE FIN D ANNÉE']
        for col in text_columns:
            if col in df_clients_clean.columns:
                df_clients_clean[col] = df_clients_clean[col].fillna('').astype(str)

        # Colonnes de dates à vérifier (pas de .astype(str) pour les dates)
        date_columns = ['DATE DE DÉBUT', 'DATE DE RENOUVELLEMENT', 'DATE DE RENOUVELLEMENT 2', 'DATE DE FIN']
        for col in date_columns:
            if col in df_clients_clean.columns:
                # Tenter au format européen (jour/mois/année)
                df_clients_clean[col] = pd.to_datetime(df_clients_clean[col], dayfirst=True, errors='coerce')

                # Pour les valeurs non converties, retenter au format US (mois/jour/année)
                mask_null = df_clients_clean[col].isna()
                if mask_null.any():
                    df_clients_clean.loc[mask_null, col] = pd.to_datetime(
                        df_clients[col],
                        format='%m/%d/%Y',
                        errors='coerce'
                    )

        # Supprimer les lignes sans dates valides ou sans groupe
        df_clients_clean = df_clients_clean[df_clients_clean['Groupe'] != ''].copy()
        # Garder uniquement les lignes avec une date de début valide (pas besoin de date de fin)
        df_clients_clean = df_clients_clean.dropna(subset=['DATE DE DÉBUT'])
        
        # Gérer les objectifs NaN - remplacer par 0 pour permettre les calculs
        if 'OBJECTIF ATTENDU' in df_clients_clean.columns:
            df_clients_clean['OBJECTIF ATTENDU'] = df_clients_clean['OBJECTIF ATTENDU'].fillna(0)

        # =========================
        # Fusion et filtrage final
        # =========================
        # Debug: afficher les groupes uniques avant fusion
        print(f"Groupes uniques dans ventes: {df_sales_clean['Groupe'].nunique()}")
        print(f"Groupes uniques dans clients: {df_clients_clean['Groupe'].nunique()}")
        print(f"Groupes sans interlocuteur dans ventes: {len(df_sales_clean[df_sales_clean['Interlocuteur'] == ''])}")
        print(f"Groupes sans interlocuteur dans clients: {len(df_clients_clean[df_clients_clean['Interlocuteur'] == ''])}")
        
        df_merged = pd.merge(df_sales_clean, df_clients_clean, on=['Groupe', 'Interlocuteur'], how='inner')

        # Garder les lignes où Posting Date est dans la période du contrat
        # Si pas de date de fin, on garde toutes les ventes après la date de début
        mask_valid = df_merged['Posting Date'] >= df_merged['DATE DE DÉBUT']
        
        # Ajouter la condition de date de fin seulement si elle existe
        mask_with_end_date = df_merged['DATE DE FIN'].notna()
        mask_valid = mask_valid & (~mask_with_end_date | (df_merged['Posting Date'] <= df_merged['DATE DE FIN']))
        
        df_final = df_merged[mask_valid].copy()

        return df_final

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement des données : {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def calculate_adjusted_objective(objectif_annuel, date_debut, date_fin, date_reference=None):
    """
    Calcule l'objectif ajusté en fonction du nombre d'années entamées.
    Si date_reference n'est pas fournie, utilise la date actuelle.
    """
    if date_reference is None:
        date_reference = datetime.now()
    
    # Si pas de date de fin, calculer jusqu'à aujourd'hui
    if pd.isna(date_fin):
        date_calcul = date_reference
    else:
        # Si on est après la fin du contrat, utiliser la date de fin
        if date_reference > date_fin:
            date_calcul = date_fin
        else:
            date_calcul = date_reference
        
    # Calculer le nombre d'années entamées
    # On ajoute 1 car la première année compte même si elle n'est pas complète
    nb_annees_entamees = ((date_calcul - date_debut).days // 365) + 1
    
    # S'assurer qu'on ne dépasse pas la durée totale du contrat si date de fin existe
    if pd.notna(date_fin):
        duree_totale_contrat = ((date_fin - date_debut).days // 365) + 1
        nb_annees_entamees = min(nb_annees_entamees, duree_totale_contrat)
    
    return objectif_annuel * nb_annees_entamees

def is_contract_ending_soon(end_date, threshold_days=75):
    """Vérifier si le contrat se termine bientôt (moins de 2.5 mois)"""
    if pd.isna(end_date):
        return False
    return (end_date - datetime.now()).days <= threshold_days

def calculate_kpis(df):
    """Calculer les KPIs globaux basés uniquement sur les ventes pendant les contrats"""
    if df.empty:
        return {
            'nb_groupes': 0,
            'nb_commandes': 0,
            'nb_produits_uniques': 0,
            'objectif_total': 0,
            'ca_realise': 0,
            'pourcentage_ca': 0
        }
    
    # Calculer l'objectif total en tenant compte du nombre d'années écoulées pour chaque contrat
    objectif_total = 0
    
    # Grouper par Groupe et Interlocuteur pour avoir un contrat unique
    for (groupe, interlocuteur), group_df in df.groupby(['Groupe', 'Interlocuteur']):
        objectif_annuel = group_df['OBJECTIF ATTENDU'].iloc[0]
        
        # Si l'objectif est 0 ou NaN, on le saute
        if pd.isna(objectif_annuel) or objectif_annuel == 0:
            continue
            
        date_debut = group_df['DATE DE DÉBUT'].iloc[0]
        date_fin = group_df['DATE DE FIN'].iloc[0]
        
        # Calculer le nombre d'années entamées depuis le début du contrat
        today = datetime.now()
        
        # Si pas de date de fin, utiliser aujourd'hui
        if pd.isna(date_fin):
            date_reference = today
        else:
            # Si on est après la fin du contrat, utiliser la date de fin
            if today > date_fin:
                date_reference = date_fin
            else:
                date_reference = today
            
        # Calculer le nombre d'années entamées
        # On ajoute 1 car la première année compte même si elle n'est pas complète
        nb_annees_entamees = ((date_reference - date_debut).days // 365) + 1
        
        # S'assurer qu'on ne dépasse pas la durée totale du contrat si date de fin existe
        if pd.notna(date_fin):
            duree_totale_contrat = ((date_fin - date_debut).days // 365) + 1
            nb_annees_entamees = min(nb_annees_entamees, duree_totale_contrat)
        
        # Ajouter l'objectif cumulé pour ce contrat
        objectif_total += objectif_annuel * nb_annees_entamees
    
    # Tous les calculs sont basés sur les ventes filtrées par période de contrat
    return {
        'nb_groupes': df['Groupe'].nunique(),
        'nb_commandes': df['Sales Document #'].nunique(),
        'nb_produits_uniques': df['Material Y#'].nunique(),
        'objectif_total': objectif_total,
        'ca_realise': df['Customer Sales'].sum(),
        'pourcentage_ca': (df['Customer Sales'].sum() / objectif_total * 100) if objectif_total > 0 else 0
    }

def display_metric_card(label, value, delta=None, color="#2E86AB"):
    """Affiche une métrique dans une carte stylisée"""
    delta_html = ""
    if delta is not None:
        delta_class = "metric-delta-positive" if delta >= 0 else "metric-delta-negative"
        delta_symbol = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {abs(delta):.2f}%</div>'
    
    st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {color};">
            <div class="metric-label"><strong>{label}</strong></div>
            <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def display_kpis(kpis):
    """Afficher les KPIs dans des colonnes avec des cartes stylisées"""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        display_metric_card("Groupes sous contrat", f"{kpis['nb_groupes']:,}", color="#2E86AB")
    with col2:
        display_metric_card("Nombre de commandes", f"{kpis['nb_commandes']:,}", color="#48BB78")
    with col3:
        display_metric_card("Nbre de réferences", f"{kpis['nb_produits_uniques']:,}", color="#805AD5")
    with col4:
        display_metric_card("Objectif attendu", f"{kpis['objectif_total']:,.2f} €", color="#ED8936")
    with col5:
        display_metric_card("CA réalisé", f"{kpis['ca_realise']:,.2f} €", color="#38B2AC")
    with col6:
        delta = kpis['pourcentage_ca'] - 100 if kpis['pourcentage_ca'] > 0 else None
        color = "#10B981" if kpis['pourcentage_ca'] >= 100 else "#EF4444"
        display_metric_card("% CA / Objectif", f"{kpis['pourcentage_ca']:.2f}%", delta=delta, color=color)

def create_main_table(df):
    """Créer la table principale avec analyse par période de contrat de chaque groupe"""
    if df.empty:
        return pd.DataFrame()
    
    # Grouper par Groupe et Interlocuteur
    grouped = df.groupby(['Groupe', 'Interlocuteur']).agg({
        'DATE DE DÉBUT': 'first',
        'DATE DE FIN': 'first',
        'Customer Sales': 'sum',
        'OBJECTIF ATTENDU': 'first',
        'Sales Document #': 'nunique',
        'Material Y#': 'nunique',
        'Customer Margin %': 'mean'
    }).reset_index()
        
    # Calculer l'objectif ajusté et le pourcentage pour chaque groupe
    today = datetime.now()
    grouped['Objectif ajusté'] = grouped.apply(lambda row: calculate_adjusted_objective(
        row['OBJECTIF ATTENDU'], 
        row['DATE DE DÉBUT'], 
        row['DATE DE FIN'], 
        today
    ), axis=1)
    
    # Calculer le pourcentage CA/Objectif ajusté
    grouped['% CA / Objectif'] = (grouped['Customer Sales'] / grouped['Objectif ajusté'] * 100).round(2)
    
    # Renommer les colonnes (on garde l'objectif ajusté au lieu de l'objectif annuel)
    grouped.columns = ['Groupe', 'Interlocuteur', 'Date début', 'Date fin', 'CA réalisé', 
                      'Objectif annuel', 'Nb commandes', 'Nbre de références', 'Marge %', 'Objectif attendu', '% CA / Objectif']
    
    # Réorganiser les colonnes dans le bon ordre
    grouped = grouped[['Groupe', 'Interlocuteur', 'Date début', 'Date fin', 'CA réalisé', 
                      'Objectif attendu', 'Nb commandes', 'Nbre de références', 'Marge %', '% CA / Objectif']]
    
    return grouped

def create_products_table(df):
    """Créer la table des produits les plus commandés pendant les périodes de contrat"""
    if df.empty:
        return pd.DataFrame()
    
    # Produits commandés uniquement pendant les périodes de contrat
    # Grouper par Material Y# ET Material Entered #
    products = df.groupby(['Material Y#', 'Material Entered #', 'Material Entered Desc']).agg({
        'Sales Document #': 'nunique',
        'Customer Sales': 'sum',
        'Customer Margin %': 'mean'
    }).reset_index()
    
    products.columns = ['Material', 'MatEntered', 'Description produit', 'Nbre de commandes', 'CA total', 'Marge %']
    
    # Trier par CA total décroissant
    products = products.sort_values('CA total', ascending=False)
    
    return products

def create_product_family_table(df):
    """Créer la table des gammes de produits"""
    if df.empty or 'Product Family Desc' not in df.columns:
        return pd.DataFrame()
    
    # Grouper par gamme de produits
    family_stats = df.groupby('Product Family Desc').agg({
        'Customer Sales': 'sum',
        'Customer Margin %': 'mean',
        'Material Y#': 'nunique'
    }).reset_index()
    
    # Calculer le pourcentage
    family_stats['Pourcentage'] = (family_stats['Customer Sales'] / family_stats['Customer Sales'].sum() * 100).round(2)
    
    # Trier par CA décroissant
    family_stats = family_stats.sort_values('Customer Sales', ascending=False)
    
    # Renommer les colonnes pour l'affichage
    family_stats.columns = ['Gamme de produits', 'CA total', 'Marge Moy%', 'Nbre de références', '% du CA']
    
    # Réorganiser les colonnes
    family_stats = family_stats[['Gamme de produits', 'CA total', '% du CA', 'Marge Moy%', 'Nbre de références']]
    
    return family_stats

def style_family_table(df):
    """Styliser la table des gammes de produits avec des couleurs sophistiquées"""
    # Définir les styles pour chaque colonne
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Appliquer les couleurs colonne par colonne
    styles['Gamme de produits'] = 'background-color: #F3E5F5; color: #4A148C; font-weight: 500'
    styles['CA total'] = 'background-color: #E1F5FE; color: #01579B; font-weight: 500'
    styles['% du CA'] = 'background-color: #FFF3E0; color: #E65100; font-weight: 500'
    styles['Marge Moy%'] = 'background-color: #F1F8E9; color: #33691E; font-weight: 500'
    styles['Nbre de références'] = 'background-color: #FCE4EC; color: #880E4F; font-weight: 500'
    
    # Appliquer les styles et le formatage
    return df.style.apply(lambda x: styles, axis=None).format({
        'CA total': '{:,.2f} €',
        '% du CA': '{:.2f}%',
        'Marge Moy%': '{:.2f}%',
        'Nbre de références': '{:,}'
    })

# Interface principale
def main():
    # Gestion de l'état de la page
    if 'page' not in st.session_state:
        st.session_state.page = 'import'
    
    # Si données disponibles, afficher les options dans la sidebar
    if 'df_merged' in st.session_state:
        setup_sidebar()
    
    # Afficher la page appropriée
    if st.session_state.page == 'import' or 'df_merged' not in st.session_state:
        show_import_page()
    elif st.session_state.page == 'global':
        show_global_results()
    elif st.session_state.page == 'interlocuteur':
        show_interlocuteur_results()
    elif st.session_state.page == 'groupe':
        show_groupe_results()

def setup_sidebar():
    """Configuration de la sidebar avec filtres"""
    st.sidebar.markdown("### 🔍 Navigation & Filtres")
    
    # Bouton pour revenir à l'import
    if st.sidebar.button("🔄 Recommencer"):
        # Nettoyer les données de session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("#### 📍 Navigation")
    
    # Boutons de navigation
    if st.sidebar.button("📊 Vue Globale", use_container_width=True):
        st.session_state.page = 'global'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # FILTRE GLOBAL DE DATE
    st.sidebar.markdown("#### 📅 Filtre de Date Global")
    
    # Radio button pour le type de filtre
    filter_type = st.sidebar.radio(
        "Type de filtre:",
        ["Toute la période", "Année Fiscale", "Dates Personnalisées"],
        key="global_date_filter_type"
    )
    
    df = st.session_state['df_merged']
    
    if filter_type == "Année Fiscale":
        # Obtenir toutes les années fiscales disponibles dans les données
        all_fiscal_years = set()
        for (groupe, interlocuteur), group_df in df.groupby(['Groupe', 'Interlocuteur']):
            fiscal_years = get_fiscal_years_with_data(group_df)
            all_fiscal_years.update(fiscal_years)
        
        if all_fiscal_years:
            selected_fiscal = st.sidebar.selectbox(
                "Sélectionner une année fiscale",
                sorted(list(all_fiscal_years)),
                key="global_fiscal_year"
            )
    
    elif filter_type == "Dates Personnalisées":
        # Obtenir les limites globales des données
        min_date = df['Posting Date'].min()
        max_date = df['Posting Date'].max()
        
        # Date picker pour début
        custom_start = st.sidebar.date_input(
            "Date de début",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="global_custom_start_input",
            format="DD/MM/YYYY"
        )
        
        # Date picker pour fin
        custom_end = st.sidebar.date_input(
            "Date de fin",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="global_custom_end_input",
            format="DD/MM/YYYY"
        )
        
        # Stocker dans session_state
        st.session_state.global_custom_start = pd.to_datetime(custom_start)
        st.session_state.global_custom_end = pd.to_datetime(custom_end)
    
    st.sidebar.markdown("---")
    
    # Navigation par critère
    st.sidebar.markdown("#### 🔍 Filtres de Navigation")
    
    # Choix du mode de filtrage
    filter_mode = st.sidebar.radio(
        "Mode de filtrage:",
        ["Par Interlocuteur", "Par Groupe"],
        key="filter_mode"
    )
    
    if filter_mode == "Par Interlocuteur":
        # Filtre interlocuteur d'abord
        interlocuteurs = sorted(df['Interlocuteur'].unique().tolist())
        interlocuteurs_display = []
        for i in interlocuteurs:
            if i == '':
                interlocuteurs_display.append("(Sans interlocuteur)")
            else:
                interlocuteurs_display.append(i)
        
        selected_display = st.sidebar.selectbox(
            "📋 Sélectionner un interlocuteur",
            ["Choisir..."] + interlocuteurs_display,
            key="sidebar_interlocuteur_mode1"
        )
        
        if selected_display != "Choisir...":
            # Convertir le label d'affichage en valeur réelle
            if selected_display == "(Sans interlocuteur)":
                selected_interlocuteur = ''
            else:
                selected_interlocuteur = selected_display
                
            st.session_state.selected_interlocuteur = selected_interlocuteur
            st.session_state.selected_interlocuteur_display = selected_display
            
            # Bouton pour voir l'analyse de l'interlocuteur
            if st.sidebar.button("👤 Voir analyse interlocuteur", use_container_width=True):
                st.session_state.page = 'interlocuteur'
                st.rerun()
            
            # Filtre groupe (si interlocuteur sélectionné)
            df_filtered = df[df['Interlocuteur'] == selected_interlocuteur]
            groupes = sorted(df_filtered['Groupe'].unique().tolist())
            
            if len(groupes) > 0:
                selected_groupe = st.sidebar.selectbox(
                    "🏢 Sélectionner un groupe",
                    ["Choisir..."] + groupes,
                    key="sidebar_groupe_mode1"
                )
                
                if selected_groupe != "Choisir...":
                    st.session_state.selected_groupe = selected_groupe
                    
                    # Bouton pour voir l'analyse du groupe
                    if st.sidebar.button("🏢 Voir analyse groupe", use_container_width=True):
                        st.session_state.page = 'groupe'
                        st.rerun()
    
    else:  # Mode Par Groupe
        # Filtre groupe d'abord
        groupes = sorted(df['Groupe'].unique().tolist())
        
        selected_groupe = st.sidebar.selectbox(
            "🏢 Sélectionner un groupe",
            ["Choisir..."] + groupes,
            key="sidebar_groupe_mode2"
        )
        
        if selected_groupe != "Choisir...":
            st.session_state.selected_groupe = selected_groupe
            
            # Trouver automatiquement l'interlocuteur de ce groupe
            df_groupe_info = df[df['Groupe'] == selected_groupe]
            interlocuteur_unique = df_groupe_info['Interlocuteur'].unique()
            
            if len(interlocuteur_unique) == 1:
                # Un seul interlocuteur pour ce groupe
                interlocuteur = interlocuteur_unique[0]
                st.session_state.selected_interlocuteur = interlocuteur
                
                if interlocuteur == '':
                    st.sidebar.info("👤 Interlocuteur: (Sans interlocuteur)")
                    st.session_state.selected_interlocuteur_display = "(Sans interlocuteur)"
                else:
                    st.sidebar.info(f"👤 Interlocuteur: {interlocuteur}")
                    st.session_state.selected_interlocuteur_display = interlocuteur
            else:
                # Plusieurs interlocuteurs possibles (cas rare)
                interlocuteurs_display = []
                for i in interlocuteur_unique:
                    if i == '':
                        interlocuteurs_display.append("(Sans interlocuteur)")
                    else:
                        interlocuteurs_display.append(i)
                
                selected_inter_display = st.sidebar.selectbox(
                    "👤 Sélectionner l'interlocuteur",
                    interlocuteurs_display,
                    key="sidebar_interlocuteur_mode2"
                )
                
                if selected_inter_display == "(Sans interlocuteur)":
                    st.session_state.selected_interlocuteur = ''
                else:
                    st.session_state.selected_interlocuteur = selected_inter_display
                st.session_state.selected_interlocuteur_display = selected_inter_display
            
            # Bouton pour voir l'analyse du groupe
            if st.sidebar.button("🏢 Voir analyse groupe", use_container_width=True):
                st.session_state.page = 'groupe'
                st.rerun()

def show_import_page():
    """Page d'import des fichiers"""
    # Titre principal avec style
    st.markdown('<h2 class="main-header">📊 Suivi Commercial des Clients sous Contrat</h2>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <p style="font-size: 1.2rem; color: #666;">
            Bienvenue dans l'application d'analyse des ventes et contrats.<br>
            Veuillez importer les deux fichiers Excel requis pour commencer l'analyse.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conteneurs pour l'upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("##### 📁 Fichier Ventes")
        st.markdown("*Importez le fichier contenant les données de ventes*")
        sales_file = st.file_uploader("", type=['xlsx', 'xls'], key="sales", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("##### 📋 Fichier Clients")
        st.markdown("*Importez le fichier des clients sous contrat*")
        clients_file = st.file_uploader("", type=['xlsx', 'xls'], key="clients", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton de traitement centré
    if sales_file is not None and clients_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                process_files(sales_file, clients_file)

def process_files(sales_file, clients_file):
    """Traiter les fichiers importés"""
    try:
        # Charger les fichiers
        with st.spinner("⏳ Chargement des fichiers..."):
            df_sales = pd.read_excel(sales_file)
            df_clients = pd.read_excel(clients_file)
        
        # Sauvegarder dans session_state
        st.session_state['df_sales_raw'] = df_sales
        st.session_state['df_clients_raw'] = df_clients
        
        
        # Prétraitement et fusion
        with st.spinner("🔄 Traitement des données..."):
            df_merged = preprocess_and_merge_data(df_sales, df_clients)
            
            if df_merged.empty:
                st.error("❌ Aucune correspondance trouvée entre les ventes et les contrats clients")
                st.info("Vérifiez que les noms de groupes et interlocuteurs correspondent entre les deux fichiers")
                return

        
        # Sauvegarder les données traitées
        st.session_state['df_merged'] = df_merged
        st.session_state['df_clients'] = df_clients  # Pour les alertes
        st.session_state.page = 'global'
        
        st.success("✅ Fichiers chargés et traités avec succès!")
        
        # Rediriger vers la vue globale
        st.rerun()
                
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des fichiers: {str(e)}")
        import traceback
        with st.expander("Détails de l'erreur"):
            st.code(traceback.format_exc())

def show_global_results():
    """Afficher les résultats globaux"""
    st.markdown('<h3 class="main-header">📊 Vue Globale - Analyse des Résultats</h3>', unsafe_allow_html=True)
    
    df = st.session_state['df_merged']
    df_clients = st.session_state['df_clients_raw']
    
    # Appliquer le filtre global de date
    df_filtered = apply_global_date_filter(df)
    
    # Afficher information sur le filtre appliqué
    if 'global_date_filter_type' in st.session_state:
        filter_type = st.session_state.global_date_filter_type
        if filter_type == "Année Fiscale" and 'global_fiscal_year' in st.session_state:
            fiscal_year = st.session_state.global_fiscal_year
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Année fiscale {fiscal_year}</strong>
            </div>
            """, unsafe_allow_html=True)
        elif filter_type == "Dates Personnalisées" and 'global_custom_start' in st.session_state:
            start_date = st.session_state.global_custom_start
            end_date = st.session_state.global_custom_end
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Période du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Information sur la logique d'analyse
    st.info("📈 **Logique d'analyse** : Seules les ventes réalisées pendant la période de contrat de chaque groupe sont prises en compte.")
    
    # KPIs globaux
    st.markdown("<h4 style='color: #2E4057;'>📈 KPIs Globaux</h4>", unsafe_allow_html=True)
    kpis = calculate_kpis(df_filtered)
    display_kpis(kpis)
    
    # Alertes pour contrats se terminant bientôt
    df_clients_processed = pd.DataFrame()
    try:
        # Prétraiter les données clients pour les alertes
        df_clients_temp = df_clients.copy()
        df_clients_temp['DATE DE FIN'] = pd.to_datetime(df_clients_temp['DATE DE FIN'], dayfirst=True, errors='coerce')
        df_clients_processed = df_clients_temp.dropna(subset=['DATE DE FIN'])
    except:
        pass
    if not df_clients_processed.empty:
        ending_soon = df_clients_processed[df_clients_processed['DATE DE FIN'].apply(is_contract_ending_soon)]
        if not ending_soon.empty:
            st.markdown('<div class="alert-container danger-alert">', unsafe_allow_html=True)
            st.warning(f"🚨 **Alerte**: {len(ending_soon)} contrat(s) se termine(nt) dans moins de 2,5 mois")
            
            for _, row in ending_soon.iterrows():
                days_left = (row['DATE DE FIN'] - datetime.now()).days
                
                if days_left < 0:  # Déjà expiré
                    st.markdown(f'<div style="border-left: 5px solid red; padding-left: 15px; margin: 10px 0;">• <strong>{row["Groupe"]} -- {row.get("Interlocuteur", "Sans interlocuteur")}</strong>: Déjà expiré il y a {abs(days_left)} jours</div>', unsafe_allow_html=True)
                elif days_left == 0:  # Expire aujourd'hui
                    st.markdown(f'<div style="border-left: 5px solid red; padding-left: 15px; margin: 10px 0;">• <strong>{row["Groupe"]} -- {row.get("Interlocuteur", "Sans interlocuteur")}</strong>: Expire aujourd\'hui</div>', unsafe_allow_html=True)
                elif days_left <= 20:  # Expire dans 20 jours ou moins
                    st.markdown(f'<div style="border-left: 5px solid orange; padding-left: 15px; margin: 10px 0;">• <strong>{row["Groupe"]} -- {row.get("Interlocuteur", "Sans interlocuteur")}</strong>: {days_left} jours restants</div>', unsafe_allow_html=True)
                else:  # Plus de 20 jours
                    st.markdown(f'<div style="border-left: 5px solid gray; padding-left: 15px; margin: 10px 0;">• <strong>{row["Groupe"]} -- {row.get("Interlocuteur", "Sans interlocuteur")}</strong>: {days_left} jours restants</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Table principale
    st.markdown("<h4 style='color: #2E4057;'>📋 Table Principale des Contrats</h4>", unsafe_allow_html=True)
    main_table = create_main_table(df_filtered)
    
    if not main_table.empty:
        # Mise en forme conditionnelle
        def style_dataframe(df):
            def highlight_dates(val):
                if pd.isna(val):
                    return 'background-color: #059669; color: #FFFFFF; font-weight: bold'  # Vert foncé pour les contrats sans date de fin
                if isinstance(val, datetime):
                    if is_contract_ending_soon(val):
                        return 'background-color: #DC2626; color: #FFFFFF; font-weight: bold'  # Rouge foncé pour alertes
                    return 'background-color: #1E40AF; color: #FFFFFF; font-weight: bold'  # Bleu foncé pour dates normales
                return ''
            
            def apply_column_colors(s):
                """Appliquer des couleurs sophistiquées par colonne"""
                colors = {
                    'Groupe': 'background-color: #F0F4F8; color: #2D3748',
                    'Interlocuteur': 'background-color: #EDF2F7; color: #2D3748',
                    'Date début': 'background-color: #1E40AF; color: #FFFFFF; font-weight: bold',
                    'Date fin': 'background-color: #DC2626; color: #FFFFFF; font-weight: bold',
                    'CA réalisé': 'background-color: #F0FFF4; color: #1A4731',
                    'Objectif attendu': 'background-color: #FFF5F5; color: #63171B',
                    'Nb commandes': 'background-color: #EBF8FF; color: #2C5282',
                    'Nbre de références': 'background-color: #FAF5FF; color: #44337A',
                    'Marge %': 'background-color: #FFFAF0; color: #7B341E',
                    '% CA / Objectif': 'background-color: #F0FFF4; color: #22543D'
                }
                return [colors.get(s.name, '') for _ in s]
            
            # Appliquer d'abord les couleurs de colonnes
            styled_df = df.style.apply(apply_column_colors)
            # Puis appliquer les conditions spéciales pour les dates (cela écrasera les couleurs de base)
            styled_df = styled_df.map(highlight_dates, subset=['Date fin'])
            return styled_df.format({
                'CA réalisé': '{:,.2f} €',
                'Objectif attendu': '{:,.2f} €',
                'Marge %': '{:.2f}%',
                '% CA / Objectif': '{:.2f}%',
                'Date début': lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '',
                'Date fin': lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else ''
            })
        
        st.dataframe(style_dataframe(main_table), use_container_width=True, hide_index=True)
    
    # Table des produits
    st.markdown("<h4 style='color: #2E4057;'>🛍️ Tous les Produits</h4>", unsafe_allow_html=True)
    products_table = create_products_table(df_filtered)
    
    if not products_table.empty:
        def style_products_table(df):
            def apply_column_colors(s):
                colors = {
                    'Material': 'background-color: #FDF2E9; color: #7E5109',
                    'MatEntered': 'background-color: #FADBD8; color: #78281F',
                    'Description produit': 'background-color: #EAF2F8; color: #21618C',
                    'Nbre de commandes': 'background-color: #E8F5E9; color: #1B5E20',
                    'CA total': 'background-color: #E8F8F5; color: #0E6655',
                    'Marge %': 'background-color: #FEF9E7; color: #7D6608'
                }
                return [colors.get(s.name, '') for _ in s]
            
            styled_df = df.style.apply(apply_column_colors)
            return styled_df.format({
                'CA total': '{:,.2f} €',
                'Marge %': '{:.2f}%'
            })
        
        st.dataframe(style_products_table(products_table), use_container_width=True, hide_index=True)
    
    # Table des gammes de produits
    if 'Product Family Desc' in df_filtered.columns:
        st.markdown("<h4 style='color: #2E4057;'>📊 Analyse par Gamme de Produits</h4>", unsafe_allow_html=True)
        
        family_table = create_product_family_table(df_filtered)
        
        if not family_table.empty:
            st.dataframe(style_family_table(family_table), use_container_width=True, hide_index=True)

def show_interlocuteur_results():
    """Afficher les résultats pour un interlocuteur spécifique"""
    if 'selected_interlocuteur' not in st.session_state:
        st.session_state.page = 'global'
        st.rerun()
        
    selected_interlocuteur = st.session_state.selected_interlocuteur
    display_name = st.session_state.get('selected_interlocuteur_display', selected_interlocuteur)
    df = st.session_state['df_merged']
    df_clients = st.session_state['df_clients_raw']
    
    # Utiliser le nom d'affichage pour le titre
    if display_name == "(Sans interlocuteur)":
        st.markdown(f'<h3 class="main-header">👤 Analyse - Groupes sans interlocuteur</h3>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3 class="main-header">👤 Analyse - {display_name}</h3>', unsafe_allow_html=True)
    
    # Filtrer les données par interlocuteur puis appliquer le filtre global de date
    df_filtered = df[df['Interlocuteur'] == selected_interlocuteur]
    df_filtered = apply_global_date_filter(df_filtered)
    
    # Afficher information sur le filtre appliqué
    if 'global_date_filter_type' in st.session_state:
        filter_type = st.session_state.global_date_filter_type
        if filter_type == "Année Fiscale" and 'global_fiscal_year' in st.session_state:
            fiscal_year = st.session_state.global_fiscal_year
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Année fiscale {fiscal_year}</strong>
            </div>
            """, unsafe_allow_html=True)
        elif filter_type == "Dates Personnalisées" and 'global_custom_start' in st.session_state:
            start_date = st.session_state.global_custom_start
            end_date = st.session_state.global_custom_end
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Période du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # KPIs pour cet interlocuteur
    st.markdown("<h4 style='color: #2E4057;'>📈 KPIs de l'Interlocuteur</h4>", unsafe_allow_html=True)
    kpis_filtered = calculate_kpis(df_filtered)
    display_kpis(kpis_filtered)
    
    # Alerte si contrat proche de la fin
    try:
        df_clients_temp = df_clients.copy()
        df_clients_temp['Interlocuteur'] = df_clients_temp['Interlocuteur'].fillna('')
        df_clients_temp['Interlocuteur'] = df_clients_temp['Interlocuteur'].str.upper().str.strip()
        df_clients_temp['DATE DE FIN'] = pd.to_datetime(df_clients_temp['DATE DE FIN'], dayfirst=True, errors='coerce')
        client_info = df_clients_temp[df_clients_temp['Interlocuteur'] == selected_interlocuteur]
        
        if not client_info.empty:
            client_info = client_info.iloc[0]
            if pd.notna(client_info['DATE DE FIN']) and is_contract_ending_soon(client_info['DATE DE FIN']):
                days_left = (client_info['DATE DE FIN'] - datetime.now()).days
                st.error(f"🚨 Contrat se termine dans {days_left} jours!")
    except:
        pass
    
    # Table pour cet interlocuteur
    st.markdown("<h4 style='color: #2E4057;'>📋 Détails par Groupe</h4>", unsafe_allow_html=True)
    main_table_filtered = create_main_table(df_filtered)
    if not main_table_filtered.empty:
        # Mise en forme conditionnelle
        def style_dataframe(df):
            def highlight_dates(val):
                if pd.isna(val):
                    return 'background-color: #059669; color: #FFFFFF; font-weight: bold'  # Vert foncé pour les contrats sans date de fin
                if isinstance(val, datetime):
                    if is_contract_ending_soon(val):
                        return 'background-color: #DC2626; color: #FFFFFF; font-weight: bold'  # Rouge foncé pour alertes
                    return 'background-color: #1E40AF; color: #FFFFFF; font-weight: bold'  # Bleu foncé pour dates normales
                return ''
            
            def apply_column_colors(s):
                """Appliquer des couleurs sophistiquées par colonne"""
                colors = {
                    'Groupe': 'background-color: #F0F4F8; color: #2D3748',
                    'Interlocuteur': 'background-color: #EDF2F7; color: #2D3748',
                    'Date début': 'background-color: #1E40AF; color: #FFFFFF; font-weight: bold',
                    'Date fin': 'background-color: #DC2626; color: #FFFFFF; font-weight: bold',
                    'CA réalisé': 'background-color: #F0FFF4; color: #1A4731',
                    'Objectif attendu': 'background-color: #FFF5F5; color: #63171B',
                    'Nb commandes': 'background-color: #EBF8FF; color: #2C5282',
                    'Nbre de références': 'background-color: #FAF5FF; color: #44337A',
                    'Marge %': 'background-color: #FFFAF0; color: #7B341E',
                    '% CA / Objectif': 'background-color: #F0FFF4; color: #22543D'
                }
                return [colors.get(s.name, '') for _ in s]
            
            # Appliquer d'abord les couleurs de colonnes
            styled_df = df.style.apply(apply_column_colors)
            # Puis appliquer les conditions spéciales pour les dates (cela écrasera les couleurs de base)
            styled_df = styled_df.map(highlight_dates, subset=['Date fin'])
            return styled_df.format({
                'CA réalisé': '{:,.2f} €',
                'Objectif attendu': '{:,.2f} €',
                'Marge %': '{:.2f}%',
                '% CA / Objectif': '{:.2f}%',
                'Date début': lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '',
                'Date fin': lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else 'Contrat en cours'
            })
        
        st.dataframe(style_dataframe(main_table_filtered), use_container_width=True, hide_index=True)
    
    # Produits pour cet interlocuteur
    st.markdown("<h4 style='color: #2E4057;'>🛍️ Tous les produits</h4>", unsafe_allow_html=True)
    products_filtered = create_products_table(df_filtered)
    if not products_filtered.empty:
        def style_products_table(df):
            def apply_column_colors(s):
                colors = {
                    'Material': 'background-color: #FDF2E9; color: #7E5109',
                    'MatEntered': 'background-color: #FADBD8; color: #78281F',
                    'Description produit': 'background-color: #EAF2F8; color: #21618C',
                    'Nbre de commandes': 'background-color: #E8F5E9; color: #1B5E20',
                    'CA total': 'background-color: #E8F8F5; color: #0E6655',
                    'Marge %': 'background-color: #FEF9E7; color: #7D6608'
                }
                return [colors.get(s.name, '') for _ in s]
            
            styled_df = df.style.apply(apply_column_colors)
            return styled_df.format({
                'CA total': '{:,.2f} €',
                'Marge %': '{:.2f}%'
            })
        
        st.dataframe(style_products_table(products_filtered), use_container_width=True, hide_index=True)
    
    # Table des gammes de produits pour l'interlocuteur
    if 'Product Family Desc' in df_filtered.columns:
        st.markdown("<h4 style='color: #2E4057;'>📊 Analyse par Gamme de Produits</h4>", unsafe_allow_html=True)
        
        family_table = create_product_family_table(df_filtered)
        
        if not family_table.empty:
            st.dataframe(family_table.style.format({
                'CA total': '{:,.2f} €',
                '% du CA': '{:.1f}%',
                'Marge Moy%': '{:.1f}%',
                'Nbre de références': '{:,}'
            }), use_container_width=True, hide_index=True)

def show_groupe_results():
    """Afficher les résultats pour un groupe spécifique"""
    if 'selected_groupe' not in st.session_state or 'selected_interlocuteur' not in st.session_state:
        st.session_state.page = 'global'
        st.rerun()
    
    selected_groupe = st.session_state.selected_groupe
    selected_interlocuteur = st.session_state.selected_interlocuteur
    display_name = st.session_state.get('selected_interlocuteur_display', selected_interlocuteur)
    df = st.session_state['df_merged']
    df_clients = st.session_state['df_clients_raw']
    
    st.markdown(f'<h3 class="main-header">🏢 Analyse - {selected_groupe}</h3>', unsafe_allow_html=True)
    
    # Afficher l'interlocuteur de manière claire
    if display_name == "(Sans interlocuteur)":
        st.markdown("**Interlocuteur**: Sans interlocuteur")
    else:
        st.markdown(f"**Interlocuteur**: {display_name}")
    
    # Filtrer les données par groupe et interlocuteur puis appliquer le filtre global de date
    df_groupe = df[(df['Groupe'] == selected_groupe) & (df['Interlocuteur'] == selected_interlocuteur)]
    df_groupe_filtered = apply_global_date_filter(df_groupe)
    
    # Afficher les dates du contrat
    if not df_groupe.empty:
        contract_start = df_groupe['DATE DE DÉBUT'].iloc[0]
        contract_end = df_groupe['DATE DE FIN'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📅 **Date de début du contrat**: {contract_start.strftime('%d/%m/%Y')}")
        with col2:
            if pd.isna(contract_end):
                st.info(f"📅 **Date de fin du contrat**: Contrat toujours en cours")
            else:
                days_remaining = (contract_end - datetime.now()).days
                if days_remaining > 0:
                    st.info(f"📅 **Date de fin du contrat**: {contract_end.strftime('%d/%m/%Y')} ({days_remaining} jours restants)")
                else:
                    st.error(f"📅 **Date de fin du contrat**: {contract_end.strftime('%d/%m/%Y')} (Contrat expiré)")
    
    # Afficher information sur le filtre appliqué
    if 'global_date_filter_type' in st.session_state:
        filter_type = st.session_state.global_date_filter_type
        if filter_type == "Année Fiscale" and 'global_fiscal_year' in st.session_state:
            fiscal_year = st.session_state.global_fiscal_year
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Année fiscale {fiscal_year}</strong>
            </div>
            """, unsafe_allow_html=True)
        elif filter_type == "Dates Personnalisées" and 'global_custom_start' in st.session_state:
            start_date = st.session_state.global_custom_start
            end_date = st.session_state.global_custom_end
            st.markdown(f"""
            <div class="fiscal-year-info">
                <strong>📅 Filtre appliqué: Période du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Informations du gestionnaire et dates de renouvellement
    try:
        df_clients_temp = df_clients.copy()
        df_clients_temp['Groupe'] = df_clients_temp['Groupe'].str.upper().str.strip()
        df_clients_temp['Interlocuteur'] = df_clients_temp['Interlocuteur'].fillna('')
        df_clients_temp['Interlocuteur'] = df_clients_temp['Interlocuteur'].str.upper().str.strip()
        client_info = df_clients_temp[
            (df_clients_temp['Groupe'] == selected_groupe) & 
            (df_clients_temp['Interlocuteur'] == selected_interlocuteur)
        ]
        
        if not client_info.empty:
            client_info = client_info.iloc[0]
            if 'Gestionnaire du compte' in client_info:
                st.info(f"👨‍💼 **Gestionnaire du compte**: {client_info['Gestionnaire du compte']}")
            
            # Afficher les dates de renouvellement si elles existent
            renewal_dates = []
            if 'DATE DE RENOUVELLEMENT' in client_info and pd.notna(client_info['DATE DE RENOUVELLEMENT']):
                renewal_dates.append(('Date de renouvellement 1', client_info['DATE DE RENOUVELLEMENT']))
            if 'DATE DE RENOUVELLEMENT 2' in client_info and pd.notna(client_info['DATE DE RENOUVELLEMENT 2']):
                renewal_dates.append(('Date de renouvellement 2', client_info['DATE DE RENOUVELLEMENT 2']))
            
            if renewal_dates:
                st.markdown("### 📅 Dates de renouvellement")
                cols = st.columns(len(renewal_dates))
                for idx, (label, date) in enumerate(renewal_dates):
                    with cols[idx]:
                        st.info(f"**{label}**: {date.strftime('%d/%m/%Y')}")
            
            # Afficher l'objectif annuel du fichier clients
            if 'OBJECTIF ATTENDU' in client_info:
                objectif_annuel = client_info['OBJECTIF ATTENDU']
                if pd.notna(objectif_annuel) and objectif_annuel > 0:
                    st.info(f"🎯 **Objectif annuel**: {objectif_annuel:,.2f} €")
                else:
                    st.info(f"🎯 **Objectif annuel**: Non défini")
    except:
        pass
    
    # KPIs du groupe avec données filtrées
    st.markdown("<h4 style='color: #2E4057;'>📊 KPIs du Groupe</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        nb_clients = df_groupe_filtered['SoldTo #'].nunique()
        display_metric_card("Filiales", f"{nb_clients:,}", color="#3182CE")
    with col2:
        ca_total = df_groupe_filtered['Customer Sales'].sum()
        display_metric_card("CA réalisé", f"{ca_total:,.2f} €", color="#48BB78")
    with col3:
        # Pour l'objectif, utiliser la logique appropriée selon le filtre
        obj_annuel = df_groupe['OBJECTIF ATTENDU'].iloc[0] if not df_groupe.empty else 0
        
        if 'global_date_filter_type' in st.session_state:
            filter_type = st.session_state.global_date_filter_type
            if filter_type == "Dates Personnalisées":
                # Pour les dates personnalisées, calculer au prorata de la période sélectionnée
                custom_start = st.session_state.global_custom_start
                custom_end = st.session_state.global_custom_end
                nb_jours_periode = (custom_end - custom_start).days + 1
                obj = obj_annuel * (nb_jours_periode / 365)
            elif filter_type == "Année Fiscale":
                # Pour une année fiscale, utiliser l'objectif annuel tel quel
                obj = obj_annuel
            else:
                # Pour "Toute la période", calculer l'objectif cumulé basé sur les années entamées
                date_debut = df_groupe['DATE DE DÉBUT'].iloc[0]
                date_fin = df_groupe['DATE DE FIN'].iloc[0]
                obj = calculate_adjusted_objective(obj_annuel, date_debut, date_fin)
        else:
            # Fallback
            date_debut = df_groupe['DATE DE DÉBUT'].iloc[0]
            date_fin = df_groupe['DATE DE FIN'].iloc[0]
            obj = calculate_adjusted_objective(obj_annuel, date_debut, date_fin)
        
        display_metric_card("Objectif", f"{obj:,.2f} €", color="#ED8936")
    with col4:
        pct = (ca_total / obj * 100) if obj > 0 else 0
        delta = pct - 100 if pct > 0 else None
        color = "#10B981" if pct >= 100 else "#EF4444"
        display_metric_card("% Objectif", f"{pct:.2f}%", delta=delta, color=color)
    
    # Graphique d'évolution mensuelle si filtre de date appliqué
    if not df_groupe_filtered.empty and 'global_date_filter_type' in st.session_state and st.session_state.global_date_filter_type != "Toute la période":
        # Créer deux colonnes pour mieux organiser l'espace
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h4 style='color: #2E4057;'>📊 Évolution Mensuelle du CA</h4>", unsafe_allow_html=True)
            # Adapter le titre selon le filtre actif
            if st.session_state.global_date_filter_type == "Dates Personnalisées":
                period_title = "Période personnalisée"
            else:
                period_title = f"Année fiscale {st.session_state.get('global_fiscal_year', 'Toute la période')}"
            
            monthly_chart = create_monthly_sales_chart(df_groupe_filtered, period_title)
            if monthly_chart:
                st.plotly_chart(monthly_chart, use_container_width=True)
        
        with col2:
            # Ajouter des statistiques mensuelles
            if not df_groupe_filtered.empty:
                st.markdown("<h4 style='color: #2E4057;'>📈 Statistiques</h4>", unsafe_allow_html=True)
                
                # Calculer les stats mensuelles
                df_stats = df_groupe_filtered.copy()
                df_stats['Month'] = df_stats['Posting Date'].dt.strftime('%b %Y')
                monthly_ca = df_stats.groupby('Month')['Customer Sales'].sum()
                
                # Créer deux colonnes pour les stats
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    display_metric_card("CA Moyen/mois", f"{monthly_ca.mean():,.2f} €", color="#805AD5")
                    display_metric_card("Nb produits uniques", f"{df_groupe_filtered['Material Y#'].nunique():,}", color="#38B2AC")
                
                with stat_col2:
                    display_metric_card("Nb commandes", f"{df_groupe_filtered['Sales Document #'].nunique():,}", color="#F56565")
                    display_metric_card("Nb lignes", f"{len(df_groupe_filtered):,}", color="#4299E1")
    
    # Table des filiales avec données filtrées
    st.markdown("<h4 style='color: #2E4057;'>🏪 Filiales du groupe</h4>", unsafe_allow_html=True)
    filiales = df_groupe_filtered.groupby(['SoldTo #', 'SoldTo Name']).agg({
        'Sales Document #': 'nunique',
        'Customer Sales': 'sum',
        'SoldTo City': 'first'
    }).reset_index()
    
    if not filiales.empty:
        # Convertir SoldTo # en entier
        filiales['SoldTo #'] = filiales['SoldTo #'].astype(int)
        
        filiales['% CA'] = (filiales['Customer Sales'] / filiales['Customer Sales'].sum() * 100).round(2)
        filiales.columns = ['Filiale', 'Nom de filiale', 'Nb commandes', 'CA', 'Ville','% CA']
        # Réorganiser les colonnes pour mettre Ville après Nom de filiale
        filiales = filiales[['Filiale', 'Nom de filiale', 'Ville', 'Nb commandes', 'CA', '% CA']]
        
        def style_filiales_table(df):
            def apply_column_colors(s):
                colors = {
                    'Filiale': 'background-color: #E8EAF6; color: #283593',
                    'Nom de filiale': 'background-color: #E0F2F1; color: #004D40',
                    'Ville': 'background-color: #F3E5F5; color: #4A148C',  
                    'Nb commandes': 'background-color: #FFF8E1; color: #F57F17',
                    'CA': 'background-color: #E3F2FD; color: #0D47A1',
                    '% CA': 'background-color: #FFEBEE; color: #B71C1C'
                }
                return [colors.get(s.name, '') for _ in s]
            
            styled_df = df.style.apply(apply_column_colors)
            return styled_df.format({
                'Filiale': '{:d}',
                'CA': '{:,.2f} €',
                '% CA': '{:.2f}%'
            })
        
        st.dataframe(style_filiales_table(filiales), use_container_width=True, hide_index=True)
    
    # Conditions de contrat
    try:
        if 'client_info' in locals() and not client_info.empty:
            st.markdown("<h4 style='color: #2E4057;'>📋 Conditions de Contrat</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if 'CONDITIONS D ACCORD' in client_info and pd.notna(client_info['CONDITIONS D ACCORD']) and str(client_info['CONDITIONS D ACCORD']).strip() != '':
                    st.info(f"**Conditions d'accord**: {client_info['CONDITIONS D ACCORD']}")
                else:
                    st.info("**Conditions d'accord**: Pas de conditions")
            with col2:
                if 'CONDITIONS DE PAIEMENT' in client_info and pd.notna(client_info['CONDITIONS DE PAIEMENT']) and str(client_info['CONDITIONS DE PAIEMENT']).strip() != '':
                    st.info(f"**Conditions de paiement**: {client_info['CONDITIONS DE PAIEMENT']}")
                else:
                    st.info("**Conditions de paiement**: Pas de conditions")
            
            # Remise de fin d'année sur une nouvelle ligne
            if 'REMISE DE FIN D ANNÉE' in client_info and pd.notna(client_info['REMISE DE FIN D ANNÉE']) and str(client_info['REMISE DE FIN D ANNÉE']).strip() != '':
                remise_raw = str(client_info['REMISE DE FIN D ANNÉE']).strip()
                # Séparer par points-virgules et créer des puces
                remises = [r.strip() for r in remise_raw.split(';') if r.strip()]
                if remises:
                    # Créer le HTML avec le même style que st.info()
                    remise_lines = ["<strong>Remise de fin d'année</strong>:"]
                    for remise in remises:
                        remise_lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• {remise}")
                    remise_html = "<br>".join(remise_lines)
                    
                    # Utiliser le même style CSS que st.info()
                    st.markdown(f'''
                    <div style="
                        padding: 0.75rem 1rem;
                        margin: 0.25rem 0px;
                        border-radius: 0.5rem;
                        border: 1px solid rgb(188, 218, 252);
                        background-color: rgb(219, 234, 254);
                        color: rgb(12, 74, 110);
                    ">
                        {remise_html}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.info("**Remise de fin d'année**: Pas de remise")
            else:
                st.info("**Remise de fin d'année**: Pas de remise")
    except:
        pass
    
    # Produits du groupe avec données filtrées
    st.markdown("<h4 style='color: #2E4057;'>🛍️ Produits Commandés par ce Groupe</h4>", unsafe_allow_html=True)
    products_groupe = create_products_table(df_groupe_filtered)
    if not products_groupe.empty:
        def style_products_table(df):
            def apply_column_colors(s):
                colors = {
                    'Material': 'background-color: #FDF2E9; color: #7E5109',
                    'MatEntered': 'background-color: #FADBD8; color: #78281F',
                    'Description produit': 'background-color: #EAF2F8; color: #21618C',
                    'Nbre de commandes': 'background-color: #E8F5E9; color: #1B5E20',
                    'CA total': 'background-color: #E8F8F5; color: #0E6655',
                    'Marge %': 'background-color: #FEF9E7; color: #7D6608'
                }
                return [colors.get(s.name, '') for _ in s]
            
            styled_df = df.style.apply(apply_column_colors)
            return styled_df.format({
                'CA total': '{:,.2f} €',
                'Marge %': '{:.2f}%'
            })
        
        st.dataframe(style_products_table(products_groupe), use_container_width=True, hide_index=True)
    
    # Table des gammes de produits pour le groupe avec données filtrées
    if 'Product Family Desc' in df_groupe_filtered.columns:
        st.markdown("<h4 style='color: #2E4057;'>📊 Analyse par Gamme de Produits</h4>", unsafe_allow_html=True)
        
        family_table = create_product_family_table(df_groupe_filtered)
        
        if not family_table.empty:
            st.dataframe(family_table.style.format({
                'CA total': '{:,.2f} €',
                '% du CA': '{:.1f}%',
                'Marge Moy%': '{:.1f}%',
                'Nbre de références': '{:,}'
            }), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
