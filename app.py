# --- 1. IMPORT NECESSARI ---
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import Geocoder
from streamlit_folium import folium_static
from datetime import datetime
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from branca.colormap import linear
import os # --- NUOVO IMPORTO --- per controllare i percorsi dei file
import pydeck as pdk # --- NUOVO IMPORTO --- per le mappe 3D
import rasterio # --- NUOVO IMPORTO --- per leggere i file GeoTIFF
from rasterio.enums import Resampling
from PIL import Image
import io
import geopandas as gpd # <-- AGGIUNGI QUESTA RIGA QUI
from folium import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium



# --- 2. CONFIGURAZIONE CENTRALE E FUNZIONI DI BASE ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRxitMYpUqvX6bxVaukG01lJDC8SUfXtr47Zv5ekR1IzfR1jmhUilBsxZPJ8hrktVHrBh6hUUWYUtox/pub?output=csv"

COLONNE_FILTRO_RIEPILOGO = [
    "LEGENDA_TEMPERATURA_MEDIANA", "LEGENDA_PIOGGE_RESIDUA", "LEGENDA_MEDIA_PORCINI_CALDO_BASE", "LEGENDA_MEDIA_PORCINI_FREDDO_BASE",
    "LEGENDA_MEDIA_PORCINI_CALDO_ST_MIGLIORE", "LEGENDA_MEDIA_PORCINI_FREDDO_ST_MIGLIORE",
    "LEGENDA_MEDIA_PORCINI_CALDO_ST_SECONDO", "LEGENDA_MEDIA_PORCINI_FREDDO_ST_SECONDO"
]

def check_password():
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("password"):
            st.session_state["password_correct"] = True; st.session_state['just_logged_in'] = True; del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False): return True
    st.text_input("Inserisci la password per accedere:", type="password", on_change=password_entered, key="password")
    if "password" in st.session_state and st.session_state.get("password") and not st.session_state.get("password_correct"): st.error("😕 Password errata. Riprova.")
    if not st.session_state.get("password_correct"): st.stop()
    return False

@st.cache_resource
def get_view_counter(): return {"count": 0}

@st.cache_data(ttl=600)
def load_and_prepare_data(url: str):
    load_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    try:
        df = pd.read_csv(url, na_values=["#N/D", "#N/A"], dtype=str, header=0, skiprows=[1])
        cleaned_cols = {}
        for col in df.columns:
            temp_name = str(col).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", "")
            cleaned_name = temp_name.strip().replace(' ', '_').upper()
            cleaned_cols[col] = cleaned_name
        df.rename(columns=cleaned_cols, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

        for sbalzo_col, suffisso in [("LEGENDA_SBALZO_TERMICO_MIGLIORE", "MIGLIORE"), ("LEGENDA_SBALZO_TERMICO_SECONDO", "SECONDO")]:
            if sbalzo_col in df.columns:
                split_cols = df[sbalzo_col].str.split(' - ', n=1, expand=True)
                if split_cols.shape[1] == 2: df[f"LEGENDA_SBALZO_NUMERICO_{suffisso}"] = pd.to_numeric(split_cols[0].str.replace(',', '.'), errors='coerce')
        
        TEXT_COLUMNS = ['CODICE', 'STAZIONE', 'LEGENDA_DESCRIZIONE', 'LEGENDA_COMUNE', 'LEGENDA_COLORE', 'LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET', 'LEGENDA_SBALZO_TERMICO_MIGLIORE', 'LEGENDA_SBALZO_TERMICO_SECONDO', 'PORCINI_CALDO_NOTE', 'PORCINI_FREDDO_NOTE', 'SBALZO_TERMICO_MIGLIORE', '2°_SBALZO_TERMICO_MIGLIORE', 'LEGENDA']

        for col in df.columns:
            if col.strip().upper() == 'DATA':
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            elif col.strip().upper() not in TEXT_COLUMNS:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        
        if 'CODICE' not in df.columns or 'STAZIONE' not in df.columns:
            st.error("ERRORE CRITICO: Le colonne 'Codice' e/o 'Stazione' non sono presenti nel file Google Sheet.")
            st.stop()
            
        return df, load_timestamp
    except Exception as e:
        st.error(f"Errore critico durante il caricamento dei dati: {e}"); return None, None

def create_map(tile, location=[43.8, 11.0], zoom=8):
    return folium.Map(location=location, zoom_start=zoom, tiles=tile)


# --- NUOVA FUNZIONE HELPER PER CONVERTIRE GRADI IN DIREZIONE CARDINALE ---
def get_aspect_direction(degrees):
    if degrees is None or np.isnan(degrees): return "N/D"
    if degrees < 0: return "Piatto"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = int(np.round((degrees % 360) / 45))
    return dirs[index]

# --- NUOVA FUNZIONE PER CALCOLARE LA TEMPERATURA PRESUNTA ---
def calcola_temp_presunta(row, temp_base_stazione, alt_base_stazione):
    """
    Calcola una stima della temperatura mediana in un punto, basandosi su altitudine ed esposizione.
    """
    if pd.isna(row['elevation']) or pd.isna(row['aspect']):
        return None

    # 1. Correzione per l'altitudine (gradiente termico di -0.65°C ogni 100m)
    delta_altitudine = row['elevation'] - alt_base_stazione
    correzione_alt = (delta_altitudine / 100) * -0.65
    
    temp_corretta_alt = temp_base_stazione + correzione_alt

    # 2. Correzione per l'esposizione (modello semplificato)
    correzione_exp = 0
    aspect = row['aspect']
    if 135 <= aspect < 225: # Versanti SUD
        correzione_exp = 1.0 
    elif 45 <= aspect < 135: # Versanti EST
        correzione_exp = 0.2
    elif 225 <= aspect < 315: # Versanti OVEST
        correzione_exp = 0.2
    elif (315 <= aspect <= 360) or (0 <= aspect < 45): # Versanti NORD
        correzione_exp = -1.0
        
    return temp_corretta_alt + correzione_exp


# --- SOSTITUISCI QUESTA FUNZIONE ---
@st.cache_data
def load_multiband_data(station_code, target_points=40000):
    """
    Carica i dati dal file multibanda locale.
    Restituisce:
    1. DataFrame per il tooltip.
    2. I confini geografici (bounds).
    3. La matrice NumPy dell'altitudine, PULITA per la serializzazione JSON.
    """
    filepath = os.path.join("multibanda", f"{station_code}.tif")
    if not os.path.exists(filepath):
        st.error(f"File multibanda non trovato: {filepath}")
        return None, None, None

    with rasterio.open(filepath) as src:
        if src.count < 2:
            st.error(f"Il file {filepath} non è multibanda.")
            return None, None, None

        total_pixels = src.width * src.height
        if total_pixels > target_points:
            factor = (total_pixels / target_points) ** 0.5
            new_shape = (int(src.height / factor), int(src.width / factor))
        else:
            new_shape = (src.height, src.width)
        
        dem_data = src.read(1, out_shape=new_shape, resampling=Resampling.bilinear)
        aspect_data = src.read(2, out_shape=new_shape, resampling=Resampling.nearest)
        bounds = src.bounds

        nodata_val_dem = src.nodatavals[0]
        if nodata_val_dem is not None:
            dem_data[dem_data == nodata_val_dem] = np.nan
        
        nodata_val_aspect = src.nodatavals[1]
        if nodata_val_aspect is not None:
            aspect_data[aspect_data == nodata_val_aspect] = np.nan
            
        # Prepariamo il DataFrame per il tooltip (usa l'array con i NaN)
        height, width = dem_data.shape
        lons = np.linspace(bounds.left, bounds.right, width)
        lats = np.linspace(bounds.bottom, bounds.top, height)
        lons_grid, lats_grid = np.meshgrid(lons, lats)

        df = pd.DataFrame({
            'lon': lons_grid.flatten(), 'lat': lats_grid.flatten(),
            'elevation': dem_data.flatten(), 'aspect': aspect_data.flatten()
        })
        df.dropna(inplace=True)
        
        # --- MODIFICA CHIAVE PER L'ULTIMO TENTATIVO ---
        # Sostituiamo i dati mancanti (NaN) con 0.0.
        # È un valore numerico che JSON e Pydeck gestiscono senza problemi, a differenza di 'None'.
        elevation_for_terrain = np.nan_to_num(dem_data, nan=0.0)
        
        return df, bounds, elevation_for_terrain

# --- QUESTA FUNZIONE VA MESSA A LIVELLO GLOBALE (FUORI DA ALTRE FUNZIONI) ---
@st.cache_data
def get_gridded_data(station_code):
    """Carica e mette in cache i dati della griglia per una data stazione."""
    df_grid, _, _ = load_multiband_data(station_code, target_points=30000)
    return df_grid


# --- VERSIONE FINALE CON SELETTORE STREAMLIT (STABILE E COMPLETA) ---
def display_station_detail(df, station_code):
    # Assicurati che l'import corretto sia all'inizio del file:
    # from streamlit_folium import st_folium

    if st.button("⬅️ Torna alla Mappa Riepilogativa"):
        if 'last_clicked_point' in st.session_state:
            del st.session_state['last_clicked_point']
        st.query_params.clear()
        st.rerun()

    df_station = df[df['CODICE'] == station_code].sort_values('DATA').copy()
    if df_station.empty:
        st.error(f"Dati non trovati per la stazione: {station_code}.")
        return

    latest_station_data = df_station.iloc[-1]
    descriptive_name = latest_station_data['STAZIONE']
    station_lat = float(latest_station_data['LATITUDINE'])
    station_lon = float(latest_station_data['LONGITUDINE'])
    temp_base = latest_station_data.get('TEMPERATURA_MEDIANA')
    alt_base = latest_station_data.get('LEGENDA_ALTITUDINE')

    st.header(f"📈 Storico Dettagliato: {descriptive_name} ({station_code})")
    st.subheader("🌍 Mappa Interattiva del Territorio")
    
    # --- SELETTORE DEI LAYER FATTO CON STREAMLIT ---
    map_type = st.radio(
        "Seleziona stile mappa:",
        ("Mappa Stradale", "Mappa Topografica", "Mappa Satellitare"),
        horizontal=True,
        index=0 # Parte con "Mappa Stradale" selezionato
    )
    
    st.info("Clicca sulla mappa per vedere i dati stimati nel punto selezionato.")

    # --- LOGICA MAPPA ---
    # Creiamo una mappa di base usando il tile selezionato
    tile_to_use = "OpenStreetMap"
    if map_type == "Mappa Topografica":
        tile_to_use = 'https://tile.opentopomap.org/{z}/{x}/{y}.png'
    elif map_type == "Mappa Satellitare":
        tile_to_use = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    
    m = folium.Map(
        location=[station_lat, station_lon], 
        zoom_start=13, 
        tiles=tile_to_use, 
        attr='Map data © OpenStreetMap contributors' if 'OpenStreetMap' in str(tile_to_use) else ''
    )

    # Aggiungiamo i marker
    folium.Marker(
        [station_lat, station_lon],
        popup=f"<b>{descriptive_name}</b>", tooltip=descriptive_name,
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    if st.session_state.get('last_clicked_point'):
        folium.Marker(
            st.session_state['last_clicked_point'], 
            icon=folium.Icon(color="blue", icon="map-marker")
        ).add_to(m)
        
    # Mostriamo la mappa e catturiamo il click
    output = st_folium(m, width=950, height=480)

    # Logica per il click (invariata)
    if output and output["last_clicked"]:
        coords = [output["last_clicked"]["lat"], output["last_clicked"]["lng"]]
        if st.session_state.get('last_clicked_point') != coords:
            st.session_state['last_clicked_point'] = coords
            st.rerun()
    
    # Logica per mostrare i dati (invariata)
    if st.session_state.get('last_clicked_point'):
        lat_click, lon_click = st.session_state['last_clicked_point']
        if temp_base is not None and alt_base is not None:
            tooltip_df = get_gridded_data(station_code)
            if tooltip_df is not None and not tooltip_df.empty:
                tooltip_df["dist"] = np.sqrt((tooltip_df["lat"] - lat_click)**2 + (tooltip_df["lon"] - lon_click)**2)
                nearest = tooltip_df.loc[tooltip_df["dist"].idxmin()]
                aspect_direction = get_aspect_direction(nearest["aspect"])
                temp_presunta = calcola_temp_presunta(nearest, temp_base, alt_base)

                st.markdown(f"### 📍 Dati Stimati per il Punto Cliccato")
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Altitudine", f"{nearest['elevation']:.1f} m")
                with col2: st.metric("Esposizione", f"{aspect_direction}")
                with col3: st.metric("Temp. Presunta", f"{temp_presunta:.1f} °C" if temp_presunta is not None else "N/D")

    st.markdown("---")
    

    # --- I GRAFICI DELLO STORICO RIMANGONO INVARIATI ---
    config_chart = {'toImageButtonOptions': {'format': 'png', 'scale': 2, 'filename': f'grafico_{station_code}'}, 'displaylogo': False}
    end_date_default = df_station['DATA'].max()
    start_date_default = end_date_default - pd.Timedelta(days=39)


    st.subheader("Andamento Precipitazioni Giornaliere")
    fig1 = go.Figure(go.Bar(x=df_station['DATA'], y=df_station['TOTALE_PIOGGIA_GIORNO']))
    max_y_rain = df_station['TOTALE_PIOGGIA_GIORNO'].max() * 1.1 if not df_station['TOTALE_PIOGGIA_GIORNO'].empty else 100
    fig1.update_layout(title="Pioggia Giornaliera", xaxis_title="Data", yaxis_title="mm", xaxis_range=[start_date_default, end_date_default], yaxis_range=[0, max_y_rain])
    st.plotly_chart(fig1, use_container_width=True, config=config_chart)

    st.subheader("Correlazione Temperatura Mediana e Piogge Residue")
    cols_needed = ['PIOGGE_RESIDUA_ZOFFOLI', 'TEMPERATURA_MEDIANA']
    if all(c in df_station.columns for c in cols_needed) and not df_station.dropna(subset=cols_needed).empty:
        df_chart = df_station.dropna(subset=cols_needed)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df_chart['DATA'], y=df_chart['PIOGGE_RESIDUA_ZOFFOLI'], name='Piogge Residua', mode='lines', line=dict(color='blue')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df_chart['DATA'], y=df_chart['TEMPERATURA_MEDIANA'], name='Temperatura Mediana', mode='lines', line=dict(color='red')), secondary_y=True)
        max_y_rain_res = df_chart['PIOGGE_RESIDUA_ZOFFOLI'].max() * 1.1; min_y_rain_res = df_chart['PIOGGE_RESIDUA_ZOFFOLI'].min() * 0.9
        max_y_temp_med = df_chart['TEMPERATURA_MEDIANA'].max() * 1.1; min_y_temp_med = df_chart['TEMPERATURA_MEDIANA'].min() * 0.9
        fig2.update_yaxes(title_text="<b>Piogge Residua</b>", range=[min_y_rain_res, max_y_rain_res], secondary_y=False)
        fig2.update_yaxes(title_text="<b>Temperatura Mediana (°C)</b>", range=[min_y_temp_med, max_y_temp_med], secondary_y=True)
        fig2.update_layout(title_text="Temp vs Piogge", xaxis_range=[start_date_default, end_date_default])
        add_sbalzo_line(fig2, df_station, 'SBALZO_TERMICO_MIGLIORE', 'Sbalzo Migliore'); add_sbalzo_line(fig2, df_station, '2°_SBALZO_TERMICO_MIGLIORE', '2° Sbalzo')
        st.plotly_chart(fig2, use_container_width=True, config=config_chart)
    else: 
        st.warning("Dati di Piogge Residue o Temperatura Mediana non disponibili per creare il grafico.")

    st.subheader("Andamento Temperature Minime e Massime")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_station['DATA'], y=df_station['TEMP_MAX'], name='Temp Max', line=dict(color='orangered')))
    fig3.add_trace(go.Scatter(x=df_station['DATA'], y=df_station['TEMP_MIN'], name='Temp Min', line=dict(color='skyblue'), fill='tonexty'))
    max_y_temp = df_station['TEMP_MAX'].max() * 1.1 if not df_station['TEMP_MAX'].empty else 40
    min_y_temp = df_station['TEMP_MIN'].min() * 0.9 if not df_station['TEMP_MIN'].empty else -10
    fig3.update_layout(title="Escursione Termica Giornaliera", xaxis_title="Data", yaxis_title="°C", xaxis_range=[start_date_default, end_date_default], yaxis_range=[min_y_temp, max_y_temp])
    st.plotly_chart(fig3, use_container_width=True, config=config_chart)

    with st.expander("Visualizza tabella dati storici completi"):
        all_cols = sorted([c for c in df_station.columns if not c.startswith('LEGENDA_') and c not in ['LATITUDINE', 'LONGITUDINE', 'COORDINATEGOOGLE']])
        defaults = ['DATA', 'CODICE', 'STAZIONE', 'TOTALE_PIOGGIA_GIORNO', 'PIOGGE_RESIDUA_ZOFFOLI', 'TEMPERATURA_MEDIANA', 'TEMPERATURA_MEDIANA_MINIMA', 'SBALZO_TERMICO', 'UMIDITA_DEL_GIORNO', 'UMIDITA_MEDIA_7GG', 'VENTO', 'SBALZO_TERMICO_MIGLIORE', 'PORCINI_CALDO_NOTE', 'DURATA_RANGE_CALDO', 'CONTEGGIO_GG_RACCOLTA_CALDO','PORCINI_FREDDO_NOTE', 'DURATA_RANGE_FREDDO', 'BOOST', 'CONTEGGIO_GG_RACCOLTA_FREDDO']
        sel_cols = st.multiselect("Seleziona colonne:", options=all_cols, default=[c for c in defaults if c in all_cols])
        if sel_cols:
            display_df = df_station[sel_cols].sort_values('DATA', ascending=False)
            ordered_cols = [col for col in defaults if col in sel_cols] + [col for col in sel_cols if col not in defaults]
            st.markdown("""<style>div[data-testid="stDataFrame"] { overflow-x: auto; }</style>""", unsafe_allow_html=True)
            st.dataframe(display_df[ordered_cols])
        else:
            st.info("Seleziona almeno una colonna.")

# Aggiungo qui le funzioni che erano state omesse, per completezza del file
def display_main_map(df, last_loaded_ts):
    st.header("🗺️ Mappa Riepilogativa (Situazione Attuale)")
    df_with_valid_dates = df.dropna(subset=['DATA'])
    if df_with_valid_dates.empty:
        st.error("ERRORE: Non sono state trovate righe con date valide nel file.")
        return
    last_date = df_with_valid_dates['DATA'].max()
    df_latest = df_with_valid_dates[df_with_valid_dates['DATA'] == last_date].copy()
    st.info(f"Visualizzazione dati aggiornati al: **{last_date.strftime('%d/%m/%Y')}**")
    st.sidebar.title("Informazioni e Filtri Riepilogo"); st.sidebar.markdown("---")
    map_tile = st.sidebar.selectbox("Tipo di mappa:", ["OpenStreetMap", "CartoDB positron"], key="tile_main")
    st.sidebar.markdown("---"); st.sidebar.subheader("Statistiche")
    counter = get_view_counter(); st.sidebar.info(f"Visite totali: **{counter['count']}**")
    if last_loaded_ts: st.sidebar.info(f"App aggiornata il: **{last_loaded_ts}**")
    try:
        if 'LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET' in df_latest.columns and not df_latest['LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET'].empty:
            st.sidebar.info(f"Sheet aggiornato il: **{df_latest['LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET'].iloc[0]}**")
    except IndexError: pass
    st.sidebar.markdown("---"); st.sidebar.subheader("Filtri Dati Standard")
    df_filtrato = df_latest.copy()
    for colonna in COLONNE_FILTRO_RIEPILOGO:
        if colonna in df.columns and pd.to_numeric(df[colonna], errors='coerce').notna().any():
            max_val = float(pd.to_numeric(df[colonna], errors='coerce').max())
            slider_label = colonna.replace('LEGENDA_', '').replace('_', ' ').title()
            val_selezionato = st.sidebar.slider(f"Filtra per {slider_label}", min_value=0.0, max_value=max_val if max_val > 0 else 1.0, value=(0.0, max_val))
            if colonna in df_filtrato.columns:
                col_numerica = pd.to_numeric(df_filtrato[colonna], errors='coerce').fillna(0)
                df_filtrato = df_filtrato[col_numerica.between(val_selezionato[0], val_selezionato[1])]
    st.sidebar.markdown("---"); st.sidebar.subheader("Filtri Sbalzo Termico")
    for sbalzo_col, suffisso in [("LEGENDA_SBALZO_NUMERICO_MIGLIORE", "Migliore"), ("LEGENDA_SBALZO_NUMERICO_SECONDO", "Secondo")]:
        if sbalzo_col in df.columns and pd.to_numeric(df[sbalzo_col], errors='coerce').notna().any():
            max_val = float(pd.to_numeric(df[sbalzo_col], errors='coerce').max())
            val_selezionato = st.sidebar.slider(f"Sbalzo Termico {suffisso}", min_value=0.0, max_value=max_val if max_val > 0 else 1.0, value=(0.0, max_val))
            if sbalzo_col in df_filtrato.columns:
                col_numerica = pd.to_numeric(df_filtrato[sbalzo_col], errors='coerce').fillna(0)
                df_filtrato = df_filtrato[col_numerica.between(val_selezionato[0], val_selezionato[1])]
    st.sidebar.markdown("---"); st.sidebar.success(f"Visualizzati {len(df_filtrato)} marker sulla mappa.")
    df_mappa = df_filtrato.dropna(subset=['LATITUDINE', 'LONGITUDINE', 'CODICE']).copy()
    mappa = create_map(map_tile)
    Geocoder(collapsed=True, placeholder='Cerca un luogo...', add_marker=True).add_to(mappa)
    def create_popup_html(row):
        html = """<style>.popup-container{font-family:Arial,sans-serif;font-size:13px;max-height:350px;overflow-y:auto;overflow-x:hidden}h4{margin-top:12px;margin-bottom:5px;color:#0057e7;border-bottom:1px solid #ccc;padding-bottom:3px}table{width:100%;border-collapse:collapse;margin-bottom:10px}td{text-align:left;padding:4px;border-bottom:1px solid #eee}td:first-child{font-weight:bold;color:#333;width:65%}td:last-child{color:#555}.btn-container{text-align:center;margin-top:15px;}.btn{background-color:#007bff;color:white;padding:8px 12px;border-radius:5px;text-decoration:none;font-weight:bold;}</style><div class="popup-container">"""
        groups = {"Info Stazione": ["STAZIONE", "CODICE", "LEGENDA_DESCRIZIONE", "LEGENDA_COMUNE", "LEGENDA_ALTITUDINE"], "Dati Meteo": ["LEGENDA_TEMPERATURA_MEDIANA_MINIMA", "LEGENDA_TEMPERATURA_MEDIANA", "LEGENDA_UMIDITA_MEDIA_7GG", "LEGENDA_PIOGGE_RESIDUA", "LEGENDA_TOTALE_PIOGGE_MENSILI"], "Analisi Base": ["LEGENDA_MEDIA_PORCINI_CALDO_BASE", "LEGENDA_MEDIA_PORCINI_CALDO_BOOST", "LEGENDA_DURATA_RANGE_CALDO", "LEGENDA_CONTEGGIO_GG_ALLA_RACCOLTA_CALDO", "LEGENDA_MEDIA_PORCINI_FREDDO_BASE", "LEGENDA_MEDIA_PORCINI_FREDDO_BOOST", "LEGENDA_DURATA_RANGE_FREDDO", "LEGENDA_CONTEGGIO_GG_ALLA_RACCOLTA_FREDDO"], "Analisi Sbalzo Migliore": ["LEGENDA_SBALZO_TERMICO_MIGLIORE", "LEGENDA_MEDIA_PORCINI_CALDO_ST_MIGLIORE", "LEGENDA_MEDIA_BOOST_CALDO_ST_MIGLIORE", "LEGENDA_GG_ST_MIGLIORE_CALDO", "LEGENDA_MEDIA_PORCINI_FREDDO_ST_MIGLIORE", "LEGENDA_MEDIA_BOOST_FREDDO_ST_MIGLIORE", "LEGENDA_GG_ST_MIGLIORE_FREDDO"], "Analisi Sbalzo Secondo": ["LEGENDA_SBALZO_TERMICO_SECONDO", "LEGENDA_MEDIA_PORCINI_CALDO_ST_SECONDO", "LEGENDA_MEDIA_BOOST_CALDO_ST_SECONDO", "LEGENDA_GG_ST_SECONDO_CALDO", "LEGENDA_MEDIA_PORCini_FREDDO_ST_SECONDO", "LEGENDA_MEDIA_BOOST_FREDDO_ST_SECONDO", "LEGENDA_GG_ST_SECONDO_FREDDO"]}
        for title, columns in groups.items():
            table_html = "<table>"; has_content = False
            for col in columns:
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != '':
                    has_content = True; val = row[col]; label = col.replace('LEGENDA_', '').replace('_', ' ').title()
                    val_str = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if isinstance(val, (int, float)) else str(val)
                    table_html += f"<tr><td>{label}</td><td>{val_str}</td></tr>"
            table_html += "</table>"
            if has_content: html += f"<h4>{title}</h4>{table_html}"
        link = f'?station={row["CODICE"]}'
        html += f"<div class='btn-container'><a href='{link}' target='_self' class='btn'>📈 Mostra Storico Stazione</a></div></div>"
        return html
    def get_marker_color(val): 
        return {"ROSSO": "red", "GIALLO": "yellow", "ARANCIONE": "orange", "VERDE": "green"}.get(str(val).strip().upper(), "gray")
    for _, row in df_mappa.iterrows():
        try:
            lat, lon = float(row['LATITUDINE']), float(row['LONGITUDINE'])
            colore = get_marker_color(row.get('LEGENDA_COLORE', 'gray'))
            popup_html = create_popup_html(row)
            popup = folium.Popup(popup_html, max_width=380)
            tooltip_text = f"Stazione: {row['STAZIONE']} ({row['CODICE']})"
            folium.CircleMarker(location=[lat, lon], radius=6, color=colore, fill=True, fill_color=colore, fill_opacity=0.9, popup=popup, tooltip=tooltip_text).add_to(mappa)
        except (ValueError, TypeError): continue
    folium_static(mappa, width=1000, height=700)

def display_period_analysis(df):
    st.header("📊 Analisi di Periodo con Dati Aggregati")
    st.sidebar.title("Filtri di Periodo")
    map_tile = st.sidebar.selectbox("Tipo di mappa:", ["OpenStreetMap", "CartoDB positron"], key="tile_period")
    df_with_dates = df.dropna(subset=['DATA', 'CODICE'])
    if df_with_dates.empty:
        st.error("ERRORE: Dati insufficienti per l'analisi di periodo.")
        return
    min_date, max_date = df_with_dates['DATA'].min().date(), df_with_dates['DATA'].max().date()
    date_range = st.sidebar.date_input("Seleziona un periodo:", value=(max_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) != 2: 
        st.warning("Seleziona un intervallo di date valido."); st.stop()
    start_date, end_date = date_range
    df_filtered = df_with_dates[df_with_dates['DATA'].dt.date.between(start_date, end_date)]
    agg_cols = {'STAZIONE': 'first', 'TOTALE_PIOGGIA_GIORNO': 'sum', 'LATITUDINE': 'first', 'LONGITUDINE': 'first', 'TEMP_MAX': 'mean', 'TEMP_MIN': 'mean', 'TEMPERATURA_MEDIANA': 'mean'}
    df_agg = df_filtered.groupby('CODICE').agg(agg_cols).reset_index().dropna(subset=['LATITUDINE', 'LONGITUDINE'])
    df_agg.rename(columns={'TEMP_MAX': 'MEDIA_TEMP_MAX', 'TEMP_MIN': 'MEDIA_TEMP_MIN', 'TEMPERATURA_MEDIANA': 'MEDIA_TEMP_MEDIANA'}, inplace=True)
    df_agg_filtered = df_agg.copy()
    st.sidebar.subheader("Filtri Dati Aggregati")
    if not df_agg.empty:
        max_rain = float(df_agg['TOTALE_PIOGGIA_GIORNO'].max()) if not df_agg['TOTALE_PIOGGIA_GIORNO'].empty else 100.0
        rain_range = st.sidebar.slider("Pioggia Totale (mm)", 0.0, max_rain, (0.0, max_rain))
        max_tmax = float(df_agg['MEDIA_TEMP_MAX'].max()) if df_agg['MEDIA_TEMP_MAX'].notna().any() else 40.0
        tmax_range = st.sidebar.slider("Temp. Max Media (°C)", 0.0, max_tmax, (0.0, max_tmax))
        max_tmin = float(df_agg['MEDIA_TEMP_MIN'].max()) if df_agg['MEDIA_TEMP_MIN'].notna().any() else 30.0
        tmin_range = st.sidebar.slider("Temp. Min Media (°C)", -20.0, max_tmin, (-20.0, max_tmin))
        max_tmed = float(df_agg['MEDIA_TEMP_MEDIANA'].max()) if df_agg['MEDIA_TEMP_MEDIANA'].notna().any() else 35.0
        tmed_range = st.sidebar.slider("Temp. Mediana Media (°C)", 0.0, max_tmed, (0.0, max_tmed))
        df_agg_filtered = df_agg[df_agg['TOTALE_PIOGGIA_GIORNO'].between(rain_range[0], rain_range[1])]
        if 'MEDIA_TEMP_MAX' in df_agg_filtered.columns and df_agg_filtered['MEDIA_TEMP_MAX'].notna().any(): df_agg_filtered = df_agg_filtered[df_agg_filtered['MEDIA_TEMP_MAX'].between(tmax_range[0], tmax_range[1])]
        if 'MEDIA_TEMP_MIN' in df_agg_filtered.columns and df_agg_filtered['MEDIA_TEMP_MIN'].notna().any(): df_agg_filtered = df_agg_filtered[df_agg_filtered['MEDIA_TEMP_MIN'].between(tmin_range[0], tmin_range[1])]
        if 'MEDIA_TEMP_MEDIANA' in df_agg_filtered.columns and df_agg_filtered['MEDIA_TEMP_MEDIANA'].notna().any(): df_agg_filtered = df_agg_filtered[df_agg_filtered['MEDIA_TEMP_MEDIANA'].between(tmed_range[0], tmed_range[1])]
    st.info(f"Visualizzando **{len(df_agg_filtered)}** stazioni che corrispondono ai filtri.")
    map_center = [df_agg_filtered['LATITUDINE'].mean(), df_agg_filtered['LONGITUDINE'].mean()] if not df_agg_filtered.empty else [43.8, 11.0]
    mappa = create_map(map_tile, location=map_center, zoom=8)
    if df_agg_filtered.empty: 
        st.warning("Nessuna stazione corrisponde ai filtri selezionati.")
    else:
        min_rain, max_rain = df_agg_filtered['TOTALE_PIOGGIA_GIORNO'].min(), df_agg_filtered['TOTALE_PIOGGIA_GIORNO'].max()
        colormap = linear.YlGnBu_09.scale(vmin=min_rain, vmax=max_rain if max_rain > min_rain else min_rain + 1); colormap.caption = 'Totale Piogge (mm) nel Periodo'; mappa.add_child(colormap)
        for _, row in df_agg_filtered.iterrows():
            title_text = f"<b>{row['STAZIONE']} ({row['CODICE']})</b>"
            fig = go.Figure(go.Bar(x=['Pioggia Totale'], y=[row['TOTALE_PIOGGIA_GIORNO']], marker_color='#007bff', text=[f"{row['TOTALE_PIOGGIA_GIORNO']:.1f} mm"], textposition='auto'))
            fig.update_layout(title_text=title_text, title_font_size=14, yaxis_title="mm", width=250, height=200, margin=dict(l=40,r=20,t=40,b=20), showlegend=False)
            iframe = folium.IFrame(fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False}), width=280, height=220)
            popup = folium.Popup(iframe, max_width=300)
            lat, lon = float(row['LATITUDINE']), float(row['LONGITUDINE'])
            color = colormap(row['TOTALE_PIOGGIA_GIORNO'])
            tooltip_text = f"Stazione: {row['STAZIONE']} ({row['CODICE']})<br>Pioggia: {row['TOTALE_PIOGGIA_GIORNO']:.1f} mm<br>T.Max: {row.get('MEDIA_TEMP_MAX', 0.0):.1f}°C<br>T.Min: {row.get('MEDIA_TEMP_MIN', 0.0):.1f}°C"
            folium.CircleMarker(location=[lat, lon], radius=8, color=color, fill=True, fill_color=color, fill_opacity=0.7, popup=popup, tooltip=tooltip_text).add_to(mappa)
    folium_static(mappa, width=1000, height=700)
    with st.expander("Vedi dati aggregati filtrati"):
        if not df_agg_filtered.empty:
            df_display = df_agg_filtered.copy()
            df_display['link_storico'] = df_display['CODICE'].apply(lambda code: f"?station={code}")
            st.data_editor(df_display, column_config={"link_storico": st.column_config.LinkColumn("Link Storico", display_text="📈 Vedi Storico"), "LATITUDINE": None, "LONGITUDINE": None},
                           column_order=("CODICE", "STAZIONE", "link_storico", "TOTALE_PIOGGIA_GIORNO", "MEDIA_TEMP_MAX", "MEDIA_TEMP_MIN", "MEDIA_TEMP_MEDIANA"),
                           hide_index=True, disabled=True)
        else:
            st.write("Nessun dato da visualizzare in base ai filtri selezionati.")

def add_sbalzo_line(fig, df_data, sbalzo_col_name, label):
    if sbalzo_col_name not in df_data.columns: return
    df_valid_sbalzo = df_data.dropna(subset=[sbalzo_col_name])
    if df_valid_sbalzo.empty: return
    for _, row in df_valid_sbalzo.iterrows():
        sbalzo_str = str(row[sbalzo_col_name])
        if " - " in sbalzo_str:
            try:
                valore, data_str = sbalzo_str.split(" - ", 1)
                sbalzo_val = valore.strip().replace(",", "."); sbalzo_date = datetime.strptime(data_str.strip(), "%d/%m/%Y")
                fig.add_shape(type="line", x0=sbalzo_date, y0=0, x1=sbalzo_date, y1=1, line=dict(color="Green", width=2, dash="dash"), xref="x", yref="paper")
                fig.add_annotation(x=sbalzo_date, y=1.05, xref="x", yref="paper", text=f"{label} ({sbalzo_val})", showarrow=False, xanchor="left", font=dict(family="Arial", size=12, color="black"))
            except ValueError: continue

def main():
    st.set_page_config(page_title="Mappa Funghi Protetta", layout="wide")
    st.title("💧 Analisi Meteo Funghi – by Bobo 🍄")
    
    df, last_loaded_ts = load_and_prepare_data(SHEET_URL)
    if df is None or df.empty: 
        st.stop()
    
    query_params = st.query_params
    if "station" in query_params:
        display_station_detail(df, query_params["station"])
    else:
        if check_password():
            counter = get_view_counter()
            if st.session_state.get('just_logged_in', False): 
                counter["count"] += 1
                st.session_state['just_logged_in'] = False
            
            mode = st.radio("Seleziona la modalità:", ["Mappa Riepilogativa", "Analisi di Periodo"], horizontal=True)

            if mode == "Mappa Riepilogativa": 
                display_main_map(df, last_loaded_ts)
            elif mode == "Analisi di Periodo": 
                display_period_analysis(df)

if __name__ == "__main__":
    main()
