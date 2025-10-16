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

@st.cache_data(ttl=3600)
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

# --- NUOVA FUNZIONE HELPER per caricare e processare il DEM ---
@st.cache_data
# --- NUOVO IMPORTO NECESSARIO ---
from rasterio.enums import Resampling

# --- NUOVA FUNZIONE HELPER per caricare e processare il DEM (MODIFICATA) ---
@st.cache_data
def load_and_process_dem(station_code, target_points=30000):
    """
    Carica il file GeoTIFF e lo sottocampiona (downsamples) a un numero
    target di punti per renderlo leggero e veloce per il web.
    """
    filepath = os.path.join("ritagli", f"{station_code}.tif")
    if not os.path.exists(filepath):
        return None, None

    with rasterio.open(filepath) as src:
        # Calcola il fattore di downsampling per raggiungere il target_points
        total_pixels = src.width * src.height
        if total_pixels > target_points:
            factor = (total_pixels / target_points) ** 0.5
            new_width = int(src.width / factor)
            new_height = int(src.height / factor)
        else:
            new_width, new_height = src.width, src.height

        # Leggi i dati ricampionandoli a una risoluzione inferiore
        elevation_data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear  # Metodo di ricampionamento di buona qualità
        )
        bounds = src.bounds
        height, width = elevation_data.shape

    # Il resto della funzione rimane simile
    lons = np.linspace(bounds.left, bounds.right, width)
    lats = np.linspace(bounds.top, bounds.bottom, height) # Modificato per correttezza
    lons_grid, lats_grid = np.meshgrid(lons, lats)

    df_pydeck = pd.DataFrame({
        'lon': lons_grid.flatten(),
        'lat': lats_grid.flatten(),
        'elevation': elevation_data.flatten()
    })
    
    # Rimuovi eventuali valori nulli che rasterio può inserire ai bordi
    df_pydeck.dropna(inplace=True)

    cell_size_lon = (bounds.right - bounds.left) / width
    
    return df_pydeck, cell_size_lon

# --- SOSTITUISCI QUESTA INTERA FUNZIONE NEL TUO CODICE ---
def display_station_detail(df, station_code):
    if st.button("⬅️ Torna alla Mappa Riepilogativa"):
        st.query_params.clear()

    df_station = df[df['CODICE'] == station_code].sort_values('DATA').copy()
    
    if df_station.empty:
        st.error(f"Dati non trovati per la stazione con codice: {station_code}.")
        return

    descriptive_name = df_station['STAZIONE'].iloc[0]
    station_lat = float(df_station['LATITUDINE'].iloc[0])
    station_lon = float(df_station['LONGITUDINE'].iloc[0])

    st.header(f"📈 Storico Dettagliato: {descriptive_name} ({station_code})")

    # --- BLOCCO VISUALIZZAZIONE 3D (MODIFICATO) ---
    st.subheader("🌍 Visualizzazione 3D del Terreno Circostante")

    # Aggiungiamo un moltiplicatore per l'elevazione per renderla più visibile
    elevation_multiplier = st.slider("Accentua rilievo 3D", min_value=1.0, max_value=10.0, value=2.5, step=0.5)

    if st.button("🗺️ Avvia/Aggiorna Visualizzazione 3D del Terreno (20x20 km)"):
        with st.spinner("Caricamento dati altimetrici e rendering della mappa 3D..."):
            dem_df, cell_size_deg = load_and_process_dem(station_code)

            if dem_df is None:
                st.error(f"File DEM non trovato per la stazione {station_code}. Assicurati che 'ritagli/{station_code}.tif' esista.")
            else:
                view_state = pdk.ViewState(
                    latitude=station_lat,
                    longitude=station_lon,
                    zoom=11,
                    pitch=50,
                    bearing=0
                )
                
                # Converti la dimensione della cella da gradi a metri (approssimazione)
                # 1 grado di longitudine all'equatore ~ 111km. Lo adattiamo alla latitudine.
                cell_size_meters = cell_size_deg * 111000 * np.cos(np.radians(station_lat))

                terrain_layer = pdk.Layer(
                    'GridLayer',
                    data=dem_df,
                    get_position='[lon, lat]',
                    get_elevation='elevation',
                    elevation_scale=elevation_multiplier, # USA IL MOLTIPLICATORE
                    extruded=True,
                    cell_size=cell_size_meters,
                    pickable=True,
                    # Colore basato sull'altitudine (da verde basso a marrone/bianco alto)
                    color_range=[
                        [1, 152, 189],
                        [73, 227, 206],
                        [216, 254, 181],
                        [254, 237, 177],
                        [254, 173, 84],
                        [209, 55, 78]
                    ],
                )

                station_marker_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=pd.DataFrame([{'lat': station_lat, 'lon': station_lon}]),
                    get_position='[lon, lat]',
                    get_fill_color='[255, 0, 0, 255]',
                    get_radius=100,
                )

                tooltip = {
                    "html": "<b>Altitudine:</b> {elevation} m",
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }

                deck = pdk.Deck(
                    layers=[terrain_layer, station_marker_layer],
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/satellite-streets-v11',
                    tooltip=tooltip
                )
                st.pydeck_chart(deck)

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
