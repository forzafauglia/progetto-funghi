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
# --- 1.5. NUOVI IMPORT PER ANALISI GEOSPAZIALE 3D ---
import rasterio
from rasterio.windows import from_bounds
import pydeck as pdk
from pyproj import Proj, Transformer
import os

# --- 2. CONFIGURAZIONE CENTRALE E FUNZIONI DI BASE ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRxitMYpUqvX6bxVaukG01lJDC8SUfXtr47Zv5ekR1IzfR1jmhUilBsxZPJ8hrktVHrBh6hUUWYUtox/pub?output=csv"

# --- 2.5. NUOVE FUNZIONI PER ANALISI MICROCLIMATICA 3D ---
DEM_PATH = os.path.join("data", "dem_toscana.tif")
ASPECT_PATH = os.path.join("data", "aspect_toscana.tif")

@st.cache_data(ttl=86400) # Cache per 24 ore
def extract_raster_data(raster_path, center_lon, center_lat, size_m=10000):
    """Estrae un'area quadrata di dati da un file raster GeoTIFF."""
    try:
        with rasterio.open(raster_path) as src:
            # --- MODIFICA CHIAVE: SE IL CRS E' GIA' 4326, NON TRASFORMIAMO ---
            if src.crs.to_epsg() == 4326:
                # Calcoliamo l'ampiezza in gradi (approssimazione)
                # 1 grado di latitudine ~ 111 km. 1 grado di longitudine ~ 111km * cos(lat)
                deg_per_meter_lat = 1 / 111320
                deg_per_meter_lon = 1 / (111320 * np.cos(np.radians(center_lat)))
                
                half_size_deg_lon = (size_m / 2) * deg_per_meter_lon
                half_size_deg_lat = (size_m / 2) * deg_per_meter_lat

                bounds = (
                    center_lon - half_size_deg_lon, 
                    center_lat - half_size_deg_lat, 
                    center_lon + half_size_deg_lon, 
                    center_lat + half_size_deg_lat
                )
            else: # Manteniamo la vecchia logica per TIF in altri CRS
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                center_x, center_y = transformer.transform(center_lon, center_lat)
                half_size = size_m / 2
                bounds = (center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)

            window = from_bounds(*bounds, src.transform)
            data = src.read(1, window=window)
            window_transform = src.window_transform(window)
            return data, window_transform, src.crs
    except FileNotFoundError:
        st.error(f"ERRORE CRITICO: File non trovato in '{raster_path}'. Assicurati che il file esista nella sottocartella 'data'.")
        return None, None, None
    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file raster {os.path.basename(raster_path)}: {e}")
        return None, None, None

def estimate_temperature(base_temp, base_alt, target_alt, aspect_degrees):
    """Stima la temperatura in un punto basandosi su altitudine e esposizione."""
    if base_temp is None or base_alt is None or target_alt is None or aspect_degrees is None or target_alt < -100:
        return None

    # 1. Correzione per l'altitudine (gradiente termico verticale standard: -0.65°C ogni 100m)
    altitude_diff = target_alt - base_alt
    temp_correction_alt = (altitude_diff / 100) * 0.65
    estimated_temp = base_temp - temp_correction_alt

    # 2. Correzione euristica per l'esposizione (semplificata)
    if 315 < aspect_degrees <= 360 or 0 <= aspect_degrees <= 45:  # Nord
        estimated_temp -= 1.0
    elif 135 < aspect_degrees <= 225:  # Sud
        estimated_temp += 1.0
    
    return estimated_temp

def degrees_to_cardinal(d):
    """Converte i gradi di esposizione in punti cardinali."""
    if d is None or d < 0: return "N/D"
    dirs = ["N", "N-NE", "NE", "E-NE", "E", "E-SE", "SE", "S-SE", "S", "S-SW", "SW", "W-SW", "W", "W-NW", "NW", "N-NW"]
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

def create_pydeck_map(station_data, df_latest_station):
    # ... Tutta la parte iniziale di estrazione di altitudine e temperatura rimane IDENTICA ...
    st.subheader("🛰️ Analisi Microclimatica 3D del Territorio")
    st.info("Passa il mouse sulla mappa per esplorare. I dati vengono calcolati per un'area di 10x10 km intorno alla stazione.")
    station_lon = station_data['LONGITUDINE'].iloc[0]
    station_lat = station_data['LATITUDINE'].iloc[0]
    station_name = station_data['STAZIONE'].iloc[0]
    station_alt = None; 
    # (codice per trovare station_alt e latest_temp che hai già)
    if 'LEGENDA_ALTITUDINE' in df_latest_station.columns and not df_latest_station.empty:
        alt_val = df_latest_station['LEGENDA_ALTITUDINE'].iloc[0]
        if pd.notna(alt_val): station_alt = float(alt_val)
    if station_alt is None and 'QUOTA_M_SLM' in station_data.columns:
        first_valid_idx = station_data['QUOTA_M_SLM'].first_valid_index()
        if first_valid_idx is not None: station_alt = float(station_data['QUOTA_M_SLM'].loc[first_valid_idx])
    if station_alt is None:
        st.warning(f"Attenzione: Non è stato possibile trovare un valore di altitudine per la stazione {station_name}. Uso un valore di default di 500m.")
        station_alt = 500.0
    if not df_latest_station.empty and 'TEMPERATURA_MEDIANA' in df_latest_station and pd.notna(df_latest_station['TEMPERATURA_MEDIANA'].iloc[0]):
        latest_temp = df_latest_station['TEMPERATURA_MEDIANA'].iloc[0]
    else:
        st.warning(f"Attenzione: Temperatura dell'ultimo giorno non disponibile per la stazione {station_name}. Uso un valore di default di 15°C.")
        latest_temp = 15.0
    
    with st.spinner("Sto generando il modello 3D del terreno... potrebbe richiedere qualche secondo."):
        dem_data, dem_transform, raster_crs = extract_raster_data(DEM_PATH, station_lon, station_lat)
        aspect_data, _, _ = extract_raster_data(ASPECT_PATH, station_lon, station_lat)

    if dem_data is None or aspect_data is None: return

    h, w = dem_data.shape
    map_data = []
    step = 10 

    for r in range(0, h, step):
        for c in range(0, w, step):
            if dem_data[r, c] < -9000: continue

            # --- MODIFICA CHIAVE: NON TRASFORMIAMO PIU' SE IL CRS E' 4326 ---
            # Le coordinate che otteniamo da dem_transform sono GIA' lon e lat!
            lon, lat = dem_transform * (c + 0.5, r + 0.5)
            
            # Controllo di validità rimane utile
            if not (-180 <= lon <= 180 and -90 <= lat <= 90): continue
            
            alt = dem_data[r, c]
            aspect = aspect_data[r, c] if aspect_data is not None and aspect_data.shape == dem_data.shape else -1
            temp_est = estimate_temperature(latest_temp, station_alt, alt, aspect)
            
            if temp_est is not None:
                map_data.append({
                    "lon": lon, "lat": lat, "altitude": alt,
                    "aspect_str": degrees_to_cardinal(aspect),
                    "temp_est": temp_est
                })

    if not map_data:
        st.warning("Nessun dato valido trovato nell'area della stazione per generare la mappa 3D.")
        st.write(f"Informazioni tecniche: CRS del raster: {raster_crs}")
        return
        
    df_map = pd.DataFrame(map_data)

    # --- BLOCCO FINALE: ARROTONDAMENTO E STABILIZZAZIONE DATI ---
    # Arrotondiamo i valori numerici per un tooltip pulito
    df_map['altitude'] = df_map['altitude'].round(0).astype(int) # Arrotonda all'intero più vicino
    df_map['temp_est'] = df_map['temp_est'].round(1) # Arrotonda a 1 cifra decimale
    df_map['lon'] = df_map['lon'].round(5) # 5 decimali per coordinate sono sufficienti
    df_map['lat'] = df_map['lat'].round(5)
    # --- FINE BLOCCO FINALE ---

    # --- BLOCCO DI DEBUG (ora lo commentiamo, non serve più) ---
    # st.write("Dati processati per la mappa 3D (prime 10 righe):")
    # st.dataframe(df_map.head(10))
    # --- FINE BLOCCO DEBUG ---

    # NUOVO BLOCCO CORRETTO CON TILELAYER

    # 1. Definiamo il ViewState (come prima)
    view_state = pdk.ViewState(latitude=station_lat, longitude=station_lon, zoom=11, pitch=50, bearing=0)

    # 2. Definiamo il layer per la mappa di base (OpenTopoMap)
    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        min_zoom=1,
        max_zoom=16,
        tile_size=256,
        opacity=0.8  # Leggera trasparenza per vedere meglio le colonne
    )

    # 3. Definiamo il layer per le nostre colonne 3D (come prima)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_elevation="altitude",
        get_fill_color="[255, (1 - (temp_est - 5) / 25) * 255, 0, 180]", 
        elevation_scale=1,
        radius=50,
        pickable=True,
        extruded=True,
    )

    # 4. Definiamo il tooltip (come prima)
    tooltip = {
        "html": """
        <b>Punto Selezionato</b><br/>
        Lat: {lat}<br/>
        Lon: {lon}<br/>
        Alt: {altitude}<br/>
        Esposizione: {aspect_str}<br/>
        Temp. Stimata: {temp_est}
        """,
        "style": {"backgroundColor": "steelblue", "color": "white", "font-family": "Arial", "z-index": "10000"}
    }

    # 5. Creiamo l'oggetto Deck, combinando i due layer
    deck = pdk.Deck(
        layers=[tile_layer, column_layer],  # <-- PRIMA LA MAPPA, POI I DATI
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None # <-- IMPORTANTE: non usiamo più uno stile di base
    )

    # 6. Mostriamo la mappa (come prima)
    st.pydeck_chart(deck)
    st.caption(f"Simulazione basata sui dati dell'ultimo giorno disponibile: {latest_temp:.1f}°C a {station_alt:.0f}m (Stazione di {station_data['STAZIONE'].iloc[0]}).")

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

# --- FUNZIONE DI CARICAMENTO DATI CORRETTA E ROBUSTA ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(url: str):
    load_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    try:
        df = pd.read_csv(url, na_values=["#N/D", "#N/A"], dtype=str, header=0, skiprows=[1])
        
        # Pulizia nomi colonne
        cleaned_cols = {}
        for col in df.columns:
            temp_name = str(col).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("'", "")
            cleaned_name = temp_name.strip().replace(' ', '_').upper()
            cleaned_cols[col] = cleaned_name
        df.rename(columns=cleaned_cols, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]

        # Gestione sbalzo termico
        for sbalzo_col, suffisso in [("LEGENDA_SBALZO_TERMICO_MIGLIORE", "MIGLIORE"), ("LEGENDA_SBALZO_TERMICO_SECONDO", "SECONDO")]:
            if sbalzo_col in df.columns:
                split_cols = df[sbalzo_col].str.split(' - ', n=1, expand=True)
                if split_cols.shape[1] == 2: df[f"LEGENDA_SBALZO_NUMERICO_{suffisso}"] = pd.to_numeric(split_cols[0].str.replace(',', '.'), errors='coerce')
        
        TEXT_COLUMNS = ['STAZIONE', 'LEGENDA_DESCRIZIONE', 'LEGENDA_COMUNE', 'LEGENDA_COLORE', 'LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET', 'LEGENDA_SBALZO_TERMICO_MIGLIORE', 'LEGENDA_SBALZO_TERMICO_SECONDO', 'PORCINI_CALDO_NOTE', 'PORCINI_FREDDO_NOTE', 'SBALZO_TERMICO_MIGLIORE', '2°_SBALZO_TERMICO_MIGLIORE', 'LEGENDA']

        # Conversione tipi di dato
        for col in df.columns:
            if col.strip().upper() == 'DATA':
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            elif col.strip().upper() not in TEXT_COLUMNS:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        
        return df, load_timestamp
    except Exception as e:
        st.error(f"Errore critico durante il caricamento dei dati: {e}"); return None, None

def create_map(tile, location=[43.8, 11.0], zoom=8):
    return folium.Map(location=location, zoom_start=zoom, tiles=tile)

# SOSTITUISCI SOLO QUESTA FUNZIONE NEL TUO CODICE

# SOSTITUISCI SOLO QUESTA FUNZIONE NEL TUO CODICE

def display_main_map(df, last_loaded_ts):
    st.header("🗺️ Mappa Riepilogativa (Situazione Attuale)")
    
    df_with_valid_dates = df.dropna(subset=['DATA'])
    if df_with_valid_dates.empty:
        st.error("ERRORE: Non sono state trovate righe con date valide nel file.")
        return

    last_date = df_with_valid_dates['DATA'].max()
    df_latest = df_with_valid_dates[df_with_valid_dates['DATA'] == last_date].copy()
    st.info(f"Visualizzazione dati aggiornati al: **{last_date.strftime('%d/%m/%Y')}**")

    # --- Sidebar e Filtri (già corretti) ---
    st.sidebar.title("Informazioni e Filtri Riepilogo"); st.sidebar.markdown("---")
    map_tile = st.sidebar.selectbox("Tipo di mappa:", ["OpenStreetMap", "CartoDB positron"], key="tile_main")
    st.sidebar.markdown("---"); st.sidebar.subheader("Statistiche")
    counter = get_view_counter(); st.sidebar.info(f"Visite totali: **{counter['count']}**")
    if last_loaded_ts: st.sidebar.info(f"App aggiornata il: **{last_loaded_ts}**")
    try:
        if 'LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET' in df_latest.columns and not df_latest['LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET'].empty:
            st.sidebar.info(f"Sheet aggiornato il: **{df_latest['LEGENDA_ULTIMO_AGGIORNAMENTO_SHEET'].iloc[0]}**")
    except IndexError:
        pass

    st.sidebar.markdown("---"); st.sidebar.subheader("Filtri Dati Standard")
    df_filtrato = df_latest.copy()

    for colonna in COLONNE_FILTRO_RIEPILOGO:
        if colonna in df.columns and pd.to_numeric(df[colonna], errors='coerce').notna().any():
            max_val = float(pd.to_numeric(df[colonna], errors='coerce').max())
            slider_label = colonna.replace('LEGENDA_', '').replace('_', ' ').title()
            val_selezionato = st.sidebar.slider(f"Filtra per {slider_label}", min_value=0.0, max_value=max_val if max_val > 0 else 1.0, value=(0.0, max_val))
            if colonna in df_filtrato.columns:
                col_numerica = pd.to_numeric(df_filtrato[colonna], errors='coerce').fillna(0)
                # <<< QUI LA CORREZIONE: da val_sezionato a val_selezionato >>>
                df_filtrato = df_filtrato[col_numerica.between(val_selezionato[0], val_selezionato[1])]

    st.sidebar.markdown("---"); st.sidebar.subheader("Filtri Sbalzo Termico")
    for sbalzo_col, suffisso in [("LEGENDA_SBALZO_NUMERICO_MIGLIORE", "Migliore"), ("LEGENDA_SBALZO_NUMERICO_SECONDO", "Secondo")]:
        if sbalzo_col in df.columns and pd.to_numeric(df[sbalzo_col], errors='coerce').notna().any():
            max_val = float(pd.to_numeric(df[sbalzo_col], errors='coerce').max())
            val_selezionato = st.sidebar.slider(f"Sbalzo Termico {suffisso}", min_value=0.0, max_value=max_val if max_val > 0 else 1.0, value=(0.0, max_val))
            if sbalzo_col in df_filtrato.columns:
                col_numerica = pd.to_numeric(df_filtrato[sbalzo_col], errors='coerce').fillna(0)
                # <<< QUI LA CORREZIONE: da val_sezionato a val_selezionato >>>
                df_filtrato = df_filtrato[col_numerica.between(val_selezionato[0], val_selezionato[1])]
    
    st.sidebar.markdown("---"); st.sidebar.success(f"Visualizzati {len(df_filtrato)} marker sulla mappa.")
    df_mappa = df_filtrato.dropna(subset=['LATITUDINE', 'LONGITUDINE']).copy()
    
    mappa = create_map(map_tile)
    Geocoder(collapsed=True, placeholder='Cerca un luogo...', add_marker=True).add_to(mappa)

    def create_popup_html(row):
        html = """<style>.popup-container{font-family:Arial,sans-serif;font-size:13px;max-height:350px;overflow-y:auto;overflow-x:hidden}h4{margin-top:12px;margin-bottom:5px;color:#0057e7;border-bottom:1px solid #ccc;padding-bottom:3px}table{width:100%;border-collapse:collapse;margin-bottom:10px}td{text-align:left;padding:4px;border-bottom:1px solid #eee}td:first-child{font-weight:bold;color:#333;width:65%}td:last-child{color:#555}.btn-container{text-align:center;margin-top:15px;}.btn{background-color:#007bff;color:white;padding:8px 12px;border-radius:5px;text-decoration:none;font-weight:bold;}</style><div class="popup-container">"""
        groups = {"Info Stazione": ["STAZIONE", "LEGENDA_DESCRIZIONE", "LEGENDA_COMUNE", "LEGENDA_ALTITUDINE"], "Dati Meteo": ["LEGENDA_TEMPERATURA_MEDIANA_MINIMA", "LEGENDA_TEMPERATURA_MEDIANA", "LEGENDA_UMIDITA_MEDIA_7GG", "LEGENDA_PIOGGE_RESIDUA", "LEGENDA_TOTALE_PIOGGE_MENSILI"], "Analisi Base": ["LEGENDA_MEDIA_PORCINI_CALDO_BASE", "LEGENDA_MEDIA_PORCINI_CALDO_BOOST", "LEGENDA_DURATA_RANGE_CALDO", "LEGENDA_CONTEGGIO_GG_ALLA_RACCOLTA_CALDO", "LEGENDA_MEDIA_PORCINI_FREDDO_BASE", "LEGENDA_MEDIA_PORCINI_FREDDO_BOOST", "LEGENDA_DURATA_RANGE_FREDDO", "LEGENDA_CONTEGGIO_GG_ALLA_RACCOLTA_FREDDO"], "Analisi Sbalzo Migliore": ["LEGENDA_SBALZO_TERMICO_MIGLIORE", "LEGENDA_MEDIA_PORCINI_CALDO_ST_MIGLIORE", "LEGENDA_MEDIA_BOOST_CALDO_ST_MIGLIORE", "LEGENDA_GG_ST_MIGLIORE_CALDO", "LEGENDA_MEDIA_PORCINI_FREDDO_ST_MIGLIORE", "LEGENDA_MEDIA_BOOST_FREDDO_ST_MIGLIORE", "LEGENDA_GG_ST_MIGLIORE_FREDDO"], "Analisi Sbalzo Secondo": ["LEGENDA_SBALZO_TERMICO_SECONDO", "LEGENDA_MEDIA_PORCINI_CALDO_ST_SECONDO", "LEGENDA_MEDIA_BOOST_CALDO_ST_SECONDO", "LEGENDA_GG_ST_SECONDO_CALDO", "LEGENDA_MEDIA_PORCini_FREDDO_ST_SECONDO", "LEGENDA_MEDIA_BOOST_FREDDO_ST_SECONDO", "LEGENDA_GG_ST_SECONDO_FREDDO"]}
        for title, columns in groups.items():
            table_html = "<table>"; has_content = False
            for col in columns:
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != '':
                    has_content = True; val = row[col]; label = col.replace('LEGENDA_', '').replace('_', ' ').title()
                    val_str = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if isinstance(val, (int, float)) else str(val)
                    table_html += f"<tr><td>{label}</td><td>{val_str}</td></tr>"
            table_html += "</table>"
            if has_content: html += f"<h4>{title}</h4>{table_html}"
        link = f'?station={row["STAZIONE"]}'; html += f"<div class='btn-container'><a href='{link}' target='_self' class='btn'>📈 Mostra Storico Stazione</a></div></div>"
        return html

    def get_marker_color(val): 
        return {"ROSSO": "red", "GIALLO": "yellow", "ARANCIONE": "orange", "VERDE": "green"}.get(str(val).strip().upper(), "gray")
    
    for _, row in df_mappa.iterrows():
        try:
            lat, lon = float(row['LATITUDINE']), float(row['LONGITUDINE'])
            colore = get_marker_color(row.get('LEGENDA_COLORE', 'gray'))
            
            popup_html = create_popup_html(row)
            popup = folium.Popup(popup_html, max_width=380)

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=colore,
                fill=True,
                fill_color=colore,
                fill_opacity=0.9,
                popup=popup,
                tooltip=f"Stazione: {row['STAZIONE']}"
            ).add_to(mappa)
        except (ValueError, TypeError):
            continue
            
    folium_static(mappa, width=1000, height=700)

def display_period_analysis(df):
    st.header("📊 Analisi di Periodo con Dati Aggregati")
    st.sidebar.title("Filtri di Periodo")
    map_tile = st.sidebar.selectbox("Tipo di mappa:", ["OpenStreetMap", "CartoDB positron"], key="tile_period")

    df_with_dates = df.dropna(subset=['DATA'])

    if df_with_dates.empty:
        st.error("ERRORE: Non sono state trovate date valide nel file. Impossibile eseguire l'analisi di periodo.")
        st.warning("Controlla la colonna 'DATA' nel tuo Google Sheet.")
        return

    min_date, max_date = df_with_dates['DATA'].min().date(), df_with_dates['DATA'].max().date()
    
    date_range = st.sidebar.date_input("Seleziona un periodo:", value=(max_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) != 2: 
        st.warning("Seleziona un intervallo di date valido.")
        st.stop()
    
    start_date, end_date = date_range
    df_filtered = df_with_dates[df_with_dates['DATA'].dt.date.between(start_date, end_date)]
    
    agg_cols = {'TOTALE_PIOGGIA_GIORNO': 'sum', 'LATITUDINE': 'first', 'LONGITUDINE': 'first', 'TEMP_MAX': 'mean', 'TEMP_MIN': 'mean', 'TEMPERATURA_MEDIANA': 'mean'}
    df_agg = df_filtered.groupby('STAZIONE').agg(agg_cols).reset_index().dropna(subset=['LATITUDINE', 'LONGITUDINE'])
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
    
    if not df_agg_filtered.empty: map_center = [df_agg_filtered['LATITUDINE'].mean(), df_agg_filtered['LONGITUDINE'].mean()]
    else: map_center = [43.8, 11.0]
    mappa = create_map(map_tile, location=map_center, zoom=8)
    
    if df_agg_filtered.empty: 
        st.warning("Nessuna stazione corrisponde ai filtri selezionati.")
    else:
        min_rain, max_rain = df_agg_filtered['TOTALE_PIOGGIA_GIORNO'].min(), df_agg_filtered['TOTALE_PIOGGIA_GIORNO'].max()
        colormap = linear.YlGnBu_09.scale(vmin=min_rain, vmax=max_rain if max_rain > min_rain else min_rain + 1); colormap.caption = 'Totale Piogge (mm) nel Periodo'; mappa.add_child(colormap)
        
        for _, row in df_agg_filtered.iterrows():
            fig = go.Figure(go.Bar(x=['Pioggia Totale'], y=[row['TOTALE_PIOGGIA_GIORNO']], marker_color='#007bff', text=[f"{row['TOTALE_PIOGGIA_GIORNO']:.1f} mm"], textposition='auto'))
            fig.update_layout(title_text=f"<b>{row['STAZIONE']}</b>", title_font_size=14, yaxis_title="mm", width=250, height=200, margin=dict(l=40,r=20,t=40,b=20), showlegend=False)
            config={'displayModeBar': False}; html_chart = fig.to_html(full_html=False, include_plotlyjs='cdn', config=config)
            
            full_html_popup = f"<div>{html_chart}</div>"
            iframe = folium.IFrame(full_html_popup, width=280, height=220) 
            popup = folium.Popup(iframe, max_width=300, parse_html=True)
            
            lat, lon = float(row['LATITUDINE']), float(row['LONGITUDINE'])
            color = colormap(row['TOTALE_PIOGGIA_GIORNO'])
            tooltip_text = (f"Stazione: {row['STAZIONE']}<br>Pioggia: {row['TOTALE_PIOGGIA_GIORNO']:.1f} mm<br>T.Max: {row.get('MEDIA_TEMP_MAX', 0.0):.1f}°C<br>T.Min: {row.get('MEDIA_TEMP_MIN', 0.0):.1f}°C")
            folium.CircleMarker(location=[lat, lon], radius=8, color=color, fill=True, fill_color=color, fill_opacity=0.7, popup=popup, tooltip=tooltip_text).add_to(mappa)
            
    folium_static(mappa, width=1000, height=700)
    
    with st.expander("Vedi dati aggregati filtrati"):
        if not df_agg_filtered.empty:
            df_display = df_agg_filtered.copy()
            df_display['link_storico'] = df_display['STAZIONE'].apply(lambda name: f"?station={name}")
            st.data_editor(df_display, column_config={"link_storico": st.column_config.LinkColumn("Link Storico", display_text="📈 Vedi Storico", help="Clicca per aprire lo storico dettagliato della stazione"),"LATITUDINE": None,"LONGITUDINE": None}, column_order=("STAZIONE", "link_storico", "TOTALE_PIOGGIA_GIORNO", "MEDIA_TEMP_MAX", "MEDIA_TEMP_MIN", "MEDIA_TEMP_MEDIANA"), hide_index=True, disabled=True)
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

def display_station_detail(df, station_name):
    if st.button("⬅️ Torna alla Mappa Riepilogativa"): 
        st.query_params.clear()

    st.header(f"📈 Storico Dettagliato: {station_name}")
    df_station = df[df['STAZIONE'] == station_name].sort_values('DATA').copy()
    
    if df_station.empty: 
        st.error("Dati non trovati per la stazione selezionata.")
        return
        
    # Ottieni l'ultimo dato disponibile per la stazione per la simulazione
    df_latest_station = df_station[df_station['DATA'] == df_station['DATA'].max()]

    # --- INIZIO NUOVA SEZIONE: ANALISI 3D ---
    with st.expander("🔬 Apri l'Analisi Microclimatica 3D del Territorio"):
        if st.button("🚀 Avvia Simulazione 3D", key="start_3d_sim"):
            # Verifica che i file TIF esistano prima di procedere
            if os.path.exists(DEM_PATH) and os.path.exists(ASPECT_PATH):
                create_pydeck_map(df_station, df_latest_station)
            else:
                st.error("File di dati geografici non trovati. Assicurati che 'dem_toscana.tif' e 'aspect_toscana.tif' siano nella cartella 'data'.")
    # --- FINE NUOVA SEZIONE: ANALISI 3D ---


    # Impostazioni dei grafici
    if not df_station.empty: 
        end_date_default = df_station['DATA'].max()
        start_date_default = end_date_default - pd.Timedelta(days=39)
    else: 
        end_date_default = datetime.now()
        start_date_default = end_date_default - pd.Timedelta(days=39)
    
    config_chart = {'toImageButtonOptions': {'format': 'png', 'scale': 2, 'filename': f'grafico_{station_name}'}, 'displaylogo': False}

    # --- Grafico 1: Precipitazioni ---
    st.subheader("Andamento Precipitazioni Giornaliere")
    fig1 = go.Figure(go.Bar(x=df_station['DATA'], y=df_station['TOTALE_PIOGGIA_GIORNO']))
    max_y_rain = df_station['TOTALE_PIOGGIA_GIORNO'].max() * 1.1 if not df_station['TOTALE_PIOGGIA_GIORNO'].empty else 100
    fig1.update_layout(title="Pioggia Giornaliera", xaxis_title="Data", yaxis_title="mm", xaxis_range=[start_date_default, end_date_default], yaxis_range=[0, max_y_rain])
    st.plotly_chart(fig1, use_container_width=True, config=config_chart)

    # --- Grafico 2: Temperatura vs Piogge Residue ---
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

    # --- Grafico 3: Temperature Min/Max ---
    st.subheader("Andamento Temperature Minime e Massime")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_station['DATA'], y=df_station['TEMP_MAX'], name='Temp Max', line=dict(color='orangered')))
    fig3.add_trace(go.Scatter(x=df_station['DATA'], y=df_station['TEMP_MIN'], name='Temp Min', line=dict(color='skyblue'), fill='tonexty'))
    max_y_temp = df_station['TEMP_MAX'].max() * 1.1 if not df_station['TEMP_MAX'].empty else 40
    min_y_temp = df_station['TEMP_MIN'].min() * 0.9 if not df_station['TEMP_MIN'].empty else -10
    fig3.update_layout(title="Escursione Termica Giornaliera", xaxis_title="Data", yaxis_title="°C", xaxis_range=[start_date_default, end_date_default], yaxis_range=[min_y_temp, max_y_temp])
    st.plotly_chart(fig3, use_container_width=True, config=config_chart)

    # --- INIZIO BLOCCO EXPANDER CORRETTO ---
    with st.expander("Visualizza tabella dati storici completi"):
        all_cols = sorted([c for c in df_station.columns if not c.startswith('LEGENDA_') and c not in ['LATITUDINE', 'LONGITUDINE', 'COORDINATEGOOGLE']])
        
        # Lista delle colonne di default con i nuovi nomi corretti e nell'ordine desiderato
        defaults = [
            'DATA', 'STAZIONE', 'TOTALE_PIOGGIA_GIORNO', 'PIOGGE_RESIDUA_ZOFFOLI', 'TEMPERATURA_MEDIANA',
            'TEMPERATURA_MEDIANA_MINIMA', 'SBALZO_TERMICO', 'UMIDITA_DEL_GIORNO', 'UMIDITA_MEDIA_7GG', 'VENTO',
            'SBALZO_TERMICO_MIGLIORE', 'PORCINI_CALDO_NOTE', 'DURATA_RANGE_CALDO', 'CONTEGGIO_GG_RACCOLTA_CALDO',
            'PORCINI_FREDDO_NOTE', 'DURATA_RANGE_FREDDO', 'BOOST', 'CONTEGGIO_GG_RACCOLTA_FREDDO'
        ]
        
        sel_cols = st.multiselect(
            "Seleziona colonne:", 
            options=all_cols, 
            default=[c for c in defaults if c in all_cols]
        )
        
        if sel_cols:
            display_df = df_station[sel_cols].sort_values('DATA', ascending=False)
            ordered_cols = [col for col in defaults if col in sel_cols]
            for col in sel_cols:
                if col not in ordered_cols:
                    ordered_cols.append(col)
            
            st.markdown("""<style>div[data-testid="stDataFrame"] { overflow-x: auto; }</style>""", unsafe_allow_html=True)
            st.dataframe(display_df[ordered_cols])
        else:
            st.info("Seleziona almeno una colonna.")




def main():
    st.set_page_config(page_title="Mappa Funghi Protetta", layout="wide")
    st.title("💧 Analisi Meteo Funghi – by Bobo 🍄")
    query_params = st.query_params

    # Carichiamo i dati
    df, last_loaded_ts = load_and_prepare_data(SHEET_URL)
    
    # Controllo di sicurezza fondamentale
    if df is None or df.empty: 
        st.error("Caricamento dati fallito o il file è vuoto. Controlla il Google Sheet.")
        st.stop()
    
    # --- IL BLOCCO DI DEBUG È STATO RIMOSSO ---

    # Logica per decidere cosa mostrare
    if "station" in query_params:
        display_station_detail(df, query_params["station"])
    else:
        # Codice per il login e la scelta della modalità
        if check_password():
            counter = get_view_counter()
            if st.session_state.get('just_logged_in', False): 
                counter["count"] += 1
                st.session_state['just_logged_in'] = False
            
            mode = st.radio(
                "Seleziona la modalità:", 
                ["Mappa Riepilogativa", "Analisi di Periodo"], 
                horizontal=True
            )

            if mode == "Mappa Riepilogativa": 
                display_main_map(df, last_loaded_ts)
            elif mode == "Analisi di Periodo": 
                display_period_analysis(df)

# Questa riga rimane, è fondamentale per far partire l'app
if __name__ == "__main__":
    main()

