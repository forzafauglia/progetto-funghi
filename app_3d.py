import streamlit as st
import pydeck as pdk
import rasterio
import os
from streamlit_deckgl import deckgl # <-- IMPORTAZIONE CORRETTA

# --- 1. CONFIGURAZIONE E MOTORE DI CALCOLO (INVARIATO) ---
CARTELLA_DATI = "." 
DEM_PATH = os.path.join(CARTELLA_DATI, "dem_toscana.tif")
ASPECT_PATH = os.path.join(CARTELLA_DATI, "aspect_toscana.tif")

@st.cache_data
def interroga_raster(lon, lat):
    risultati = {"altitudine": "N/D", "desc_aspect": "Non disponibile"}
    try:
        with rasterio.open(DEM_PATH) as dem, rasterio.open(ASPECT_PATH) as asp:
            val_dem = next(dem.sample([(lon, lat)]))[0]
            if val_dem < -1000:
                raise ValueError("Valore nullo")
            risultati["altitudine"] = f"{val_dem:.2f} m"
            
            val_aspect = next(asp.sample([(lon, lat)]))[0]
            if val_aspect == -1:
                risultati["desc_aspect"] = "Pianeggiante"
            elif 337.5 <= val_aspect <= 360 or 0 <= val_aspect < 22.5:
                risultati["desc_aspect"] = "Nord"
            elif 22.5 <= val_aspect < 67.5:
                risultati["desc_aspect"] = "Nord-Est"
            elif 67.5 <= val_aspect < 112.5:
                risultati["desc_aspect"] = "Est"
            elif 112.5 <= val_aspect < 157.5:
                risultati["desc_aspect"] = "Sud-Est"
            elif 157.5 <= val_aspect < 202.5:
                risultati["desc_aspect"] = "Sud"
            elif 202.5 <= val_aspect < 247.5:
                risultati["desc_aspect"] = "Sud-Ovest"
            elif 247.5 <= val_aspect < 292.5:
                risultati["desc_aspect"] = "Ovest"
            elif 292.5 <= val_aspect < 337.5:
                risultati["desc_aspect"] = "Nord-Ovest"
    except Exception:
        risultati["altitudine"] = "Fuori area o dato nullo"
        risultati["desc_aspect"] = "N/D"
    return risultati

# --- 2. INTERFACCIA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Analisi Terreno 3D")
st.title("🛰️ Analisi 3D del Terreno")
st.write("Clicca sulla mappa per ottenere i dati di altitudine ed esposizione del terreno.")

stazione_esempio = { "lat": 44.146, "lon": 10.665 }
view_state = pdk.ViewState(
    latitude=stazione_esempio["lat"],
    longitude=stazione_esempio["lon"],
    zoom=12,
    pitch=50,
    bearing=0
)
layer_stazione = pdk.Layer(
    "ScatterplotLayer",
    data=[{"position": [stazione_esempio["lon"], stazione_esempio["lat"]], "nome": "Stazione Esempio"}],
    get_position="position",
    get_radius=200,
    get_fill_color=[255, 0, 0, 180],
    pickable=True
)
deck_config = {
    "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "initialViewState": view_state,
    "layers": [layer_stazione],
    "tooltip": {"html": "<b>{nome}</b><br/>Clicca qui vicino per esplorare"}
}

# --- 3. GESTIONE INTERATTIVITÀ ---
# Usiamo il componente corretto 'deckgl'
mappa_data = deckgl(deck_config, key='deck-gl') # <-- CHIAMATA CORRETTA

st.header("Risultati del Punto Selezionato")

# Questa libreria restituisce un semplice dizionario, quindi .get() funziona!
if mappa_data and mappa_data.get("point"):
    
    point_info = mappa_data.get("point")
    if point_info and "coordinate" in point_info:
        lon_cliccata = point_info["coordinate"][0]
        lat_cliccata = point_info["coordinate"][1]
        
        with st.spinner("Calcolo dati..."):
            dati = interroga_raster(lon_cliccata, lat_cliccata)
        
        st.success("Dati calcolati!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Latitudine", f"{lat_cliccata:.4f}")
        col2.metric("Longitudine", f"{lon_cliccata:.4f}")
        col3.metric("Altitudine Stimata", dati["altitudine"])
        
        st.metric("Esposizione del Versante", dati["desc_aspect"])
else:
    st.info("Nessun punto cliccato. Seleziona una posizione sulla mappa.")
