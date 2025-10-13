import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import os

# --- CONFIGURAZIONE ---
CARTELLA_DATI = r"C:\Users\afantei\Desktop\ProgettoFunghi\dati_gis"
DEM_PATH = os.path.join(CARTELLA_DATI, "dem_toscana.tif")
ASPECT_PATH = os.path.join(CARTELLA_DATI, "aspect_toscana.tif")

def interroga_raster(lon, lat):
    risultati = {"altitudine": "N/D", "desc_aspect": "Non disponibile"}
    try:
        with rasterio.open(DEM_PATH) as dem, rasterio.open(ASPECT_PATH) as asp:
            val_dem = next(dem.sample([(lon, lat)]))[0]
            val_aspect = next(asp.sample([(lon, lat)]))[0]
            risultati["altitudine"] = f"{val_dem:.2f} m"
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
        risultati["altitudine"] = "Fuori area"
    return risultati

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(layout="wide", page_title="Analisi Terreno 3D (clic attivo)")
st.title("🛰️ Analisi 3D del Terreno")
st.write("Clicca sulla mappa per ottenere i dati di altitudine ed esposizione del terreno.")

# Mappa Folium centrata su Toscana
m = folium.Map(location=[43.77, 11.25], zoom_start=8)
st_map = st_folium(m, height=600, width=1000)

if st_map and st_map.get("last_clicked"):
    lat = st_map["last_clicked"]["lat"]
    lon = st_map["last_clicked"]["lng"]

    st.success(f"Hai cliccato su: {lat:.4f}, {lon:.4f}")
    dati = interroga_raster(lon, lat)

    col1, col2, col3 = st.columns(3)
    col1.metric("Latitudine", f"{lat:.4f}")
    col2.metric("Longitudine", f"{lon:.4f}")
    col3.metric("Altitudine", dati["altitudine"])
    st.metric("Esposizione del versante", dati["desc_aspect"])
else:
    st.info("Clicca su un punto della mappa per interrogare il terreno.")
