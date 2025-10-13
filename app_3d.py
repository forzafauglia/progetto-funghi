import streamlit as st
import pydeck as pdk
import rasterio
import os

# --- 1. CONFIGURAZIONE E MOTORE DI CALCOLO ---

CARTELLA_DATI = "." # !!! CAMBIAMENTO CHIAVE !!! Ora che i file sono nella stessa cartella, il percorso è semplicemente "."
DEM_PATH = os.path.join(CARTELLA_DATI, "dem_toscana.tif")
ASPECT_PATH = os.path.join(CARTELLA_DATI, "aspect_toscana.tif")

@st.cache_data
def interroga_raster(lon, lat):
    risultati = {"altitudine": "N/D", "desc_aspect": "Non disponibile"}
    try:
        with rasterio.open(DEM_PATH) as dem, rasterio.open(ASPECT_PATH) as asp:
            val_dem = next(dem.sample([(lon, lat)]))[0]
            if val_dem < -1000 or val_dem == dem.nodata: # Gestisce nodata del DEM
                raise ValueError("Valore nullo o fuori area")
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
    except Exception as e:
        risultati["altitudine"] = "Fuori area o errore"
        risultati["desc_aspect"] = "N/D"
        st.warning(f"Errore di lettura raster: {e}") # Debugging
    return risultati

# --- 2. INTERFACCIA STREAMLIT CON MAPPA 3D (Pydeck) ---

st.set_page_config(layout="wide", page_title="Analisi Terreno 3D")
st.title("🛰️ Analisi 3D del Terreno")
st.write("Clicca sulla mappa per ottenere i dati di altitudine ed esposizione del terreno.")

# Dati di esempio per una stazione (es. una stazione sull'Appennino)
stazione_esempio = { "lat": 44.146, "lon": 10.665 }

# Creiamo la vista 3D iniziale della mappa, centrata sulla stazione
view_state = pdk.ViewState(
    latitude=stazione_esempio["lat"],
    longitude=stazione_esempio["lon"],
    zoom=12,
    pitch=50,  # <-- L'angolo di 50 gradi crea la VISTA 3D
    bearing=0
)

# Layer per la stazione (un punto rosso)
layer_stazione = pdk.Layer(
    "ScatterplotLayer",
    data=[{"position": [stazione_esempio["lon"], stazione_esempio["lat"]], "nome": "Stazione Esempio"}],
    get_position="position",
    get_radius=200,
    get_fill_color=[255, 0, 0, 180],
    pickable=True # Fondamentale per ricevere i dati del click
)

# Creazione della mappa Pydeck
deck = pdk.Deck(
    map_provider="carto",
    map_style="light",
    initial_view_state=view_state,
    layers=[layer_stazione],
    tooltip={"html": "<b>{nome}</b><br/>Clicca qui vicino per esplorare"},
    # Non usare use_container_width=True per ottenere il ritorno del click
    # Impostiamo una dimensione fissa, o la gestiamo via CSS se necessario
)

# --- 3. GESTIONE DELL'INTERATTIVITÀ ---

# Ora usiamo st.pydeck_chart, che restituirà il dict con i dati del click
# Se vuoi la larghezza massima, dovrai gestirla con st.container e css
# Per ora, rendiamo la mappa leggermente più piccola per garantire il click
mappa_data = st.pydeck_chart(deck, height=600, width=800) # <--- CAMBIAMENTO CHIAVE

st.header("Risultati del Punto Selezionato")

# Controlliamo se 'mappa_data' esiste (cioè c'è un click) E se ha la chiave 'picked_object'
if mappa_data and mappa_data.get("picked_object"):
    
    picked_info = mappa_data.get("picked_object") # Usiamo .get() per sicurezza
    
    # Verifichiamo che picked_info e coordinate esistano prima di accedervi
    if picked_info and "coordinate" in picked_info:
        lon_cliccata = picked_info["coordinate"][0]
        lat_cliccata = picked_info["coordinate"][1]
        
        with st.spinner("Calcolo dati..."):
            dati = interroga_raster(lon_cliccata, lat_cliccata)
        
        st.success("Dati calcolati!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Latitudine", f"{lat_cliccata:.4f}")
        col2.metric("Longitudine", f"{lon_cliccata:.4f}")
        col3.metric("Altitudine Stimata", dati["altitudine"])
        
        st.metric("Esposizione del Versante", dati["desc_aspect"])
    else:
        st.info("Cliccato, ma nessun oggetto specifico selezionato o dati coordinate non trovati.")
else:
    st.info("Nessun punto cliccato. Seleziona una posizione sulla mappa.")
