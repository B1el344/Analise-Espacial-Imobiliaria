"""Geocodificação de endereços via Google Maps, com cache para evitar
chamadas repetidas (e custo) à API para o mesmo endereço."""
 
import streamlit as st
from geopy.geocoders import GoogleV3
 
 
def _extrair_bairro_google(location) -> str | None:
    """Tenta extrair o nome do bairro/sublocalidade a partir dos componentes
    estruturados retornados pelo Google Maps (não do texto livre do
    endereço formatado). Retorna None se o Google não tiver classificado
    nenhum componente como bairro/sublocalidade."""
    componentes = location.raw.get("address_components", [])
    for tipo in ("sublocality_level_1", "sublocality", "neighborhood"):
        for componente in componentes:
            if tipo in componente.get("types", []):
                return componente["long_name"]
    return None
 
 
@st.cache_data(show_spinner=False)
def buscar_coordenadas(endereco_input: str):
    """Busca latitude/longitude, endereço formatado e bairro (segundo o
    Google) para um endereço em Santa Maria - RS. Retorna None se não
    encontrado.
 
    Resultado é cacheado por endereço: buscas repetidas do mesmo texto não
    geram nova chamada à API do Google Maps.
    """
    chave_google = st.secrets["GOOGLE_API_KEY"]
    geolocator = GoogleV3(api_key=chave_google)
 
    busca_completa = f"{endereco_input}, Santa Maria, RS, Brasil"
    location = geolocator.geocode(busca_completa)
 
    if location is None:
        return None
 
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "endereco": location.address,
        "bairro_google": _extrair_bairro_google(location),
    }
