"""Carregamento (cacheado) dos dados espaciais e do modelo preditivo."""

import geopandas as gpd
import pandas as pd
import joblib
import streamlit as st

import config


@st.cache_data
def carregar_dados_espaciais():
    """Carrega o shapefile de bairros, os coeficientes do GWR e a base de
    imóveis com clusters LISA já calculados.

    Retorna:
        (bairros_mapa, df_imoveis_lisa)
    """
    try:
        bairros_gdf = gpd.read_file(config.SHAPEFILE_BAIRROS)
        bairros_gdf = bairros_gdf.to_crs(epsg=4326)

        df_coeficientes = pd.read_csv(config.CSV_COEFICIENTES_GWR)
        bairros_mapa = bairros_gdf.merge(
            df_coeficientes, on=config.COLUNA_JOIN_BAIRRO, how="left"
        )

        df_imoveis_lisa = pd.read_csv(config.CSV_IMOVEIS_LISA)

        return bairros_mapa, df_imoveis_lisa

    except FileNotFoundError as e:
        st.error(
            f"Arquivo de dados não encontrado: **{e.filename}**. "
            "Verifique se os arquivos de dados estão na mesma pasta do app."
        )
        st.stop()


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo XGBoost treinado (joblib)."""
    try:
        return joblib.load(config.MODELO_XGB)
    except FileNotFoundError:
        st.error(f"Modelo não encontrado: **{config.MODELO_XGB}**.")
        st.stop()
