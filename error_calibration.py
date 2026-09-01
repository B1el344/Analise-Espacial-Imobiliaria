"""Calibração da faixa de erro do modelo, com hierarquia bairro -> cluster
LISA -> MAPE global.

Um endereço novo buscado pelo usuário não tem, "de fábrica", um cluster
LISA associado (o LISA é calculado sobre a matriz de vizinhança entre os
imóveis observados). A abordagem usada aqui é uma aproximação prática, não
conformal prediction formal com garantias de cobertura estatística:

1. Identifica o bairro oficial do imóvel buscado via join espacial
   (point-in-polygon) com o shapefile de bairros.
2. Se aquele bairro tiver imóveis suficientes na base histórica, usa o MAPE
   do modelo medido especificamente nesse bairro (nível mais granular).
3. Caso o bairro tenha poucos imóveis para uma estimativa confiável, cai
   para o MAPE do cluster LISA predominante daquele bairro (nível
   intermediário, mais amostras por agrupar vários bairros do mesmo
   "regime espacial").
4. Se nem isso for possível (endereço fora da área mapeada, por exemplo),
   usa o MAPE global do modelo.

Essa calibração hierárquica existe porque, na prática, boa parte dos
bairros de uma cidade não forma um cluster LISA estatisticamente
significativo (a maioria das áreas urbanas costuma cair em "Não
Significativo" no teste de permutação do LISA — isso é uma propriedade
normal do método, não uma falha nos dados). Se a calibração dependesse só
do cluster, ela perderia granularidade geográfica justamente nos casos mais
comuns. Calibrar primeiro por bairro (quando há dados suficientes) evita
esse problema.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

import config

MIN_OBS_POR_GRUPO = 30


def _erros_percentuais(_modelo, df: pd.DataFrame):
    """Roda o modelo sobre df e retorna o erro percentual absoluto por
    linha (Series alinhada ao índice de df), ou None se a base não tiver as
    colunas de features exigidas pelo modelo."""
    colunas_features = list(_modelo.get_booster().feature_names)
    faltando = set(colunas_features) - set(df.columns)
    if faltando or "preco_imovel" not in df.columns:
        return None

    X = df[colunas_features]
    precos_reais = df["preco_imovel"]
    precos_previstos = np.exp(_modelo.predict(X))
    return (precos_previstos - precos_reais).abs() / precos_reais


@st.cache_data(show_spinner=False)
def calcular_mape_por_cluster(_modelo, df_imoveis_lisa: pd.DataFrame) -> dict:
    """MAPE do modelo agregado por cluster LISA. Retorna
    {nome_do_cluster: mape_percentual}. Clusters com menos de
    MIN_OBS_POR_GRUPO observações usam o MAPE global como fallback."""
    mape_global = config.METRICAS_MODELOS["XGBoost"]["MAPE"]
    erro_percentual = _erros_percentuais(_modelo, df_imoveis_lisa)
    if erro_percentual is None:
        return {cluster: mape_global for cluster in config.CORES_LISA.keys()}

    df_erro = pd.DataFrame(
        {
            "LISA_Cluster": df_imoveis_lisa["LISA_Cluster"].values,
            "erro_percentual": erro_percentual.values,
        }
    )

    mape_por_cluster = {}
    for cluster, grupo in df_erro.groupby("LISA_Cluster"):
        if len(grupo) >= MIN_OBS_POR_GRUPO:
            mape_por_cluster[cluster] = float(grupo["erro_percentual"].mean() * 100)
        else:
            mape_por_cluster[cluster] = mape_global

    for cluster in config.CORES_LISA.keys():
        mape_por_cluster.setdefault(cluster, mape_global)

    return mape_por_cluster


@st.cache_data(show_spinner=False)
def calcular_mape_por_bairro(_modelo, df_com_bairro: pd.DataFrame) -> dict:
    """MAPE do modelo agregado por bairro oficial (coluna 'bairro_shapefile',
    atribuída por atribuir_bairro_por_geometria). Só inclui no dict os
    bairros com pelo menos MIN_OBS_POR_GRUPO imóveis — os demais ficam de
    fora e usam o cluster/MAPE global como fallback em
    resolver_mape_do_imovel."""
    erro_percentual = _erros_percentuais(_modelo, df_com_bairro)
    if erro_percentual is None:
        return {}

    df_erro = pd.DataFrame(
        {
            "bairro_shapefile": df_com_bairro["bairro_shapefile"].values,
            "erro_percentual": erro_percentual.values,
        }
    )

    mape_por_bairro = {}
    for bairro, grupo in df_erro.groupby("bairro_shapefile"):
        if len(grupo) >= MIN_OBS_POR_GRUPO:
            mape_por_bairro[bairro] = float(grupo["erro_percentual"].mean() * 100)

    return mape_por_bairro


def identificar_bairro(lat: float, lon: float, bairros_mapa: gpd.GeoDataFrame):
    """Identifica o bairro de um ponto via join espacial (point-in-polygon).
    Retorna None se o ponto não cair dentro de nenhum bairro mapeado."""
    ponto = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
    resultado = gpd.sjoin(ponto, bairros_mapa, how="left", predicate="within")

    if resultado.empty:
        return None

    valor = resultado.iloc[0].get(config.COLUNA_JOIN_BAIRRO)
    return None if pd.isna(valor) else valor


@st.cache_data(show_spinner=False)
def atribuir_bairro_por_geometria(
    df_imoveis_lisa: pd.DataFrame, _bairros_mapa: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Atribui a cada imóvel da base o bairro OFICIAL (segundo o shapefile),
    via join espacial ponto-em-polígono — em vez de usar o texto livre do
    campo 'bairro' extraído por scraping.

    Isso é necessário porque o texto de 'bairro' vem do site de anúncios e
    frequentemente não bate, caractere a caractere, com o nome oficial do
    bairro no shapefile (acentuação, abreviações, grafias diferentes). Usar
    coordenadas em vez de texto elimina esse tipo de divergência.

    O argumento '_bairros_mapa' leva underscore porque um GeoDataFrame não é
    hasheável pelo Streamlit (por causa da coluna de geometria) — o
    underscore diz ao st.cache_data para não tentar gerar um hash dele.
    """
    pontos = gpd.GeoDataFrame(
        df_imoveis_lisa.copy(),
        geometry=gpd.points_from_xy(
            df_imoveis_lisa["longitude_calculada"], df_imoveis_lisa["latitude_calculada"]
        ),
        crs="EPSG:4326",
    )

    juncao = gpd.sjoin(
        pontos,
        _bairros_mapa[[config.COLUNA_JOIN_BAIRRO, "geometry"]],
        how="left",
        predicate="within",
    )
    juncao = juncao[~juncao.index.duplicated(keep="first")]

    df_resultado = df_imoveis_lisa.copy()
    df_resultado["bairro_shapefile"] = juncao[config.COLUNA_JOIN_BAIRRO].reindex(
        df_resultado.index
    )
    return df_resultado


def cluster_predominante_do_bairro(df_imoveis_com_bairro: pd.DataFrame, bairro):
    """Retorna o cluster LISA mais frequente observado historicamente no
    bairro informado (usando a coluna 'bairro_shapefile'). Retorna None se
    o bairro for desconhecido ou não tiver imóveis."""
    if bairro is None:
        return None

    imoveis_bairro = df_imoveis_com_bairro[df_imoveis_com_bairro["bairro_shapefile"] == bairro]
    if imoveis_bairro.empty:
        return None

    return imoveis_bairro["LISA_Cluster"].mode().iloc[0]


def resolver_mape_do_imovel(lat: float, lon: float, bairros_mapa, df_imoveis_lisa, modelo) -> dict:
    """Resolve a faixa de erro do imóvel buscado com calibração hierárquica:
    1) MAPE do próprio bairro, se houver dados suficientes;
    2) senão, MAPE do cluster LISA predominante do bairro;
    3) senão, MAPE global do modelo.

    Retorna um dict com "mape", "bairro", "cluster" e "nivel_calibracao"
    ("bairro", "cluster" ou "global") — esse último serve para o app
    explicar ao usuário de onde veio a faixa exibida.
    """
    mape_global = config.METRICAS_MODELOS["XGBoost"]["MAPE"]

    df_com_bairro = atribuir_bairro_por_geometria(df_imoveis_lisa, bairros_mapa)
    mape_por_bairro = calcular_mape_por_bairro(modelo, df_com_bairro)
    mape_por_cluster = calcular_mape_por_cluster(modelo, df_imoveis_lisa)

    bairro = identificar_bairro(lat, lon, bairros_mapa)
    cluster = cluster_predominante_do_bairro(df_com_bairro, bairro)

    if bairro in mape_por_bairro:
        return {"mape": mape_por_bairro[bairro], "bairro": bairro, "cluster": cluster, "nivel_calibracao": "bairro"}

    if cluster:
        return {"mape": mape_por_cluster.get(cluster, mape_global), "bairro": bairro, "cluster": cluster, "nivel_calibracao": "cluster"}

    return {"mape": mape_global, "bairro": bairro, "cluster": cluster, "nivel_calibracao": "global"}
