"""Construção dos mapas usados nas abas do app.

Cada função recebe os dados já carregados e devolve um objeto de mapa
pronto para ser renderizado pelo Streamlit (pydeck.Deck ou folium.Map).
Nenhuma função aqui chama st.* diretamente — a renderização fica no app.py.
"""

import folium
import pandas as pd
import pydeck as pdk
import branca.colormap as cm

import config


def montar_mapa_densidade(df_imoveis_lisa: pd.DataFrame) -> pdk.Deck:
    """Mapa 3D de densidade de imóveis (verticalização) via HexagonLayer."""
    view_state = pdk.ViewState(
        latitude=-29.69,
        longitude=-53.80,
        zoom=11.5,
        pitch=45,
        bearing=0,
    )

    camada_hexagonal = pdk.Layer(
        "HexagonLayer",
        data=df_imoveis_lisa,
        get_position=["longitude_calculada", "latitude_calculada"],
        radius=150,
        elevation_scale=2,
        elevation_range=[0, 1000],
        extruded=True,
        pickable=True,
        coverage=0.9,
        get_fill_color="[255, (1 - (elevationValue / 50)) * 255, 0, 200]",
    )

    return pdk.Deck(
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK,
        layers=[camada_hexagonal],
        initial_view_state=view_state,
        tooltip={"text": "Imóveis nesta área: {elevationValue}"},
    )


def montar_mapa_lisa(df_imoveis_lisa: pd.DataFrame) -> folium.Map:
    """Mapa de clusters de preço (LISA), com uma camada toggleável por
    categoria de cluster."""
    m_lisa = folium.Map(
        location=config.CENTRO_MAPA, zoom_start=config.ZOOM_PADRAO, tiles="OpenStreetMap"
    )

    grupos = {}
    for nome_cluster in config.CORES_LISA.keys():
        mostrar_por_padrao = nome_cluster != "Não Significativo"
        grupos[nome_cluster] = folium.FeatureGroup(
            name=nome_cluster, show=mostrar_por_padrao
        )
        grupos[nome_cluster].add_to(m_lisa)

    for _, row in df_imoveis_lisa.iterrows():
        cluster = row["LISA_Cluster"]
        if pd.notna(cluster) and cluster in grupos:
            cor = config.CORES_LISA[cluster]
            folium.CircleMarker(
                location=[row["latitude_calculada"], row["longitude_calculada"]],
                radius=4,
                color=cor,
                fill=True,
                fill_color=cor,
                fill_opacity=0.8,
                weight=0.5,
                tooltip=(
                    f"<b>Bairro:</b> {row['bairro']}<br>"
                    f"<b>Preço:</b> R$ {row['preco_imovel']:,.2f}<br>"
                    f"<b>Padrão:</b> {cluster}"
                ),
            ).add_to(grupos[cluster])

    folium.LayerControl(collapsed=False).add_to(m_lisa)
    return m_lisa


def montar_mapa_gwr(bairros_mapa, coluna_beta: str, variavel_escolhida: str) -> folium.Map:
    """Mapa coroplético dos coeficientes locais do GWR para a variável
    escolhida pelo usuário."""
    vmin = bairros_mapa[coluna_beta].min()
    vmax = bairros_mapa[coluna_beta].max()

    coolwarm_colors = ["#3b4cc0", "#8c9cdd", "#dddddd", "#e28a75", "#b40426"]
    colormap = cm.LinearColormap(colors=coolwarm_colors, vmin=vmin, vmax=vmax)
    colormap.caption = f"Coeficiente: {variavel_escolhida}"

    m_gwr = folium.Map(
        location=config.CENTRO_MAPA, zoom_start=config.ZOOM_PADRAO, tiles="OpenStreetMap"
    )

    folium.GeoJson(
        bairros_mapa,
        style_function=lambda feature: {
            "fillColor": (
                colormap(feature["properties"][coluna_beta])
                if feature["properties"].get(coluna_beta) is not None
                else "transparent"
            ),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["nome", coluna_beta],
            aliases=["Bairro:", "Coeficiente:"],
            localize=True,
        ),
    ).add_to(m_gwr)

    colormap.add_to(m_gwr)
    return m_gwr
