"""Engenharia de features geoespaciais.

IMPORTANTE: este módulo deve ser importado tanto pelo notebook/script que
treinou o modelo XGBoost quanto pelo app do Streamlit. Isso garante que a
mesma lógica de cálculo de distâncias seja usada em treino e em produção,
eliminando o risco de "train/serving skew" (previsões erradas por
divergência silenciosa entre as duas etapas).
"""

from geopy.distance import geodesic

# Coordenadas de referência de pontos relevantes de Santa Maria - RS.
# Se algum dia esses pontos mudarem, altere apenas aqui.
COORDENADAS_REFERENCIA = {
    "Dist_Centro": (-29.6882, -53.8051),
    "Dist_UFSM": (-29.7209, -53.7148),
    "Dist_Royal": (-29.6901, -53.7949),
    "Dist_Pnova": (-29.7072, -53.8295),
    "Dist_Caridade": (-29.6918, -53.8061),
    "Dist_Parque_Itaimbe": (-29.6848, -53.8030),
    "Dist_Praça_Saldanha": (-29.6860, -53.8069),
    "Dist_Club_Dores": (-29.6879, -53.7968),
    "Dist_Aeroporto": (-29.7084, -53.7005),
}


def calcular_distancias(lat: float, lon: float) -> dict:
    """Calcula a distância (em metros) de um ponto até cada referência urbana.

    Args:
        lat: latitude do imóvel.
        lon: longitude do imóvel.

    Returns:
        dict no formato {"Dist_Centro": metros, "Dist_UFSM": metros, ...}
    """
    coords_imovel = (lat, lon)
    return {
        nome: geodesic(coords_imovel, coords_ref).meters
        for nome, coords_ref in COORDENADAS_REFERENCIA.items()
    }
