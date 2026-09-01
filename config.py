"""Configurações e constantes do app de análise espacial imobiliária."""

# ----------------------------------------------------
# Caminhos dos arquivos de dados
# ----------------------------------------------------
SHAPEFILE_BAIRROS = "santa_maria.shp"
CSV_COEFICIENTES_GWR = "coeficientes_medios_gwr_por_bairro.csv"
CSV_IMOVEIS_LISA = "base_imoveis_com_lisa.csv"
MODELO_XGB = "melhor_xgb.joblib"

# Nome da coluna usada para unir o shapefile de bairros aos coeficientes do GWR
COLUNA_JOIN_BAIRRO = "nome"

# ----------------------------------------------------
# Configurações de mapa
# ----------------------------------------------------
CENTRO_MAPA = [-29.70, -53.80]
ZOOM_PADRAO = 12

CORES_LISA = {
    "High-High (Alto-Alto)": "#d7191c",
    "Low-Low (Baixo-Baixo)": "#0e0d54",
    "High-Low (Alto-Baixo)": "#fa7a39",
    "Low-High (Baixo-Alto)": "#7c7afa",
    "Não Significativo": "#e0e0e0",
}

OPCOES_VARIAVEIS_GWR = {
    "Intercepto (Valor Base)": "beta_intercepto",
    "Área Útil (m²)": "beta_area_util",
    "Quantidade de Quartos": "beta_quartos",
    "Quantidade de Banheiros": "beta_banheiros",
    "Quantidade de Garagens": "beta_garagens",
    "Valorização por ser Apartamento": "beta_apartamento",
    "Valorização por ser Cobertura": "beta_cobertura",
    "Distância para o Centro": "beta_Dist_Centro",
    "Distância para UFSM": "beta_Dist_UFSM",
}

# ----------------------------------------------------
# Métricas de comparação de modelos (aba de notas metodológicas)
# ----------------------------------------------------
METRICAS_MODELOS = {
    "GWR": {"MAE": 226048.81, "RMSE": 671215.71, "MAPE": 28.87},
    "XGBoost": {"MAE": 146327.38, "RMSE": 289435.58, "MAPE": 19.57},
}
