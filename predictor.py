"""Montagem de features e execução da previsão de preço via XGBoost."""
 
import numpy as np
import pandas as pd
 
from geo_features import calcular_distancias
 
 
def montar_features(
    area_util: float,
    quartos: int,
    banheiros: int,
    garagens: int,
    tipo_imovel: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Monta o DataFrame de entrada do modelo a partir dos dados do formulário."""
    distancias = calcular_distancias(lat, lon)
 
    return pd.DataFrame(
        {
            "area_util": [area_util],
            "quartos": [quartos],
            "banheiros": [banheiros],
            "garagens": [garagens],
            "apartamento": [1 if tipo_imovel == "Apartamento" else 0],
            "cobertura": [1 if tipo_imovel == "Cobertura" else 0],
            **{nome: [valor] for nome, valor in distancias.items()},
            "latitude_calculada": [lat],
            "longitude_calculada": [lon],
        }
    )
 
 
def prever_preco(modelo, dados_usuario: pd.DataFrame) -> float:
    """Valida o schema esperado pelo modelo e retorna o preço previsto em R$.
 
    Lança ValueError com uma mensagem clara caso alguma coluna esperada pelo
    modelo esteja faltando nos dados montados — em vez de deixar o XGBoost
    falhar com um erro genérico.
    """
    colunas_esperadas = list(modelo.get_booster().feature_names)
    faltando = set(colunas_esperadas) - set(dados_usuario.columns)
    if faltando:
        raise ValueError(
            f"Faltam colunas esperadas pelo modelo: {sorted(faltando)}. "
            "Verifique se geo_features.py está alinhado com o pipeline de treino."
        )
 
    dados_usuario = dados_usuario[colunas_esperadas]  # garante a mesma ordem do treino
    ln_preco_previsto = modelo.predict(dados_usuario)[0]
    return float(np.exp(ln_preco_previsto))
 
 
def prever_faixa_preco(modelo, dados_usuario: pd.DataFrame, mape_modelo: float) -> dict:
    """Retorna o preço previsto junto com uma faixa de erro aproximada.
 
    IMPORTANTE: esta faixa NÃO é um intervalo de confiança estatístico (que
    exigiria, por exemplo, regressão quantílica ou bootstrap dos resíduos).
    É uma faixa simples de +/- MAPE em torno do ponto previsto, usando o
    erro percentual médio do modelo medido no conjunto de teste. Serve para
    comunicar a incerteza da previsão de forma honesta, não para dar uma
    garantia estatística formal.
 
    Args:
        modelo: modelo XGBoost treinado.
        dados_usuario: DataFrame já montado com montar_features().
        mape_modelo: MAPE do modelo no conjunto de teste, em percentual
            (ex.: 19.57 para 19,57%).
 
    Returns:
        dict com "preco_previsto", "preco_minimo" e "preco_maximo".
    """
    preco_previsto = prever_preco(modelo, dados_usuario)
    margem = mape_modelo / 100
 
    return {
        "preco_previsto": preco_previsto,
        "preco_minimo": preco_previsto * (1 - margem),
        "preco_maximo": preco_previsto * (1 + margem),
    }
