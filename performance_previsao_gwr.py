# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:01:28 2026

@author: gabri
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR

print("--- INICIANDO VALIDAÇÃO DE MACHINE LEARNING (TRAIN/TEST SPLIT) ---")

# 1. DIVISÃO DOS DADOS (80% Treino / 20% Teste)
# O parâmetro random_state=42 garante que a divisão seja sempre a mesma se você rodar de novo
df_train, df_test = train_test_split(gdf_gwr, test_size=0.2, random_state=42)

print(f"Tamanho do Treino: {len(df_train)} imóveis")
print(f"Tamanho do Teste: {len(df_test)} imóveis")

# Variáveis exatas do seu modelo
variaveis_explicativas_gwr = [
    'apartamento', 'cobertura', 'area_util', 'quartos', 
    'banheiros', 'garagens', 'Dist_Centro', 'Dist_UFSM'
]

# 2. PREPARAÇÃO DO CONJUNTO DE TREINO
y_train = df_train['ln_preco'].values.reshape(-1, 1)
X_train = df_train[variaveis_explicativas_gwr].values
coords_train = list(zip(df_train['longitude_calculada'], df_train['latitude_calculada']))

# 3. TREINAMENTO DO MODELO (Apenas nos 80%)
print("\nCalibrando o Bandwidth para os dados de treino (isso pode demorar um pouco)...")
seletor_bw = Sel_BW(coords_train, y_train, X_train, fixed=False, kernel='gaussian')
bw_otimo = seletor_bw.search(search_method='golden_section', criterion='AICc')

print(f"Bandwidth ótimo encontrado: {bw_otimo}. Ajustando o modelo...")
modelo_gwr_train = GWR(coords_train, y_train, X_train, bw_otimo, fixed=False, kernel='gaussian')
resultados_train = modelo_gwr_train.fit()

# 4. PREVISÃO NO CONJUNTO DE TESTE (Os 20% Ocultos)
print("\nRealizando previsões fora da amostra (Vizinho Mais Próximo)...")
previsoes_ln = []
coords_treino_array = np.array(coords_train)

# Loop passando por cada imóvel do conjunto de teste
for index, row in df_test.iterrows():
    coord_alvo = np.array([row['longitude_calculada'], row['latitude_calculada']])
    
    # Encontrar o imóvel de treino mais próximo
    distancias = np.sqrt(np.sum((coords_treino_array - coord_alvo)**2, axis=1))
    indice_vizinho = np.argmin(distancias)
    
    # Extrair betas locais
    betas_locais = resultados_train.params[indice_vizinho]
    
    # Montar vetor X do imóvel testado (Sempre adicionando o 1.0 do intercepto no início)
    x_alvo = np.array([1.0] + row[variaveis_explicativas_gwr].tolist())
    
    # Produto escalar para prever o logaritmo
    ln_previsto = np.dot(x_alvo, betas_locais)
    previsoes_ln.append(ln_previsto)

# 5. CÁLCULO DAS MÉTRICAS DE ERRO
# Adicionando os resultados no DataFrame de Teste para comparação
df_test['ln_preco_previsto'] = previsoes_ln
df_test['preco_previsto'] = np.exp(df_test['ln_preco_previsto'])
df_test['preco_real'] = np.exp(df_test['ln_preco'])  # Revertendo o log real para Reais

y_real = df_test['preco_real']
y_previsto = df_test['preco_previsto']

mae = mean_absolute_error(y_real, y_previsto)
rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
mape = mean_absolute_percentage_error(y_real, y_previsto) * 100

print("\n" + "="*50)
print("      MÉTRICAS FINAIS DE PERFORMANCE (OUT-OF-SAMPLE)      ")
print("="*50)
print(f"MAE  (Erro Médio Absoluto)    : R$ {mae:,.2f}")
print(f"RMSE (Penalização de Erros)   : R$ {rmse:,.2f}")
print(f"MAPE (Erro Percentual Médio)  : {mape:.2f}%")
print("="*50)

# 6. ANÁLISE GRÁFICA (Preço Real vs Erro Percentual)
df_test['erro_percentual'] = ((df_test['preco_previsto'] - df_test['preco_real']) / df_test['preco_real']) * 100

plt.figure(figsize=(10, 6))
sns.scatterplot(x='preco_real', y='erro_percentual', data=df_test, alpha=0.5, color='blue')

# Linhas de referência
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Previsão Perfeita')
plt.axhline(10, color='orange', linestyle=':', label='+10% Erro')
plt.axhline(-10, color='orange', linestyle=':')

plt.title('Análise de Resíduos: Preço Real vs Erro Percentual (20% da Amostra)')
plt.xlabel('Preço Real do Imóvel (R$)')
plt.ylabel('Erro da Previsão (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analise_erro_gwr.png', dpi=300)
plt.show()

print("\nGráfico de análise de erro salvo como 'analise_erro_gwr.png'")