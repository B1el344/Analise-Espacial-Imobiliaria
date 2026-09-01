import numpy as np
import streamlit as st
from streamlit_folium import st_folium

import config
from data_loader import carregar_dados_espaciais, carregar_modelo
from error_calibration import resolver_mape_do_imovel
from geocoding import buscar_coordenadas
from maps import montar_mapa_densidade, montar_mapa_gwr, montar_mapa_lisa
from predictor import montar_features, prever_faixa_preco

st.set_page_config(page_title="Análise Espacial Imobiliária", layout="wide")

bairros_mapa, df_imoveis_lisa = carregar_dados_espaciais()

# ----------------------------------------------------
# Layout
# ----------------------------------------------------
st.title("📍 Inteligência de Mercado Imobiliário - Santa Maria, RS")
st.markdown(
    "Explore como as características e a localização afetam a precificação "
    "dos imóveis à venda na cidade."
)

aba1, aba2, aba3, aba4, aba5 = st.tabs(
    [
        "📝 Notas Metodológicas",
        "🏙️ Densidade e Verticalização",
        "🔮 Clusters de Preço (LISA)",
        "🗺️ Elasticidades Locais (GWR)",
        "🎯 Previsão Preço do Imóvel (XGBoost)",
    ]
)

# ==========================================
# ABA 1: NOTAS METODOLÓGICAS
# ==========================================
with aba1:
    st.subheader("Arquitetura Analítica e Metodologia")
    st.markdown(
        "Este portfólio consolida técnicas de econometria espacial e "
        "*Machine Learning* para decodificar a dinâmica de preços do "
        "mercado imobiliário em Santa Maria - RS, com base em dados de 2025."
    )

    with st.expander("1. 📖 Guia do Usuário: Como utilizar este painel"):
        st.markdown(
            """
Este painel foi desenvolvido para auxiliar compradores, vendedores e incorporadoras a compreenderem a dinâmica de preços de imóveis residenciais em Santa Maria - RS.

**O que você encontra em cada aba:**

* **🏙️ Densidade e Verticalização (Aba 2):** Mostra a concentração de prédios na cidade, permitindo identificar as áreas com maior adensamento urbano.
* **🔮 Clusters de Preço (Aba 3):** Ideal para identificar tendências, como regiões que estão recebendo imóveis mais novos e de alto padrão.
* **🗺️ Elasticidades Locais (Aba 4):** Fundamental para incorporadoras e consumidores. O mapa revela quais características geram mais valor de acordo com o bairro. Por exemplo: imóveis compactos em Camobi geram grande valorização pela proximidade com a UFSM, enquanto os mesmos imóveis em bairros como Tancredo Neves apresentam dinâmica diferente. Consumidores também podem usar esta aba para descobrir onde pagarão mais (ou menos) por um quarto adicional.
* **🎯 Previsão Preço do Imóvel (Aba 5):** Uma ferramenta para estimar o valor de venda de um imóvel com base em suas características físicas e locacionais. Como se trata de um modelo preditivo com erro médio de 20%, recomendamos analisar sempre a faixa de preço sugerida como um balizador para a negociação.
            """
        )

    with st.expander("📌 2. Coleta e Engenharia de Dados"):
        st.markdown(
            """
A elaboração da base de dados consistiu em um *web scraping* do site Chaves na Mão na data de 31/08/2025,
extraindo assim as principais características físicas dos imóveis, bem como seu endereço.   
Ademais, a base de dados constituiu cerca de 5400 imóveis, que após limpeza consolidou-se em 4433 imóveis.   
Para o georreferenciamento dos imóveis, foi utilizada a API do Google Maps devido à sua maior precisão. Tal API também é utilizada para a busca da localização na aba para previsão do preço para o imóvel.
            """
        )

    with st.expander("📚 3. Fundamentação Teórica (Precificação Hedônica)"):
        st.markdown(
            """
O Método de Precificação Hedônica (MPH) tem como marco teórico fundamental os trabalhos de Court (1939), Lancaster (1966) e Rosen (1974),
no qual os autores contribuíram significativamente para a teoria microeconômica, ao determinar que não é o bem em si que garante a utilidade, mas sim as suas características intrínsecas.   
Esse método é amplamente utilizado para determinação dos preços de imóveis, dessa forma, o MPH se destaca por incorporar variáveis físicas, locacionais, amenidades e desamenidades,
garantindo precificar o quanto dessas variáveis intangíveis afetam o preço do imóvel.   
Por isso, o MPH se caracteriza como uma técnica poderosa para inferência sobre dinâmicas urbanas.
            """
        )

    with st.expander("🗺️ 4. Análise Espacial (LISA e GWR)"):
        st.markdown(
            """
É conhecido na literatura a dependência espacial para o mercado imobiliário como em Can (1992), o que viola hipóteses das regressões por MQO. Dessa forma, são empregados duas técnicas:

1. **LISA MAP**   
   O mapa LISA é um processo de clusterização espacial fundamentado por Anselin (1995), onde se separa os imóveis em *clusters*.
   Dessa forma o LISA consegue identificar concentrações de imóveis com base em seu preço, dividindo-os em:
      * Alto - Alto: Indicando um submercado de alto-padrão ou valorização
      * Alto - Baixo: Indica um possível processo de gentrificação, imóveis novos ou erros de precificação
      * Baixo - Baixo: Pode refletir áreas periféricas ou com imóveis antigos que estão desvalorizando
      * Baixo - Alto: Pode representar unidades em mau estado de conservação, oportunidades de investimento ou unidades de imóveis antigos rodeados por novos imóveis

   Ademais, O LISA Map costuma ser um diagnosticador para violação da não-estacionareidade espacial em imóveis, o que seria recomendado utilizar modelos como o GWR.

2. **Regressão Geograficamente Ponderada (GWR)**   
   O modelo GWR é um modelo poderoso pois ele permite que os coeficientes variem conforme a posição geográfica das observações.
   Desse modo, é um modelo que consegue capturar a influência que um aumento na quantidade de quartos tem em determinado bairro, por exemplo.   
   Por isso, tal modelo permite a construção de mapas de elasticidade como o exposto na Aba 4 do presente trabalho, aonde se agrega os imóveis conforme o bairro e calcula-se a médias de seus coeficientes para o bairro. Cabe ressaltar que devido a variável dependente estar em logaritmo natural este coeficiente para ser traduzido em valor monetário deve passar por uma transformação. No entando, coeficientes maiores denotam que o bairro em questão possui uma maior valorização ou prêmio por determinada característica.   
   Tal modelo permite, por exemplo, conclusões mais assertivas sobre as características de valorização em uma cidade. Por exemplo, nota-se que em Camobi o custo de um quarto a mais é maior se comparado a outras zonas da cidade, desta forma consumidores que precisem de muitos quartos talvez terão que pagar mais por tal característica no bairro, mas também denota que o foco em Camobi recai para apartamentos mais compactos, devido a atender demandas de estudantes.
            """
        )

    with st.expander("🧠 5. Pipeline Preditivo (*Machine Learning*)"):
        m_gwr, m_xgb = config.METRICAS_MODELOS["GWR"], config.METRICAS_MODELOS["XGBoost"]
        st.markdown(
            rf"""
Apesar de o modelo GWR ser também um modelo de previsão, uma vez que é uma regressão, seu foco recai sobre a inferência e menos na predição em si.
Dessa forma, foi comparado o poder de predição do GWR com o do XGBoost utilizando a técnica *Out of Sample*, separando 80% dos dados para o ajuste do modelo e utilizando os 20% restantes para a previsão. Os resultados estão expostos abaixo, por modelo:

1. **GWR**   
    MAE  (Erro Médio Absoluto)    : R\$ {m_gwr['MAE']:,.2f}   
    RMSE (Penalização de Erros)   : R\$ {m_gwr['RMSE']:,.2f}   
    MAPE (Erro Percentual Médio)  : {m_gwr['MAPE']}%

2. **XGBoost**   
     MAE  (Erro Médio Absoluto)    : R\$ {m_xgb['MAE']:,.2f}   
     RMSE (Penalização de Erros)   : R\$ {m_xgb['RMSE']:,.2f}   
     MAPE (Erro Percentual Médio)  : {m_xgb['MAPE']}%

Comparando os dois modelos, atesta-se a capacidade superior de predição do XGBoost frente ao modelo GWR em todas as métricas. Além disso, o indicador RMSE aponta que o XGBoost é menos sensível a *outliers* se comparado ao GWR, que possui esse indicador representando quase o triplo do seu MAE.

Tais resultados são interessantes, uma vez que demonstram que modelos focados em inferência, de fato, têm desempenhos inferiores aos de modelos focados em predição pura, como o XGBoost. Este último também se mostrou mais robusto ao lidar com *outliers* na amostra.

Entretanto, nota-se ainda que para um produto de dados de uma proptech, como o QuintoAndar, o indicador MAPE de 19,57% ainda é alto (as previsões de mercado dessas empresas giram em torno de 7% a 10%). Porém, isso se dá devido a limitações dos dados da amostra, uma vez que variáveis omitidas — como andar, idade do imóvel, padrão de acabamento e projeto arquitetônico — afetam a qualidade da previsão. Grandes empresas tratam esse problema com robusta engenharia de dados, coletando um volume maior de informações para, assim, gerar predições com um MAPE mais próximo de 10%.
            """
        )

    with st.expander("6. ⚙️ Notas Técnicas e Limitações de Modelagem"):
        st.markdown(
            """
*Esta seção é destinada a profissionais de dados e avaliadores técnicos interessados na arquitetura e nas decisões de modelagem do projeto.*

**Limitações e Pontos de Atenção:**

* **Variáveis Omitidas:** Conforme exposto na seção *4 - Pipeline Preditivo*, a ausência de certas variáveis nos dados coletados afeta a qualidade preditiva do modelo base (XGBoost) e acaba transferindo uma influência maior para as características físicas no modelo espacial (GWR).
* **Multicolinearidade e "Ilhas de Imóveis":** Durante a modelagem GWR, a alta verticalização da amostra gerou locais com perfis idênticos, resultando em *Dummy Trap* (multicolinearidade perfeita para alguns imóveis). Para resolver isso sem perder os dados, a redundância foi absorvida pelo intercepto do modelo espacial — uma característica da álgebra linear sob o capô do pacote `mgwr` (PySAL). Isso garantiu a estabilidade e a validade dos coeficientes das demais variáveis do modelo.
* **Dados de Corte Transversal (Cross-Sectional):** A base de dados reflete o cenário imobiliário capturado em 2025. Como o mercado é dinâmico, previsões futuras devem ser ponderadas considerando fatores macroeconômicos e inflacionários que não estão embutidos na amostra estática atual.
* **Faixa de Erro da Previsão (Aba 5):** A faixa de preço exibida na aba de previsão não é um intervalo de confiança estatístico formal. Ela é calibrada de forma hierárquica:   
    (1) usa o MAPE do modelo medido especificamente no **bairro** do imóvel buscado, quando há imóveis suficientes na base para uma estimativa confiável naquele bairro;    
    (2) caso contrário, recorre ao MAPE do **cluster LISA** predominante da região; (3) na ausência de dados suficientes em ambos os níveis, usa o MAPE global do modelo.    
    Essa hierarquia existe porque a maior parte do território urbano tende a não formar um cluster LISA estatisticamente significativo — uma propriedade normal do método (Anselin, 1995), já que o teste de significância exige um padrão espacial forte o suficiente para rejeitar a hipótese de aleatoriedade. Se a calibração dependesse só do cluster, ela perderia granularidade justamente nas regiões mais comuns da cidade.

Para um aprofundamento na formulação matemática, metodologia e no referencial teórico do modelo espacial (GWR), consulte minha monografia acessando este link: https://repositorio.ufsm.br/handle/1/39182
 """
        )

# ==========================================
# ABA 2: MAPA 3D DE VERTICALIZAÇÃO (PYDECK)
# ==========================================
with aba2:
    st.subheader("Densidade de Imóveis (Verticalização)")
    st.markdown(
        "A altura das colunas representa a quantidade de imóveis anunciados "
        "no mesmo raio. Áreas com torres altas indicam forte verticalização."
    )
    deck = montar_mapa_densidade(df_imoveis_lisa)
    st.pydeck_chart(deck, use_container_width=True)

# ==========================================
# ABA 3: MAPA DE CLUSTERS (LISA) COM FOLIUM
# ==========================================
with aba3:
    st.subheader("Zonas de Concentração de Preços (Autocorrelação Espacial)")
    m_lisa = montar_mapa_lisa(df_imoveis_lisa)
    st_folium(m_lisa, width=1400, height=900, returned_objects=[])

# ==========================================
# ABA 4: MAPA DE COEFICIENTES (GWR) COM FOLIUM
# ==========================================
with aba4:
    st.subheader("Variação do Impacto das Características por Bairro")

    variavel_escolhida = st.selectbox(
        "Selecione a variável para visualizar o impacto no preço:",
        options=list(config.OPCOES_VARIAVEIS_GWR.keys()),
    )
    coluna_beta = config.OPCOES_VARIAVEIS_GWR[variavel_escolhida]

    m_gwr = montar_mapa_gwr(bairros_mapa, coluna_beta, variavel_escolhida)
    st_folium(m_gwr, width=1400, height=900, returned_objects=[])

# ==========================================
# ABA 5: PREVISÃO DOS PREÇOS DE IMÓVEIS
# ==========================================
with aba5:
    st.subheader("Previsão dos preços dos imóveis")
    modelo_xgb = carregar_modelo()
 
    if "lat_imovel" not in st.session_state:
        st.session_state["lat_imovel"] = None
    if "lon_imovel" not in st.session_state:
        st.session_state["lon_imovel"] = None
    if "endereco_salvo" not in st.session_state:
        st.session_state["endereco_salvo"] = ""
    if "bairro_google" not in st.session_state:
        st.session_state["bairro_google"] = None
 
    st.markdown("#### 1. Localização do Imóvel")
    col_busca, col_status = st.columns([2, 1])
 
    with col_busca:
        endereco_input = st.text_input(
            "Endereço (Ex: Rua do Acampamento,100,Centro, Santa Maria)",
            placeholder="Rua, Número, Bairro, Cidade",
        )
 
        if st.button("Buscar Coordenadas 🔍"):
            if endereco_input:
                with st.spinner("Buscando no satélite..."):
                    try:
                        resultado = buscar_coordenadas(endereco_input)
                        if resultado:
                            st.session_state["lat_imovel"] = resultado["latitude"]
                            st.session_state["lon_imovel"] = resultado["longitude"]
                            st.session_state["endereco_salvo"] = resultado["endereco"]
                            st.session_state["bairro_google"] = resultado["bairro_google"]
                        else:
                            st.error("Endereço não encontrado.")
                    except Exception as e:
                        st.error(f"Erro no serviço de busca. Detalhes: {e}")
            else:
                st.warning("Digite um endereço primeiro.")
 
    with col_status:
        if st.session_state["lat_imovel"] is not None:
            st.success("✅ Localização Registrada!")
            st.caption(f"{st.session_state['endereco_salvo']}")
 
    st.markdown("---")
    st.markdown("#### 2. Características Físicas")
 
    with st.form("form_previsao"):
        col1, col2 = st.columns(2)
 
        with col1:
            tipo_imovel = st.radio(
                "Tipo do Imóvel", options=["Apartamento", "Cobertura", "Outro (Casa/Sobrado)"]
            )
            area_util = st.number_input("Área Útil (m²)", min_value=15.0, value=60.0, step=5.0)
 
        with col2:
            quartos = st.number_input("Número de Quartos", min_value=1, value=2)
            banheiros = st.number_input("Número de Banheiros", min_value=1, value=1)
            garagens = st.number_input("Vagas de Garagem", min_value=0, value=1)
 
        pode_prever = st.session_state["lat_imovel"] is not None
        botao_prever = st.form_submit_button(
            "Gerar Previsão de Preço 🚀", disabled=not pode_prever
        )
 
        if not pode_prever:
            st.info("👆 Busque o endereço no passo 1 para habilitar o botão de avaliação.")
 
    if botao_prever:
        lat_final = st.session_state["lat_imovel"]
        lon_final = st.session_state["lon_imovel"]
 
        with st.spinner("Analisando padrões espaciais e calculando preço..."):
            dados_usuario = montar_features(
                area_util, quartos, banheiros, garagens, tipo_imovel, lat_final, lon_final
            )
            try:
                calibracao = resolver_mape_do_imovel(
                    lat_final, lon_final, bairros_mapa, df_imoveis_lisa, modelo_xgb
                )
                resultado = prever_faixa_preco(modelo_xgb, dados_usuario, calibracao["mape"])
 
                def _fmt_real(valor: float) -> str:
                    return (
                        f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    )
 
                st.success("Avaliação concluída com sucesso!")
 
                bairro_google = st.session_state.get("bairro_google")
                bairro_oficial = calibracao["bairro"]
                if (
                    bairro_google
                    and bairro_oficial
                    and bairro_google.strip().lower() != str(bairro_oficial).strip().lower()
                ):
                    st.info(
                        f"ℹ️ O endereço buscado é popularmente associado ao bairro "
                        f"**{bairro_google}**, mas está tecnicamente dentro do limite oficial "
                        f"do bairro **{bairro_oficial}** (usado para calibrar a faixa abaixo). "
                        "Isso é comum perto de divisas entre bairros — o nome popular de uma "
                        "rua nem sempre coincide com o limite administrativo oficial."
                    )
 
                st.metric(
                    label="Valor Estimado de Mercado (ponto central)",
                    value=_fmt_real(resultado["preco_previsto"]),
                )
 
                col_min, col_max = st.columns(2)
                col_min.metric("Piso da faixa estimada", _fmt_real(resultado["preco_minimo"]))
                col_max.metric("Teto da faixa estimada", _fmt_real(resultado["preco_maximo"]))
 
                if calibracao["nivel_calibracao"] == "bairro":
                    st.caption(
                        f"⚠️ Faixa calculada como valor previsto ± {calibracao['mape']:.2f}%, "
                        f"calibrada com o erro histórico do modelo especificamente no bairro "
                        f"**{calibracao['bairro']}**. Não é um intervalo de confiança "
                        "estatístico formal."
                    )
                elif calibracao["nivel_calibracao"] == "cluster":
                    st.caption(
                        f"⚠️ Faixa calculada como valor previsto ± {calibracao['mape']:.2f}%. "
                        f"O bairro **{calibracao['bairro']}** tem poucos imóveis na base para "
                        f"uma calibração própria, então foi usado o erro histórico do cluster "
                        f"espacial predominante na região (**{calibracao['cluster']}**). "
                        "Não é um intervalo de confiança estatístico formal."
                    )
                else:
                    st.caption(
                        f"⚠️ Faixa calculada como valor previsto ± {calibracao['mape']:.2f}% "
                        "(MAPE global do modelo — não foi possível identificar o bairro deste "
                        "endereço para uma calibração mais específica). "
                        "Não é um intervalo de confiança estatístico formal."
                    )
 
                dist_debug = {
                    k: v
                    for k, v in zip(
                        ["Dist_UFSM", "Dist_Centro", "Dist_Royal"],
                        [
                            dados_usuario["Dist_UFSM"].iloc[0],
                            dados_usuario["Dist_Centro"].iloc[0],
                            dados_usuario["Dist_Royal"].iloc[0],
                        ],
                    )
                }
                with st.expander("Ver detalhes espaciais extraídos"):
                    st.write(f"**Distância para a UFSM:** {dist_debug['Dist_UFSM'] / 1000:.2f} km")
                    st.write(f"**Distância para o Centro:** {dist_debug['Dist_Centro'] / 1000:.2f} km")
                    st.write(f"**Distância para o Shopping Royal:** {dist_debug['Dist_Royal'] / 1000:.2f} km")
 
            except ValueError as e:
                st.error(f"Erro na previsão: {e}")
