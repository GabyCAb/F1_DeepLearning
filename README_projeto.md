# Classificação com Redes Neurais — Previsão de Top 10 na Fórmula 1

Projeto de Inteligência Artificial cujo objetivo é aplicar redes neurais em um problema real de classificação com dados tabulares. Em vez do MNIST sugerido no enunciado, o projeto usa dados reais de resultados de Fórmula 1 (1950–2026) para prever se um piloto termina uma corrida entre os 10 primeiros colocados.

## Pergunta do projeto

Dado o grid de largada, a equipe, o circuito e a forma recente do piloto/equipe na temporada, **o piloto termina a corrida entre os 10 primeiros colocados?**

Essa é uma classificação binária: `1` = terminou no Top 10, `0` = não terminou no Top 10 (fora dos pontos ou não completou a corrida).

## Fonte dos dados

- **API:** [jolpica-f1](https://github.com/jolpica/jolpica-f1) — sucessor open-source e comunitário da antiga Ergast Developer API, disponível em `api.jolpi.ca`.
- **Arquivos usados:** `results.csv` (26.137 linhas — um piloto por corrida, 1950 a 2026) e `races.csv` (calendário e circuitos: localização, latitude/longitude).
- **Atualização:** os dados foram complementados manualmente via a API para cobrir até o GP da Holanda (23/08/2026), já que o dataset original (Kaggle, `fleongg/formula-1-results-and-standings-1950-present`) estava parado no GP de Mônaco (07/06/2026).
- **Licença:** CC BY-NC-SA 4.0 (uso não comercial, com atribuição) — projeto de fins educacionais.

## Por que "Top 10" e não "pontuação"

O sistema de pontos da F1 mudou várias vezes ao longo da história (por décadas só os 6 primeiros pontuavam; a pontuação até o 10º lugar só existe desde 2010). Por isso o alvo foi definido como **"terminar entre os 10 primeiros colocados"** — um critério válido em qualquer época — em vez de "pontuar", que dependeria da regra vigente em cada temporada.

## Engenharia de features (evitando data leakage)

Um cuidado central do projeto foi usar apenas informações disponíveis **antes** do resultado da corrida, para não vazar dados do futuro para o modelo:

| Feature | Tipo | Descrição |
|---|---|---|
| `grid` | numérica | Posição de largada (resultado do qualifying) |
| `lat`, `long` | numérica | Coordenadas do circuito |
| `season` | numérica | Temporada (captura evolução de regras e carros) |
| `driver_avg_pos_season` | numérica | Média de posição do piloto na temporada, calculada só com corridas **anteriores** à atual |
| `constructor_avg_points_season` | numérica | Média de pontos da equipe na temporada, mesma lógica |
| `constructor_id` | categórica | Equipe |
| `circuit_id` | categórica | Circuito |
| `driver_nationality` | categórica | Nacionalidade do piloto |

Colunas como `points`, `status`, `laps` e `time_millis` foram **excluídas** por só existirem depois da corrida acontecer — usá-las seria data leakage (o modelo "colaria" a resposta em vez de aprender a prever).

## Divisão treino/teste

Em vez de um split aleatório, foi usado um **split temporal**:

- **Treino:** temporadas 1950–2023 (24.915 linhas)
- **Teste:** temporadas 2024–2026 (1.222 linhas, nunca vistas durante o treino)

Essa escolha simula o cenário real de uso do modelo — prever corridas futuras a partir do histórico — e evita qualquer contaminação temporal entre treino e teste.

## Pré-processamento

- **Variáveis numéricas:** padronização com `StandardScaler` (necessário para a rede convergir bem).
- **Variáveis categóricas:** one-hot encoding com `OneHotEncoder(handle_unknown='ignore')` — o parâmetro `handle_unknown` é essencial porque equipes/circuitos novos que só aparecem no conjunto de teste (ex.: Cadillac F1 Team, que entrou em 2026) não quebram o pipeline.

## Arquitetura da rede neural

MLP (*Multilayer Perceptron*) construída em Keras/TensorFlow:

```
Input (325 features após one-hot)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu) → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1, sigmoid)
```

- **Otimizador:** Adam (learning rate 1e-3)
- **Função de perda:** Binary Crossentropy
- **Regularização:** Dropout + `EarlyStopping` monitorando o AUC de validação (paciência de 8 épocas, restaurando os melhores pesos)
- **Validação:** os últimos 15% do conjunto de treino (já ordenado por temporada/rodada), correspondendo às temporadas mais recentes dentro do período de treino

## Métricas utilizadas e por quê

| Métrica | O que mede | Por que foi usada aqui |
|---|---|---|
| **Acurácia** | Proporção geral de previsões corretas | Métrica intuitiva, mas isolada pode enganar — o alvo é razoavelmente balanceado (~45% Top10), então aqui é informativa |
| **F1-score** | Média harmônica entre precisão e recall | Importante porque equilibra falsos positivos e falsos negativos, além da simples acurácia |
| **AUC-ROC** | Capacidade do modelo de separar as duas classes em todos os limiares de decisão | Métrica mais robusta a desbalanceamento e independente do limiar de corte (0.5) escolhido |
| **Matriz de confusão** | Detalha acertos/erros por classe (verdadeiro/falso positivo/negativo) | Permite ver se o modelo erra mais para um lado (ex.: prevê Top10 demais ou de menos) |
| **Curvas de treino (loss/acurácia por época)** | Evolução do aprendizado | Ajuda a diagnosticar overfitting (quando a curva de validação piora enquanto a de treino melhora) |

## Resultados obtidos

Avaliação no conjunto de teste (temporadas 2024–2026, nunca vistas pelo modelo):

| Métrica | Valor |
|---|---|
| Acurácia | 77,8% |
| F1-score | 0,783 |
| AUC-ROC | 0,853 |

Matriz de confusão (1.222 corridas-piloto no teste):

|  | Previsto: Fora do Top10 | Previsto: Top10 |
|---|---|---|
| **Real: Fora do Top10** | 462 | 160 |
| **Real: Top10** | 111 | 489 |

O grid de largada é, isoladamente, um forte preditor (quem larga na frente tende a terminar na frente), mas a rede também aprende padrões relevantes de equipe, circuito e forma recente do piloto/equipe na temporada — o ganho de desempenho em relação a um modelo trivial confirma que essas features agregam informação real.

> **Nota sobre reprodutibilidade:** por não-determinismo de ponto flutuante do TensorFlow em CPU (paralelismo de threads, oneDNN), rodar o notebook do zero novamente pode produzir números ligeiramente diferentes destes (variação tipicamente de ±1 ponto percentual). Os valores acima são os da execução salva no notebook, e são a mesma execução que gerou `dashboard_data.json`, `dashboard.html` e `predicoes_proximas_corridas_2026.csv` — os três artefatos são consistentes entre si porque vêm de uma única execução, exportada ao final do notebook (seção 10), em vez de retreinos separados.

### Previsão para corridas futuras (demonstração de uso)

Como aplicação prática, o modelo foi usado para estimar o Top 10 das 3 próximas corridas da temporada 2026 (GP da Itália, GP da Espanha em Madri, GP do Azerbaijão), usando a forma da temporada até o GP da Holanda (última corrida disputada). Resultado consistente entre as três corridas: Russell, Leclerc e Antonelli disputando o topo, Hamilton, Norris e Piastri completando o grupo de ponta, Verstappen e Lawson garantindo a zona de pontos, e uma disputa mais incerta entre Hulkenberg e Lindblad pela 9ª/10ª posição (probabilidades entre 48% e 59%).

**Limitação importante dessa previsão:** como o treino classificatório dessas corridas ainda não aconteceu, o `grid` real foi substituído pela média de grid do piloto na temporada — ou seja, é uma projeção da tendência atual de desempenho, não uma previsão definitiva do resultado do fim de semana. O ideal é rodar o modelo novamente com o grid real assim que o qualifying de cada corrida acontecer.

## Possíveis melhorias futuras

- Incluir dados de qualifying (tempo de volta, não só posição final do treino classificatório) e de pit stops, que não estavam disponíveis nesta versão do dataset.
- Testar outras arquiteturas (mais camadas, regularização L2, tuning de hiperparâmetros com Keras Tuner ou Optuna).
- Restringir a análise à era moderna (2010 em diante, quando a regra do Top10 pontuando é a mesma de hoje) para um recorte mais homogêneo de regras e formato de grid.
- Comparar com baselines simples (regressão logística, árvore de decisão, XGBoost) para quantificar o ganho real de usar uma rede neural em vez de um modelo mais simples.
- Recalcular as previsões de corridas futuras assim que o grid real de cada qualifying estiver disponível.

## Arquivos do projeto

- `dataframes/results.csv` — resultados de cada piloto em cada corrida (1950–2026)
- `dataframes/races.csv` — calendário de corridas e dados de circuitos
- `dataframes/README.md` — documentação original do dataset (fonte, licença, colunas)
- `F1_Classificacao_Top10.ipynb` — notebook completo do projeto (pipeline, modelo, avaliação e previsões), já executado com todas as saídas e gráficos embutidos
- `dashboard_data.json` — métricas, curvas e previsões exportadas pelo notebook (fonte dos dois artefatos abaixo)
- `dashboard.html` — painel visual interativo com os resultados (métricas, curvas de treino, matriz de confusão, ROC e previsões), abre em qualquer navegador sem precisar de servidor ou instalação
- `predicoes_proximas_corridas_2026.csv` — probabilidades de Top10 previstas para as próximas 3 corridas
- `f1_top10_model.keras` — modelo treinado salvo (pode ser recarregado com `keras.models.load_model(...)` sem precisar retreinar)
- `README_projeto.md` — este documento

## Como visualizar os resultados

A forma mais rápida é abrir `dashboard.html` direto no navegador (duplo clique no arquivo) — painel visual com todos os resultados, sem precisar instalar nada.

Para o detalhamento do código e da análise, abra `F1_Classificacao_Top10.ipynb` em Jupyter, VSCode (com a extensão Jupyter) ou Google Colab — os resultados (métricas, gráficos, previsões) já estão salvos dentro do arquivo, sem necessidade de executar nada. Para rodar o notebook do zero, é necessário Python 3.11+ com `pandas`, `numpy`, `scikit-learn`, `tensorflow` e `matplotlib` instalados (ver `requirements.txt`).

## Como manter os dados atualizados

Este projeto usa dados até uma corrida específica de 2026 (ver seção "Fonte dos dados"), mas a temporada continua avançando. Se você clonou este repositório e quer manter os dados e as previsões atualizados a cada corrida, há três caminhos possíveis:

**1. Atualização manual, corrida a corrida.** Depois de cada GP, é possível buscar o resultado mais recente na API [jolpica-f1](https://api.jolpi.ca) e adicionar a linha correspondente em `results.csv`/`races.csv`. É a forma mais simples, mas exige repetir o processo manualmente a cada fim de semana de corrida.

**2. Tarefa agendada (para quem usa Claude/Cowork).** Quem tiver acesso a um agente com tarefas agendadas pode configurar uma rotina recorrente (ex.: semanal, ou logo após cada corrida) que busca os resultados novos na API, atualiza os CSVs e roda a etapa de inferência para gerar previsões frescas das próximas corridas — tudo de forma automática, sem intervenção manual.

**3. Rodar `build.py` localmente (opção mais independente).** A pasta `dataframes/` traz um `build.py` que reconstrói os datasets diretamente da API jolpica-f1 (`pip install pandas requests` e depois `python build.py`). Rodar esse script na sua própria máquina é o caminho mais robusto para manter os dados sempre em dia, já que não depende de nenhum serviço de terceiros além da própria API — e pode ser agendado para rodar sozinho (por exemplo, via Agendador de Tarefas do Windows) logo após o horário de cada corrida.

**Importante: o modelo treinado não precisa ser retreinado a cada atualização.** O `f1_top10_model.keras` já capturou os padrões gerais de grid, equipe, circuito e forma dos pilotos — o que precisa ser recalculado a cada corrida nova são apenas os *features* de forma da temporada (`driver_avg_pos_season`, `constructor_avg_points_season`), que são médias até a última corrida disputada. Com os dados atualizados, basta recarregar o modelo salvo (`keras.models.load_model('f1_top10_model.keras')`) e rodar a etapa de inferência novamente para obter previsões atualizadas das próximas corridas, sem precisar treinar do zero.

## Sobre publicar este projeto

Antes de subir para um repositório público no GitHub, vale ter em mente:

- **Licença dos dados:** o dataset (via jolpica-f1/Ergast) é CC BY-NC-SA 4.0 — uso não comercial, com atribuição e compartilhamento pela mesma licença. Adequado para um projeto de portfólio/acadêmico, mas mencione a fonte e a licença no repositório (este README já faz isso) e não o apresente como produto comercial.
- **Marca:** "Fórmula 1", "F1" e os nomes de equipes/pilotos são marcas de terceiros; este projeto não é afiliado, endossado ou associado à Formula One Group. Bom incluir esse aviso no README do repositório.
- **`dashboard.html` é standalone:** funciona sozinho, sem depender do link privado do Artifact gerado durante o desenvolvimento — é esse arquivo (não o link) que deve ir para o repositório.
- **Ambiente:** adicione um `requirements.txt` (ou `environment.yml`) fixando as versões usadas (`tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`) para que o notebook seja reprodutível por outra pessoa.
