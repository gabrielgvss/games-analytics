# Game Sales Analytics — CRISP-DM

## Análise dos fatores associados ao desempenho comercial de videogames

**Objetivo:** analisar quais características dos jogos e indicadores de recepção estão associados ao desempenho comercial global, integrando Business Analytics e regressão linear dentro da metodologia CRISP-DM.

**Unidade de análise:** um jogo em determinada plataforma.  
**Alvo:** `Global_Sales`, em milhões de unidades.  
**Uso pretendido:** explicação e benchmarking pós-lançamento, não previsão absoluta de sucesso antes do lançamento.

## 1. Business Understanding

O problema de negócio é entender por que alguns jogos apresentam desempenho comercial superior a outros e transformar essa análise em referências para portfólio, gênero e plataforma.

O sucesso técnico foi definido antes da modelagem:

- RMSE como métrica principal, por penalizar erros grandes;
- MAE e R² como métricas complementares;
- comparação dos candidatos na mesma validação cruzada de cinco dobras;
- teste reservado avaliado somente depois da seleção;
- uso exclusivo de regressões lineares e extensões lineares regularizadas.

## 2. Data Understanding

A base possui 16.719 registros e 16 variáveis. `Global_Sales` não apresenta valores ausentes, mas os atributos de avaliação têm ausência relevante. O marcador `tbd` de `User_Score` não representa uma nota e foi convertido em valor ausente.

Os padrões de ausência variam com ano, gênero e nível de vendas. Por isso, remover todos os registros incompletos poderia introduzir viés. As observações foram preservadas e o tratamento foi realizado dentro dos pipelines.

A distribuição das vendas é fortemente assimétrica: poucos blockbusters concentram valores muito superiores à maioria. Eles foram mantidos porque representam resultados comerciais reais, não erros de coleta. A assimetria motivou testes com transformações logarítmicas e avaliação dos resíduos por faixa de vendas.

Não há duplicatas integrais. Algumas combinações de jogo e plataforma se repetem, mas os demais campos diferem; portanto, nenhuma linha foi removida sem evidência de duplicação real.

## 3. Data Preparation

As vendas `NA_Sales`, `EU_Sales`, `JP_Sales` e `Other_Sales` foram excluídas dos preditores porque formam diretamente `Global_Sales`. Usá-las permitiria reconstruir o alvo e caracterizaria vazamento de dados.

`Name` também foi excluída por funcionar como identificador de alta cardinalidade. `Publisher` e `Developer` foram avaliadas com agrupamento automático de categorias raras, evitando uma coluna específica para ocorrências pouco frequentes.

A divisão foi feita em 80% para treino e 20% para teste, com semente 42. Antes dessa divisão, foram criadas somente transformações determinísticas por linha. Imputação, indicadores de ausência, escala e one-hot encoding permaneceram dentro dos pipelines e foram aprendidos apenas nas partições de treino.

## 4. Modeling

Foram comparadas as seguintes abordagens, todas pertencentes à família linear:

1. baseline pela mediana;
2. regressão linear simples com `Critic_Count`;
3. regressão linear múltipla com atributos básicos;
4. inclusão controlada de `Publisher` e `Developer`;
5. transformação `log1p` das contagens;
6. transformação `log1p` do alvo;
7. Ridge com busca de `alpha`;
8. Ridge com termos quadráticos e interação.

### Resultados da validação cruzada

| Modelo | RMSE médio | Desvio do RMSE | MAE médio | R² médio |
|---|---:|---:|---:|---:|
| Ridge com termos expandidos, alpha 10 | 1,2000 | 0,0650 | 0,4930 | 0,2650 |
| **Ridge, alpha 10** | **1,2063** | **0,0623** | **0,4970** | **0,2573** |
| Linear múltipla com contagens em log | 1,2070 | 0,0610 | 0,5000 | 0,2570 |
| Linear múltipla com Publisher/Developer | 1,2120 | 0,0690 | 0,4870 | 0,2510 |
| Linear com alvo em log | 1,2130 | 0,0610 | 0,3880 | 0,2490 |
| Linear múltipla base | 1,2660 | 0,0690 | 0,5030 | 0,1830 |
| Linear simples | 1,3430 | 0,0660 | 0,5290 | 0,0790 |
| Baseline pela mediana | 1,4440 | 0,0600 | 0,4570 | -0,0650 |

O Ridge com termos expandidos obteve o menor RMSE, mas sua vantagem precisa ficou abaixo do limiar predefinido de 0,5% em relação ao Ridge mais simples. O modelo expandido foi rejeitado por parcimônia: adicionava termos correlacionados sem ganho material. O modelo final foi **Ridge com `alpha=10`**, contagens em log e categorias raras agrupadas.

A transformação logarítmica do alvo produziu o melhor MAE, mas não o melhor RMSE. Como o projeto definiu RMSE como métrica principal antes dos testes, ela não foi escolhida apenas por favorecer erros típicos e reduzir a influência dos blockbusters.

## 5. Evaluation

O modelo selecionado foi ajustado em todo o treino e aplicado uma única vez ao teste reservado.

| Previsão no teste | RMSE | MAE | R² |
|---|---:|---:|---:|
| Bruta | 1,8070 | 0,5270 | 0,2090 |
| **Limitada a zero** | **1,8010** | **0,4936** | **0,2142** |

Antes do corte, 506 previsões — 15,13% do teste — eram negativas. Como vendas negativas são impossíveis, o resultado operacional foi limitado a zero. As métricas brutas foram preservadas para não esconder essa limitação estrutural da regressão linear.

O teste apresentou erro maior do que a média da validação, embora o MAE tenha permanecido próximo. Isso indica sensibilidade do RMSE à composição de blockbusters na amostra reservada. Os diagnósticos do notebook mostram resíduos, maiores erros, viés por faixa de vendas e desempenho por gênero e plataforma.

O R² de 0,2142 confirma capacidade explicativa moderada: os atributos disponíveis acrescentam informação sobre vendas, mas não capturam grande parte da variabilidade comercial. Coeficientes foram analisados entre dobras, e sinais instáveis não foram usados para conclusões fortes. As associações não devem ser interpretadas como causalidade.

### Validação complementar com OLS

`LinearRegression` já utiliza mínimos quadrados ordinários para estimar seus coeficientes. Para acrescentar testes estatísticos, foi ajustado também um OLS reduzido com `statsmodels`, usando somente o treino, alvo em `log1p`, atributos numéricos padronizados e dummies de gênero, plataforma e classificação.

| Diagnóstico OLS | Resultado |
|---|---:|
| R² ajustado no alvo em log | 0,3388 |
| p-valor de Breusch–Pagan | 1,50 × 10⁻¹⁴⁸ |
| p-valor de Jarque–Bera | < 0,001 |
| Número de condição após padronização | 171,8 |
| RMSE no teste, escala original | 1,9426 |
| MAE no teste, escala original | 0,4528 |
| R² no teste, escala original | 0,0858 |

Breusch–Pagan rejeita homocedasticidade e Jarque–Bera rejeita normalidade dos resíduos. Por isso, os erros-padrão convencionais não são adequados; o notebook usa a correção robusta HC3 e inclui gráficos de resíduos, escala-localização, Q–Q e intervalos de confiança.

O OLS reduzido melhora o MAE, mas apresenta RMSE e R² inferiores aos do Ridge. Ele foi mantido como instrumento de interpretação e verificação de pressupostos, enquanto o Ridge continua sendo o modelo preditivo final escolhido pela validação cruzada.

### Comunicação visual

O notebook contém 11 visualizações executadas, organizadas para responder perguntas específicas:

- padrão e concentração dos valores ausentes;
- distribuição, cauda acumulada e blockbusters;
- correlações e diferenças entre gêneros e plataformas;
- relações de cobertura crítica e engajamento com vendas;
- comparação dos modelos com dispersão entre dobras;
- diferença entre desempenho de treino e validação;
- densidade entre valores reais e previstos e desempenho por decis;
- resíduos, maiores erros e mapa de MAE por gênero e faixa de vendas;
- pressupostos e intervalos robustos do OLS.

## 6. Deployment e valor analítico

O modelo pode apoiar benchmarking entre jogos e segmentos semelhantes, identificação de padrões de recepção e localização de grupos em que o erro é sistematicamente maior.

As principais limitações são a ausência de orçamento, marketing, força prévia da franquia, distribuição, concorrência no período de lançamento e outros fatores externos. Esses elementos ajudam a explicar tanto o R² moderado quanto a dificuldade de estimar blockbusters.

O principal resultado do projeto é um fluxo auditável: decisões fundamentadas nos dados, prevenção de vazamento, comparação justa de regressões lineares, otimização regularizada e diagnóstico explícito dos limites do modelo.
