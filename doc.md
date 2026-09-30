# Game Sales Analytics

## Análise de vendas de videogames com regressão linear

Este projeto busca entender quais características dos jogos têm relação com suas vendas globais. A base contém informações como plataforma, gênero, ano de lançamento, notas da crítica e avaliações dos usuários.

O trabalho segue as etapas do CRISP-DM: entender o problema, conhecer os dados, preparar a base, criar os modelos e avaliar os resultados.

**Unidade analisada:** um jogo em uma plataforma.  
**Variável que queremos estimar:** Global_Sales, em milhões de unidades.  
**Objetivo:** usar regressão linear para analisar padrões de vendas.

## 1. Entendimento do problema

A pergunta principal é:

> Quais informações da base ajudam a explicar as vendas globais de um jogo?

O modelo pode ajudar na comparação entre jogos, gêneros e plataformas. Ele não deve ser entendido como uma previsão exata de sucesso antes do lançamento, pois algumas variáveis, como avaliações dos usuários, só aparecem depois que o jogo já foi lançado.

Também faltam informações importantes, como orçamento, marketing e popularidade anterior da franquia. Portanto, esperamos que o modelo explique apenas uma parte das vendas.

## 2. Entendimento dos dados

A base possui 16.719 registros e 16 colunas. Não existem valores ausentes em Global_Sales, mas várias colunas de avaliação possuem dados faltando.

### Valores ausentes

User_Score possui o texto tbd em alguns registros. Essa sigla significa que a nota ainda seria definida, então ela foi tratada como valor ausente.

Os gráficos mostram que os dados ausentes aparecem mais em alguns anos e gêneros. Se todas essas linhas fossem removidas, a base poderia ficar concentrada em determinados tipos de jogo.

Por esse motivo, os registros foram mantidos. Os valores numéricos ausentes foram preenchidos pela mediana dentro do pipeline.

### Distribuição das vendas

A maior parte dos jogos vende pouco ou moderadamente. Poucos jogos, chamados de blockbusters, vendem muito acima da maioria.

Esses jogos não foram removidos porque são casos reais. Porém, eles dificultam a modelagem, pois uma previsão errada para um blockbuster produz um erro muito alto.

Também usamos gráficos em escala logarítmica para visualizar melhor os jogos comuns e os blockbusters no mesmo espaço.

### Duplicatas

Não foram encontradas linhas totalmente duplicadas. Alguns jogos aparecem mais de uma vez na mesma plataforma, mas possuem diferenças em outras colunas. Sem evidência de que sejam cópias, esses registros foram mantidos.

## 3. Preparação dos dados

### Variáveis removidas

As colunas NA_Sales, EU_Sales, JP_Sales e Other_Sales foram retiradas dos preditores. Elas representam vendas regionais e sua soma forma Global_Sales.

Se essas colunas fossem utilizadas, o modelo receberia praticamente a resposta que deveria estimar. Isso é chamado de vazamento de dados.

Name também foi removida porque funciona como um identificador e possui muitos valores diferentes. O modelo poderia memorizar nomes específicos em vez de aprender padrões gerais.

### Tratamentos utilizados

- valores numéricos ausentes foram preenchidos pela mediana;
- categorias foram transformadas em números com one-hot encoding;
- variáveis numéricas foram padronizadas;
- categorias muito raras de Publisher e Developer foram agrupadas;
- Critic_Count e User_Count receberam versões em log para diminuir sua assimetria.

Esses tratamentos foram colocados em um pipeline. Isso garante que as informações usadas no preenchimento e na padronização venham somente dos dados de treino.

### Treino e teste

A base foi dividida em:

- 80% para treino e comparação dos modelos;
- 20% para o teste final.

O conjunto de teste ficou separado durante a escolha do modelo. Assim, ele representa melhor o comportamento em dados ainda não vistos.

## 4. Modelos

### Regressão linear

A regressão linear tenta representar o alvo como uma soma de contribuições das variáveis:

**vendas = intercepto + coeficiente × variável + erro**

Na regressão simples usamos apenas uma variável. Na regressão múltipla usamos várias variáveis ao mesmo tempo.

O método utilizado é o OLS, ou mínimos quadrados ordinários. Ele escolhe os coeficientes que deixam os erros quadráticos tão pequenos quanto possível.

### Regressão polinomial

Também testamos uma regressão polinomial com:

- quadrado da nota da crítica;
- quadrado da nota dos usuários;
- interação entre as duas notas.

Mesmo usando termos ao quadrado, o ajuste dos coeficientes continua sendo feito por OLS. Esses termos permitem que o modelo represente uma pequena curva.

Não foram criadas todas as combinações possíveis, pois isso deixaria o modelo grande e difícil de interpretar.

### Modelos comparados

1. baseline usando a mediana;
2. regressão simples com Critic_Count;
3. regressão múltipla com variáveis básicas;
4. regressão múltipla com Publisher e Developer;
5. regressão múltipla com contagens em log;
6. regressão com o alvo em log;
7. regressão polinomial.

## 5. Como os modelos foram comparados

Foi utilizada validação cruzada com cinco divisões. Em cada repetição, o modelo treina em quatro partes e é avaliado na parte restante.

As métricas utilizadas foram:

- **MAE:** erro absoluto médio. É fácil de interpretar na unidade das vendas.
- **RMSE:** também mede o erro, mas dá mais peso aos erros grandes.
- **R²:** indica quanto da variação das vendas foi explicada pelo modelo.

O RMSE foi escolhido como métrica principal porque os erros muito grandes, principalmente nos blockbusters, são importantes para o problema.

### Resultados da validação

| Modelo | RMSE médio | MAE médio | R² médio |
|---|---:|---:|---:|
| Regressão polinomial | 1,201 | 0,497 | 0,264 |
| **Regressão múltipla com contagens em log** | **1,207** | **0,500** | **0,257** |
| Múltipla com Publisher e Developer | 1,212 | 0,487 | 0,251 |
| Alvo em log | 1,212 | 0,388 | 0,250 |
| Regressão múltipla base | 1,266 | 0,503 | 0,183 |
| Regressão simples | 1,343 | 0,529 | 0,079 |
| Baseline | 1,444 | 0,457 | -0,065 |

A regressão polinomial apresentou o menor RMSE, mas a diferença foi de apenas 0,006. Essa diferença é bem menor que a variação observada entre as divisões da validação.

Por isso, escolhemos a regressão múltipla com contagens em log. Ela apresentou resultado muito parecido e é mais simples de explicar.

## 6. Resultado no teste

Depois da escolha, o modelo foi treinado novamente usando todo o conjunto de treino e aplicado ao teste.

| Previsão | RMSE | MAE | R² |
|---|---:|---:|---:|
| Original | 1,7968 | 0,5297 | 0,2179 |
| Limitada a zero | 1,7900 | 0,4945 | 0,2238 |

O MAE de 0,4945 significa que o erro absoluto médio ficou próximo de 0,49 milhão de unidades.

O R² de 0,2238 indica que o modelo explicou cerca de 22% da variação das vendas no teste. Esse resultado é melhor que o baseline, mas também mostra que boa parte das vendas depende de informações que não estão na base.

### Previsões negativas

A regressão linear pode produzir valores negativos, mesmo que vendas negativas não existam. Isso aconteceu em 539 registros do teste.

Para o resultado final, essas previsões foram limitadas a zero. As métricas antes do ajuste também foram mantidas para deixar essa limitação visível.

### Análise dos erros

Os gráficos de resíduos mostram que os maiores erros acontecem principalmente nos jogos com vendas muito altas. O modelo funciona melhor para jogos próximos do padrão geral da base.

Também foram analisados erros por gênero, plataforma e faixa de vendas. Isso ajuda a identificar grupos em que o modelo tem mais dificuldade.

## 7. OLS com statsmodels

O LinearRegression do scikit-learn já utiliza OLS. Também foi criado um modelo menor com statsmodels para conferir o funcionamento do método com um conjunto reduzido de variáveis.

Esse modelo utiliza apenas as variáveis numéricas e o alvo em log. O resultado foi:

| Métrica | Resultado |
|---|---:|
| R² ajustado no treino | 0,2242 |
| RMSE no teste | 1,9789 |
| MAE no teste | 0,4867 |
| R² no teste | 0,0514 |

O modelo reduzido teve resultado inferior ao modelo principal. Isso era esperado, pois ele utiliza menos informações. Ele foi mantido apenas como uma comparação adicional entre uma versão simples do OLS e o pipeline completo.

O resultado não muda a escolha final: a regressão múltipla com contagens em log continua sendo o modelo principal, pois apresentou RMSE e R² melhores no teste.

## 8. Principais conclusões

- A regressão simples superou pouco o baseline, mostrando que uma única variável não é suficiente.
- A regressão múltipla melhorou o resultado ao combinar avaliações e características dos jogos.
- A transformação em log ajudou nas variáveis de contagem.
- A regressão polinomial teve o melhor RMSE médio, mas sua vantagem foi muito pequena.
- O modelo final explicou cerca de 22% da variação das vendas no teste.
- Os blockbusters continuam sendo os casos mais difíceis.
- Informações como marketing, orçamento e força da franquia provavelmente ajudariam o modelo.

O projeto mostra que não basta escolher automaticamente o menor erro. Também é importante verificar se a melhora é relevante, analisar os erros e considerar se o modelo continua compreensível.
