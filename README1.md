Com certeza! Vou compilar todos os passos e códigos em um arquivo Markdown completo, organizado por cada conjunto de dados e apresentação, incluindo o passo final de suavização de dados.

Você pode copiar e colar o conteúdo abaixo diretamente em um arquivo `.md` ou em uma célula Markdown em um notebook Colab/Jupyter.

```markdown
# Análise de Dados com Pandas e Visualização

Este documento detalha o processo de análise de dados realizado utilizando a biblioteca Pandas para manipulação e Matplotlib/Plotly para visualização em Python. O fluxo de trabalho é guiado pelos arquivos de apresentação fornecidos e dividido em três seções principais, cada uma focada em um conjunto de dados específico.

---

## 1. Análise de Aplicativos da Google Play Store (`apps.csv`)

Esta seção segue as atividades propostas no arquivo `Pandas - Framework apps - Copiar.pptx`, focando na limpeza e exploração do dataset de aplicativos.

### 1.1. Configuração Inicial e Carregamento de Dados

Primeiro, conectamos o Google Colab ao Google Drive para acessar os arquivos CSV e carregamos o `apps.csv`.

```python
import pandas as pd
import os
from google.colab import drive

# 1. Conectar o Google Colab ao Google Drive
# Uma janela de autenticação será aberta para autorizar o acesso.
drive.mount('/content/gdrive')

# 2. Navegar até a pasta onde estão os arquivos CSV
# Por padrão, assumimos que os arquivos estão na raiz do seu Google Drive.
# Se estiverem em uma subpasta (ex: 'MeusDados'), o caminho seria '/content/gdrive/MyDrive/MeusDados'
caminho_dos_arquivos = '/content/gdrive/MyDrive/'
os.chdir(caminho_dos_arquivos)

# 3. Carregar o arquivo CSV para um DataFrame
df_apps = pd.read_csv('apps.csv')
```

### 1.2. Exploração Inicial do Dataset

Realizamos uma exploração inicial para entender a estrutura, dimensões e tipos de dados do `df_apps`.

```python
print("### Exploração Inicial de df_apps ###")
print("\nPrimeiras 5 linhas do DataFrame:")
print(df_apps.head())
print("\nÚltimas 5 linhas do DataFrame:")
print(df_apps.tail())
print("\nDimensão do DataFrame (linhas, colunas):")
print(df_apps.shape)
print("\nNomes das colunas:")
print(df_apps.columns)
print("\nInformações detalhadas do DataFrame (tipos de dados, valores não nulos):")
df_apps.info()
```

### 1.3. Limpeza de Dados: Remoção de Duplicatas

Identificamos e removemos quaisquer linhas duplicadas para garantir a integridade da análise.

```python
print("\n### Limpeza de Dados: Duplicatas em df_apps ###")
duplicatas_antes = df_apps.duplicated().sum()
print(f"\nNúmero de duplicatas encontradas antes da remoção: {duplicatas_antes}")

# Remove as duplicatas inplace (modifica o DataFrame original)
df_apps.drop_duplicates(inplace=True)

duplicatas_depois = df_apps.duplicated().sum()
print(f"Número de duplicatas após a remoção: {duplicatas_depois}")
print(f"Nova dimensão de df_apps após remoção de duplicatas: {df_apps.shape}")
```

### 1.4. Limpeza de Dados: Colunas 'Installs' e 'Price'

As colunas `Installs` e `Price` contêm caracteres não numéricos. Removemos esses caracteres e convertemos as colunas para tipos numéricos apropriados.

```python
print("\n### Limpeza de Dados: 'Installs' e 'Price' ###")

# Limpeza e conversão da coluna 'Installs'
print("\nValores únicos de 'Installs' antes da limpeza (exemplo de caracteres):")
print(df_apps['Installs'].unique()[:5]) # Mostra apenas os primeiros 5 para concisão

df_apps['Installs'] = df_apps['Installs'].str.replace(',', '') # Remove vírgulas
df_apps['Installs'] = df_apps['Installs'].str.replace('+', '', regex=False) # Remove o sinal de '+'
df_apps['Installs'] = pd.to_numeric(df_apps['Installs']) # Converte para tipo numérico
print("\nInformações da coluna 'Installs' após limpeza e conversão:")
print(df_apps['Installs'].info())


# Limpeza e conversão da coluna 'Price'
print("\nValores únicos de 'Price' antes da limpeza (exemplo de caracteres):")
print(df_apps['Price'].unique()[:5]) # Mostra apenas os primeiros 5 para concisão

df_apps['Price'] = df_apps['Price'].str.replace('$', '', regex=False) # Remove o símbolo '$'
df_apps['Price'] = pd.to_numeric(df_apps['Price']) # Converte para tipo numérico (float para decimais)
print("\nInformações da coluna 'Price' após limpeza e conversão:")
print(df_apps['Price'].info())

print("\nInformações finais de df_apps após todas as limpezas:")
df_apps.info()
```

### 1.5. Transformação de Dados: Coluna 'Genres'

A coluna `Genres` pode conter múltiplos gêneros separados por ponto e vírgula. Para analisar cada gênero individualmente, dividimos e "empilhamos" os dados.

```python
import plotly.express as px

print("\n### Transformação de Dados: Coluna 'Genres' ###")
print("\nFrequência de gêneros antes da separação (Top 5):")
print(df_apps['Genres'].value_counts().sort_values(ascending=False).head())

# Divide a coluna 'Genres' em múltiplas colunas e, em seguida, empilha-as em uma única Série
stack_genres = df_apps['Genres'].str.split(';', expand=True).stack()
print(f'\nDimensão da Série `stack_genres` após split e stack: {stack_genres.shape}')

# Conta a frequência de cada gênero individual
num_genres = stack_genres.value_counts()
print(f'Número de gêneros únicos identificados: {len(num_genres)}')
print('\nTop 10 Gêneros mais frequentes:')
print(num_genres.head(10))
```

### 1.6. Visualização: Top Gêneros com Plotly

Criamos um gráfico de barras interativo utilizando Plotly Express para visualizar os 15 gêneros mais populares.

```python
print("\n### Visualização: Top Gêneros com Plotly ###")

# Cria o gráfico de barras
bar = px.bar(x = num_genres.index[:15], # Nomes dos gêneros (índice da série)
             y = num_genres.values[:15], # Contagem dos gêneros (valores da série)
             title='Top 15 Gêneros na Google Play Store',
             hover_name=num_genres.index[:15], # Exibe o nome do gênero ao passar o mouse
             color=num_genres.values[:15], # Colore as barras com base na contagem
             color_continuous_scale='Plasma') # Define a escala de cores

# Ajusta o layout do gráfico
bar.update_layout(xaxis_title='Gênero',
                  yaxis_title='Número de Aplicativos',
                  coloraxis_showscale=False) # Remove a legenda de cores
bar.show()
```

---

## 2. Análise de Dados LEGO (`sets.csv` e `themes.csv`)

Esta seção aborda a análise de dados relacionais, utilizando os arquivos `sets.csv` (conjuntos LEGO) e `themes.csv` (temas LEGO), conforme o arquivo `Pandas e dados relacionais (4).pptx`.

### 2.1. Configuração e Carregamento de Dados

Carregamos os dois datasets relacionados a LEGO. (Se estiver em um novo notebook, as etapas de conexão ao Drive e `chdir` devem ser repetidas).

```python
# As linhas abaixo são para garantir que os arquivos sejam carregados,
# caso este trecho seja executado em um notebook separado.
# import pandas as pd
# import os
# from google.colab import drive
# drive.mount('/content/gdrive')
# caminho_dos_arquivos = '/content/gdrive/MyDrive/'
# os.chdir(caminho_dos_arquivos)

df_sets = pd.read_csv('sets.csv')
df_themes = pd.read_csv('themes.csv')

print("### Carregamento de Dados LEGO ###")
print(f"Dimensão de df_sets: {df_sets.shape}")
print(f"Dimensão de df_themes: {df_themes.shape}")
```

### 2.2. Exploração Inicial dos Datasets LEGO

Exploramos a estrutura e as colunas de `df_sets` e `df_themes` para entender suas relações.

```python
print("\n### Exploração Inicial de df_sets e df_themes ###")

print("\nPrimeiras 5 linhas de df_sets:")
print(df_sets.head())
print("\nInformações de df_sets:")
df_sets.info()

print("\nPrimeiras 5 linhas de df_themes:")
print(df_themes.head())
print("\nInformações de df_themes:")
df_themes.info()
```

### 2.3. Contagem de Conjuntos por Tema (IDs) e Necessidade de Junção

Contamos o número de conjuntos para cada `theme_id` em `df_sets` e observamos a necessidade de associar os nomes dos temas (que estão em `df_themes`).

```python
print("\n### Contagem de Conjuntos por Theme ID ###")
sets_per_theme = df_sets['theme_id'].value_counts()
print("Número de conjuntos por Theme ID (Top 5 - IDs numéricos):")
print(sets_per_theme.head())
```

### 2.4. Juntando DataFrames com `.merge()`

Realizamos uma junção (merge) entre `df_sets` e `df_themes` usando `theme_id` de `df_sets` e `id` de `df_themes` como chaves, para trazer os nomes dos temas para o DataFrame de conjuntos.

```python
print("\n### Junção de DataFrames (df_sets e df_themes) ###")
# O 'left' merge mantém todas as linhas de df_sets e adiciona as informações correspondentes de df_themes
merged_df = pd.merge(df_sets, df_themes, left_on='theme_id', right_on='id', suffixes=('_set', '_theme'))

print("\nPrimeiras 5 linhas do DataFrame mesclado (com nomes de temas):")
print(merged_df.head())
print(f"\nDimensão do DataFrame mesclado: {merged_df.shape}")
print("\nInformações do DataFrame mesclado:")
merged_df.info()
```

### 2.5. Contagem de Conjuntos por Nome do Tema

Agora que temos o `name_theme` disponível no `merged_df`, podemos contar os conjuntos por seus nomes reais.

```python
print("\n### Contagem de Conjuntos por Nome do Tema ###")
sets_per_theme_name = merged_df['name_theme'].value_counts()
print("Número de conjuntos por Nome do Tema (Top 10 - Nomes de temas):")
print(sets_per_theme_name.head(10))
```

### 2.6. Visualização: Top Temas com Matplotlib

Criamos um gráfico de barras com Matplotlib para visualizar os 10 temas LEGO com o maior número de conjuntos.

```python
import matplotlib.pyplot as plt

print("\n### Visualização: Top Temas LEGO com Matplotlib ###")
plt.figure(figsize=(12, 6)) # Define o tamanho da figura para melhor legibilidade
plt.bar(sets_per_theme_name.index[:10], sets_per_theme_name.values[:10]) # Cria o gráfico de barras
plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para evitar sobreposição
plt.title('Top 10 Temas LEGO por Número de Conjuntos', fontsize=16)
plt.xlabel('Nome do Tema', fontsize=12)
plt.ylabel('Número de Conjuntos', fontsize=12)
plt.tight_layout() # Ajusta o layout para garantir que todos os elementos sejam visíveis
plt.show()
```

---

## 3. Análise de Séries Temporais e Matplotlib (`QueryResults.csv`)

Esta seção explora o arquivo `QueryResults.csv`, focando na análise de séries temporais e visualização de tendências com Matplotlib, conforme o `Pandas & Matplotlib (2).pptx`.

### 3.1. Configuração e Carregamento de Dados

Carregamos o dataset de resultados de consulta do Stack Overflow. (Se estiver em um novo notebook, as etapas de conexão ao Drive e `chdir` devem ser repetidas).

```python
# As linhas abaixo são para garantir que os arquivos sejam carregados,
# caso este trecho seja executado em um notebook separado.
# import pandas as pd
# import os
# from google.colab import drive
# drive.mount('/content/gdrive')
# caminho_dos_arquivos = '/content/gdrive/MyDrive/'
# os.chdir(caminho_dos_arquivos)

# Carrega o CSV, especificando que a primeira linha (índice 0) é o cabeçalho
df_query_results = pd.read_csv('QueryResults.csv', header=0)

print("### Carregamento de QueryResults.csv ###")
print(f"Dimensão de df_query_results: {df_query_results.shape}")
```

### 3.2. Exploração Inicial e Renomeação de Colunas

Exploramos o DataFrame e corrigimos os nomes das colunas, que foram lidos incorretamente na primeira tentativa.

```python
print("\n### Exploração e Renomeação de Colunas em df_query_results ###")
print("\nPrimeiras 5 linhas após recarregar com header=0 (nomes de colunas originais):")
print(df_query_results.head())

# Renomeia as colunas para nomes mais significativos
# 'm' -> 'mdate' (monthly date), 'TagName' -> 'tag', e a coluna sem nome para 'posts'
df_query_results.columns = ['mdate', 'tag', 'posts']

print("\nPrimeiras 5 linhas após renomear colunas:")
print(df_query_results.head())
print("\nInformações de df_query_results após renomear:")
df_query_results.info()
```

### 3.3. Conversão da Coluna de Data para Datetime

Convertemos a coluna `mdate` para o tipo `datetime` para permitir manipulações e análises de séries temporais.

```python
print("\n### Conversão de 'mdate' para Datetime ###")
df_query_results['mdate'] = pd.to_datetime(df_query_results['mdate'])

print("\nInformações de df_query_results após a conversão de 'mdate':")
df_query_results.info()
```

### 3.4. Preparando Dados para Gráficos de Linha (Pivot Table)

Definimos a coluna `mdate` como o índice do DataFrame e remodelamos os dados usando `pivot_table` para que cada tag de linguagem seja uma coluna separada, com o número de posts como valores.

```python
print("\n### Preparando Dados para Gráficos de Linha (Pivot Table) ###")
# Define 'mdate' como o índice do DataFrame para análise de séries temporais
df_query_results.set_index('mdate', inplace=True)

# Remodela o DataFrame: 'mdate' como índice, 'tag' como colunas, 'posts' como valores
reshaped_df = df_query_results.pivot_table(index='mdate', columns='tag', values='posts')

print("\nPrimeiras 5 linhas do DataFrame remodelado:")
print(reshaped_df.head())
print("\nInformações do DataFrame remodelado:")
reshaped_df.info()
```

### 3.5. Criando Gráficos de Linha (Matplotlib)

Visualizamos as tendências do número de posts para cada linguagem de programação ao longo do tempo usando gráficos de linha do Matplotlib.

```python
import matplotlib.pyplot as plt

print("\n### Criando Gráficos de Linha ###")
plt.figure(figsize=(16, 10)) # Define o tamanho da figura
plt.xticks(fontsize=14)     # Ajusta o tamanho da fonte dos rótulos do eixo X
plt.yticks(fontsize=14)     # Ajusta o tamanho da fonte dos rótulos do eixo Y

plt.xlabel('Data', fontsize=14)
plt.ylabel('Número de Posts', fontsize=14)
plt.title('Tendência do Número de Posts por Linguagem de Programação', fontsize=16)

# Plota cada coluna (tag) como uma linha separada no gráfico
plt.plot(reshaped_df.index, reshaped_df.values, linewidth=3)
plt.legend(reshaped_df.columns, fontsize=12) # Adiciona uma legenda para identificar as linhas

plt.show()
```

### 3.6. Suavização de Dados com Média Móvel (`rolling().mean()`)

Para suavizar as tendências e facilitar a identificação de padrões em séries temporais "emboladas", aplicamos a técnica de média móvel (`rolling().mean()`).

```python
print("\n### Suavização de Dados com Média Móvel ###")

# Calcula a média móvel para cada coluna (tag) com uma janela de 6 meses
# O método .rolling() cria uma "janela" deslizante, e .mean() calcula a média dentro dessa janela.
roll_df = reshaped_df.rolling(window=6).mean()

print("\nPrimeiras 5 linhas do DataFrame suavizado (média móvel):")
print(roll_df.head())

# Plota o DataFrame suavizado para visualizar as tendências mais limpas
plt.figure(figsize=(16, 10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Número de Posts (Média Móvel de 6 Meses)', fontsize=14)
plt.title('Tendência Suavizada do Número de Posts por Linguagem de Programação', fontsize=16)

# Plota cada coluna (tag) do DataFrame suavizado
plt.plot(roll_df.index, roll_df.values, linewidth=3)
plt.legend(roll_df.columns, fontsize=12)

plt.show()
```

---

## Conclusão

Este notebook detalhou um fluxo de trabalho completo de análise de dados, desde o carregamento e a limpeza até a transformação e a visualização, utilizando as bibliotecas Pandas, Matplotlib e Plotly em Python. Abordamos a manipulação de diferentes tipos de dados (categóricos, relacionais e séries temporais) e aplicamos técnicas essenciais como remoção de duplicatas, conversão de tipos, junção de tabelas e suavização de séries temporais para extrair insights significativos dos dados.
```
