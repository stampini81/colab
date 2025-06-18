# Projeto de Regressão com Redes Neurais MLP

Este repositório contém um projeto de Machine Learning focado em **Regressão** utilizando uma Rede Neural Perceptron Multicamadas (MLPRegressor) da biblioteca `scikit-learn`. O projeto foi desenvolvido em ambiente Google Colab e demonstra o fluxo completo, desde o carregamento dos dados até a avaliação do modelo e a previsão em novas amostras.

Você pode escolher entre dois datasets para executar o projeto:
1.  **`kc_house_data.csv`**: Um dataset clássico de preços de casas (perfeito para regressão).
2.  **`irisMLP.csv`**: O famoso dataset Iris, adaptado para um problema de regressão (prever `petal.width`).

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e executar o projeto no Google Colab.

### Pré-requisitos

* Uma conta Google (para Google Colab e Google Drive).
* Os arquivos `kc_house_data.csv` e `irisMLP.csv` devem ser carregados na **raiz** do seu Google Drive.

### Passos Detalhados

1.  **Abra o Projeto no Google Colab:**
    * Crie um novo notebook Python 3 no Google Colab.
    * Copie e cole o código de cada passo nas células do notebook.

2.  **PASSO 1: Configuração Inicial e Importação de Bibliotecas**
    * Esta célula configura o ambiente e importa todas as bibliotecas necessárias.
    * Ela também montará seu Google Drive, permitindo que o notebook acesse seus arquivos. Uma janela de autenticação do Google pode aparecer; siga as instruções.

    ```python
    # Bibliotecas essenciais para o projeto de MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from google.colab import drive
    import seaborn as sns

    # Montar o Google Drive para acessar seus arquivos
    drive.mount('/content/drive')

    print("Bibliotecas importadas e Google Drive montado com sucesso!")
    ```

3.  **PASSO 2: Carregamento dos Dados**
    * Nesta etapa, você escolhe qual dataset deseja usar (`kc_house_data.csv` ou `irisMLP.csv`) e o carrega para um DataFrame do Pandas.
    * **Certifique-se de que o nome do arquivo escolhido está corretamente descomentado.**
    * **Verifique a saída desta célula para garantir que o arquivo foi carregado com sucesso.** Se houver um erro `FileNotFoundError`, confirme o nome do arquivo e sua presença na raiz do Google Drive (`/content/drive/MyDrive/`).

    ```python
    # --- ESCOLHA SEU DATASET AQUI ---
    # Defina qual arquivo você quer carregar, comentando ou descomentando a linha apropriada:

    # Para usar o dataset de dados de casas (Recomendado para Regressão)
    nome_arquivo_csv = 'kc_house_data.csv'

    # Para usar o dataset Iris (Adaptado para Regressão, veja o PASSO 3 para detalhes da adaptação)
    # nome_arquivo_csv = 'irisMLP.csv'

    # Caminho completo para o arquivo na raiz do Google Drive
    caminho_arquivo = f'/content/drive/MyDrive/{nome_arquivo_csv}'

    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"Dataset '{nome_arquivo_csv}' carregado com sucesso!")
        print("\n--- Primeiras 5 linhas do DataFrame ---")
        print(df.head())
        print("\n--- Informações do DataFrame ---")
        df.info()
        print("\n--- Estatísticas Descritivas do DataFrame ---")
        print(df.describe())

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{nome_arquivo_csv}' NÃO FOI ENCONTRADO na raiz do seu Google Drive.")
        print("Por favor, verifique o nome do arquivo (incluindo maiúsculas/minúsculas) e se ele está diretamente na pasta 'Meu Drive'.")
        print("\nTentando listar os arquivos na sua pasta 'Meu Drive' para ajudar na depuração:")
        !ls /content/drive/MyDrive/
        df = None # Define df como None para evitar NameError mais tarde
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao carregar o arquivo: {e}")
        df = None
    ```

4.  **PASSO 3: Separação de Features (X) e Target (y)**
    * Aqui, definimos quais colunas do seu DataFrame serão as variáveis de entrada (`X`) e qual será a variável de saída (`y`) que o modelo tentará prever.
    * **Atenção:** As definições de `X` e `y` são condicionadas ao dataset escolhido no Passo 2. O código já contém a lógica para `kc_house_data.csv` e `irisMLP.csv` (com nomes de coluna com ponto).
    * **Verifique a saída:** Se houver um `AVISO` sobre colunas não encontradas, isso indica um problema nos nomes das colunas. Verifique a seção "Colunas disponíveis" na saída do Passo 2 e ajuste `expected_features` e `expected_target` conforme necessário.

    ```python
    if df is not None:
        # --- DEFINIÇÃO DE FEATURES (X) E TARGET (y) ---

        print("\n--- Verificando Colunas Disponíveis no DataFrame ---")
        print("Colunas disponíveis:", df.columns.tolist())

        if nome_arquivo_csv == 'kc_house_data.csv':
            # Para kc_house_data.csv:
            features_para_remover = ['id', 'date', 'price']
            X = df.drop(columns=features_para_remover, errors='ignore')
            y = df['price']

            print(f"\nVariáveis de entrada (X) para {nome_arquivo_csv}: {X.columns.tolist()}")
            print(f"Variável de saída (y) para {nome_arquivo_csv}: {y.name}")

        elif nome_arquivo_csv == 'irisMLP.csv':
            # Para irisMLP.csv (adaptando para regressão):
            expected_features = ['sepal.length', 'sepal.width', 'petal.length'] # Nomes de colunas com '.'
            expected_target = 'petal.width' # Nome da coluna com '.'

            missing_features = [col for col in expected_features if col not in df.columns]
            missing_target = expected_target not in df.columns

            if missing_features or missing_target:
                print("\nAVISO: Algumas colunas esperadas para o Iris dataset NÃO foram encontradas.")
                if missing_features: print(f"Features faltando: {missing_features}")
                if missing_target: print(f"Target faltando: {expected_target}")
                print("Por favor, examine as 'Colunas disponíveis' acima e ajuste a definição de X e y.")
                X = None
                y = None
            else:
                X = df[expected_features]
                y = df[expected_target]
                print(f"\nVariáveis de entrada (X) para {nome_arquivo_csv}: {X.columns.tolist()}")
                print(f"Variável de saída (y) para {nome_arquivo_csv}: {y.name}")
        else:
            print("\nATENÇÃO: Você precisa definir as colunas de X e y para o seu dataset específico.")
            X = None
            y = None

        if X is not None and y is not None:
            print(f"Formato de X (features): {X.shape}")
            print(f"Formato de y (target): {y.shape}")
    else:
        print("\nNão foi possível definir X e y porque o DataFrame 'df' não foi carregado com sucesso no Passo 2.")
        print("Por favor, resolva o erro no Passo 2 primeiro.")
    ```

5.  **PASSO 4: Divisão dos Dados em Conjuntos de Treino e Teste**
    * Esta etapa divide seus dados em conjuntos de treino (para o modelo aprender) e teste (para avaliar o desempenho em dados novos).
    * **Certifique-se de que as células dos Passos 1, 2 e 3 foram executadas com sucesso antes de executar esta.**

    ```python
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"\n--- Divisão dos Dados ---")
        print(f"Tamanho do conjunto de treino (X_train): {X_train.shape}")
        print(f"Tamanho do conjunto de teste (X_test): {X_test.shape}")
        print(f"Tamanho do target de treino (y_train): {y_train.shape}")
        print(f"Tamanho do target de teste (y_test): {y_test.shape}")
    else:
        print("\nImpossível dividir os dados. X ou y não foram definidos corretamente nos passos anteriores.")
    ```

6.  **PASSO 5: Criação do Pipeline de Pré-processamento e Modelo**
    * O Pipeline encadeia a normalização dos dados (usando `StandardScaler` ou `MinMaxScaler`) com o modelo `MLPRegressor`. Isso garante consistência nas transformações.

    ```python
    if X is not None and y is not None:
        # Escolha o scaler que melhor se adapta aos seus dados:
        scaler_selected = StandardScaler() # Geralmente uma boa opção para MLP
        # scaler_selected = MinMaxScaler()

        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50), # Exemplo: duas camadas ocultas, 100 e 50 neurônios
            max_iter=1000,                # Aumente se o modelo não convergir (verbose=True ajuda a ver isso)
            activation='relu',            # Função de ativação
            solver='adam',                # Otimizador
            random_state=42,              # Para reprodutibilidade
            verbose=True                  # Para ver o progresso do treinamento
        )

        model_pipeline = Pipeline([
            ('scaler', scaler_selected),
            ('mlp', mlp)
        ])

        print("\nPipeline de pré-processamento e modelo criado:")
        print(model_pipeline)
    else:
        print("\nImpossível criar o pipeline. X ou y não foram definidos corretamente.")
    ```

7.  **PASSO 6: Treinamento do Modelo**
    * O treinamento da Rede Neural. Pode levar alguns instantes, dependendo do tamanho do dataset e da complexidade da rede.

    ```python
    if 'model_pipeline' in locals() and model_pipeline is not None:
        print("\nIniciando o treinamento do modelo...")
        model_pipeline.fit(X_train, y_train)
        print("Treinamento do modelo concluído.")
    else:
        print("\nImpossível treinar o modelo. O pipeline não foi criado ou X_train/y_train não estão definidos.")
    ```

8.  **PASSO 7: Avaliação do Modelo nos Dados de Teste**
    * Avalia o desempenho do modelo em dados que ele não usou para aprender, usando métricas de regressão como MAE, MSE, RMSE e R². Gráficos de dispersão e resíduos são fornecidos para análise visual.

    ```python
    if 'model_pipeline' in locals() and model_pipeline is not None and X_test is not None:
        y_pred = model_pipeline.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        print("\n--- Métricas de Avaliação no Conjunto de Teste ---")
        print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")
        print(f"R² Score (Coeficiente de Determinação): {r2:.4f}")

        # Visualização das Previsões vs. Valores Reais
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel(f"Valores Reais ({y.name if hasattr(y, 'name') else 'Target'})")
        plt.ylabel(f"Valores Preditos ({y.name if hasattr(y, 'name') else 'Target'})")
        plt.title("Valores Reais vs. Valores Preditos (Conjunto de Teste)")
        plt.grid(True)
        plt.show()

        # Visualização dos Resíduos (Erros)
        residuos = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuos, kde=True)
        plt.xlabel("Resíduos (Erro)")
        plt.ylabel("Frequência")
        plt.title("Distribuição dos Resíduos")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuos, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Valores Preditos")
        plt.ylabel("Resíduos (Real - Predito)")
        plt.title("Resíduos vs. Valores Preditos")
        plt.grid(True)
        plt.show()
    else:
        print("\nImpossível avaliar o modelo. Pipeline não treinado ou dados de teste não definidos.")
    ```

9.  **PASSO 8: Teste Individual de Novas Amostras**
    * Demonstra como usar o modelo treinado para fazer previsões em novas amostras. O pipeline cuida da normalização automaticamente.
    * **Ajuste os valores de exemplo** para as características da nova amostra que você deseja prever.

    ```python
    if 'model_pipeline' in locals() and model_pipeline is not None:
        print("\n--- Testando Previsões para Novas Amostras Individuais ---")

        if nome_arquivo_csv == 'kc_house_data.csv':
            # Exemplo de uma nova amostra para o dataset kc_house_data.csv
            # Forneça valores para TODAS as features usadas em X_train, na ordem correta.
            # Verifique a ordem das colunas em X_train.columns para ser preciso.
            # Este é um exemplo simplificado; ajuste para todas as features relevantes.
            nova_casa_exemplo = pd.DataFrame({
                'bedrooms': [3], 'bathrooms': [2.5], 'sqft_living': [2000],
                'sqft_lot': [7000], 'floors': [2], 'waterfront': [0],
                'view': [0], 'condition': [3], 'grade': [7],
                'sqft_above': [2000], 'sqft_basement': [0], 'yr_built': [1990],
                'yr_renovated': [0], 'zipcode': [98103], 'lat': [47.65],
                'long': [-122.34], 'sqft_living15': [1800], 'sqft_lot15': [7500]
            })
            nova_casa_exemplo = nova_casa_exemplo[X_train.columns] # Garante a ordem correta das colunas

            prediction_nova_casa = model_pipeline.predict(nova_casa_exemplo)
            print(f"Características da nova casa (exemplo): {nova_casa_exemplo.values[0][:5]}... (primeiras 5 features)")
            print(f"Previsão do PREÇO da casa: ${prediction_nova_casa[0]:,.2f}")

            scaled_input = model_pipeline.named_steps['scaler'].transform(nova_casa_exemplo)
            print(f"Valores de entrada normalizados para a casa: {scaled_input[0][:5]}... (primeiras 5)")

        elif nome_arquivo_csv == 'irisMLP.csv':
            # Exemplo de uma nova amostra para o dataset Iris
            # Features usadas: 'sepal.length', 'sepal.width', 'petal.length'
            sepal_length_nova = 5.2
            sepal_width_nova = 3.0
            petal_length_nova = 1.5
            nova_flor_exemplo = np.array([[sepal_length_nova, sepal_width_nova, petal_length_nova]])

            prediction_nova_flor = model_pipeline.predict(nova_flor_exemplo)
            print(f"Características da nova flor: sepal.length={sepal_length_nova}, sepal.width={sepal_width_nova}, petal.length={petal_length_nova}")
            print(f"Previsão do petal.width (largura da pétala): {prediction_nova_flor[0]:.4f}")

            scaled_input = model_pipeline.named_steps['scaler'].transform(nova_flor_exemplo)
            print(f"Valores de entrada normalizados para a flor: {scaled_input[0]}")

        else:
            print("Não há um exemplo de previsão individual configurado para o dataset selecionado.")
            print("Por favor, adicione seu próprio exemplo de nova amostra para prever aqui.")

    else:
        print("\nImpossível fazer previsões individuais. O pipeline não foi criado ou treinado.")
    ```

---

## 🤝 Contribuição

Sinta-se à vontade para fazer um "fork" deste repositório, propor melhorias, adicionar novos datasets ou explorar diferentes configurações de modelo.

## 📄 Licença

Este projeto é de código aberto e está disponível sob a [Licença MIT](https://opensource.org/licenses/MIT).

---
