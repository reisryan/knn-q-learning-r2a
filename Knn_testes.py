import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Gerar dados fictícios para exemplo
np.random.seed(42)
# Características: largura de banda, latência, perda de pacotes (valores simulados)
X = np.random.rand(100, 3) * [100, 50, 5]  # Escala das características
# Saída: Taxa de bits ideal (valores simulados entre 500 e 5000 kbps)
y = np.random.randint(500, 5001, 100)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo KNN para regressão
knn = KNeighborsRegressor(n_neighbors=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliar o desempenho do modelo (Erro quadrático médio)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio: {mse:.2f}")

# Exemplo de previsão com novos dados
novos_dados = np.array([[80, 30, 2]])  # Largura de banda: 80, Latência: 30, Perda de pacotes: 2
predicao = knn.predict(novos_dados)
print(f"Taxa de bits recomendada: {predicao[0]:.2f} kbps")
