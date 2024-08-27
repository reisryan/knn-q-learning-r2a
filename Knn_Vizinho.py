from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ajustar hiperparâmetros usando GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Melhor modelo encontrado
best_knn = grid_search.best_estimator_

# Fazer previsões com o melhor modelo
y_pred = best_knn.predict(X_test_scaled)

# Avaliar o desempenho do melhor modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Melhor erro quadrático médio: {mse:.2f}")
print(f"Melhor número de vizinhos: {grid_search.best_params_['n_neighbors']}")
