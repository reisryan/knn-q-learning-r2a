# Cenários simulados
cenarios = {
    "Alta Qualidade": np.array([[90, 10, 1]]),  # Alta largura de banda, baixa latência, baixa perda de pacotes
    "Rede Congestionada": np.array([[20, 80, 10]]),  # Baixa largura de banda, alta latência, alta perda de pacotes
    "Rede Variável": np.array([[50, 40, 5]])  # Variações moderadas
}

# Normalizar os cenários simulados
cenarios_normalizados = {nome: scaler.transform(valor) for nome, valor in cenarios.items()}

# Testar o modelo em cada cenário
for nome, cenario in cenarios_normalizados.items():
    predicao = best_knn.predict(cenario)
    print(f"{nome}: Taxa de bits recomendada: {predicao[0]:.2f} kbps")
