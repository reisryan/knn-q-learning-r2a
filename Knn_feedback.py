import time

# Função para simular a coleta contínua de dados de rede
def coletar_dados_rede():
    # Exemplo: valores fictícios, substitua por dados reais
    largura_banda = np.random.uniform(10, 100)  # Mbps
    latencia = np.random.uniform(10, 100)  # ms
    perda_pacotes = np.random.uniform(0, 10)  # %
    return np.array([[largura_banda, latencia, perda_pacotes]])

# Função para ajustar a taxa de bits em tempo real
def ajustar_taxa_bits(modelo, scaler):
    while True:
        # Coletar novos dados de rede
        novos_dados = coletar_dados_rede()
        novos_dados_normalizados = scaler.transform(novos_dados)
        
        # Prever a taxa de bits ideal
        taxa_bits_recomendada = modelo.predict(novos_dados_normalizados)
        print(f"Taxa de bits recomendada: {taxa_bits_recomendada[0]:.2f} kbps")
        
        # Aqui você pode ajustar a taxa de bits no streaming de vídeo
        
        time.sleep(5)  # Simula a coleta contínua de dados a cada 5 segundos

# Executar o ajuste em tempo real
ajustar_taxa_bits(best_knn, scaler)
