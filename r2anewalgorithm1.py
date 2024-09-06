#############################################################
# Alunos: Antônio Paulo Abreu Marques - 190084499           #
#         Marcos Gabriel de Oliveira de Souza - 170126307   #
#         Ryan Reis Fontenele - 211036132                   #
#############################################################
# dataset original: "http://45.171.101.167/DASHDataset/BigBuckBunny/1sec/BigBuckBunny_1s_simple_2014_05_09.mpd",
# dataset de TESTES FUNCIONAL: "http://164.41.67.41/DASHDatasetTest/BigBuckBunny/1sec/BigBuckBunny_1s_simple_2014_05_09.mpd",
from r2a.ir2a import IR2A
from player.parser import *
from player.player import *
from statistics import mean
from sklearn.neighbors import KNeighborsRegressor as KnnR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time

class KNNModel:
    def __init__(self, n_neighbors=1):
        """
        Inicializa o modelo KNN.
        
        :param n_neighbors: Número de vizinhos a serem considerados.
        """
        self.n_neighbors = n_neighbors
        self.model = KnnR(n_neighbors=self.n_neighbors, weights='distance')  # Usar distância como ponderação

    def fit(self, X, y):
        """
        Treina o modelo KNN nos dados de treino.
        
        :param X: Conjunto de características (features).
        :param y: Conjunto de rótulos (targets).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Faz previsões nos dados fornecidos.
        
        :param X: Conjunto de características (features) para previsão.
        :return: Previsões do modelo.
        """
        return self.model.predict(X)

    def knn_weights(self, X):
        """
        Calcula os pesos w(k) para os K vizinhos mais próximos.
        
        :param X: Estado atual para o qual encontrar os vizinhos.
        :return: Pesos w(k) dos vizinhos.
        """
        # Garanta que X seja uma matriz 2D
        if np.isscalar(X):
            X = np.array([[X]])  # Transforme o escalar em uma matriz 2D com uma única linha
        elif X.ndim == 1:
            X = X.reshape(1, -1)

        distances, indices = self.model.kneighbors(X)
        
        # Calcular w(k) = (1/d(k)) / sum(1/d(k)) para cada k
        weights = []
        for d in distances:
            inverse_distances = 1 / (d + 1e-10)  # Adiciona um pequeno valor para evitar divisão por zero
            sum_inverse_distances = np.sum(inverse_distances)  # Somatório de 1/d(k)
            w = inverse_distances / sum_inverse_distances  # Ponderação final
            weights.append(w)
        
        return weights

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo usando o erro quadrático médio (MSE).
        
        :param X_test: Conjunto de características de teste.
        :param y_test: Conjunto de rótulos de teste.
        :return: O MSE das previsões do modelo.
        """
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

class Qlearn:
    def __init__(self, s, a):
        self.s = s
        self.a = a
        self.Q = np.zeros((self.s.size,self.a.size))
        self.KNN = KNNModel()
        if self.s.ndim == 1:
            self.k = self.s.reshape(-1, 1)
        self.KNN.fit(self.k, self.a)
        self.alpha, self.beta, self.gamma, self.e = 0.5, 0.3, 0.15, 0.2 # gtaxa de aprendizado, fator de desconto e taxa de exploração

    def choose_action(self, s):
        if np.random.uniform(0, 1) < self.e:  # taxa de exploração
            return np.random.choice(self.a.size)  # escolhe uma ação aleatória
        else:
            return np.argmax(self.Q[s])

    def update(self, s, a, r, next):
        if np.isin(s, self.s):
            max = self.choose_action(next)
            upd = r + self.beta * self.Q[next,max] - self.Q[s, a]
            self.Q[s,a] = self.Q[s,a] + self.alpha * upd
            print("updated", s, a, r, next)
        elif not np.isin(s, self.s):
            # Escolhe a ação para o próximo estado
            max_action = self.choose_action(next)

            # Obtém os pesos dos k-vizinhos mais próximos do estado atual
            w = self.KNN.knn_weights(s)

            # Obtém os índices dos k-vizinhos mais próximos para o estado atual e próximo
            next_neighbors_distances, next_neighbors_indices = self.KNN.model.kneighbors(np.array([[next]]))
            s_neighbors_distances, s_neighbors_indices = self.KNN.model.kneighbors(np.array([[s]]))

            # Atualiza Q usando a fórmula ajustada
            upd = (r + self.beta * sum(w[i] * np.max(self.Q[next_neighbors_indices[i], max_action]) for i in range(len(w))) 
            - sum(w[i] * self.Q[s_neighbors_indices[i], max_action] for i in range(len(w))))

            # Atualiza a tabela Q para o estado s e ação a
            self.Q[s_neighbors_indices, a] = self.Q[s_neighbors_indices, a] + self.alpha * upd

    def train(self, e, env):
        """
        Treina o agente em um determinado ambiente.
        
        :param e: Número de episódios de treinamento.
        :param env: O ambiente de treinamento, que deve ter métodos `reset`, `step` e `render`.
        """
        for episode in range(e):
            state = env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                
                env.render()

class R2ANewAlgorithm1(IR2A):

    def __init__(self, id):
        #SimpleModule.__init__(self, id)
        IR2A.__init__(self, id) #herda o IR2A e seus métodos abstratos
        self.qi = [] #lista de qualidades
        self.ls = [] #lista das ultimas-qualidades
        self.throughputs = [] #lista de throughputs
        self.td = [] # lista dos time-of-downloads
        self.buffersizeseconds = []
        self.parsed_mpd = ''
        self.Q = Qlearn(np.empty(20), np.empty(20))
        self.statesdef = False
        self.act = 0

    def handle_xml_request(self, msg):
        self.request_time = time.perf_counter()
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_xml_response(self, msg):
        
        self.parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = self.parsed_mpd.get_qi()
        
        a = np.array(self.qi)
        # definir o (state s, action a, reward r)
        if self.statesdef:
            self.Q = Qlearn(states, a)
        
        self.send_up(msg) # envia até a Camada Superior (Player)

    def handle_segment_size_request(self, msg):
        self.request_time = time.perf_counter()
        #avg = mean(self.throughputs) / 2
        
        # Inicializa 'act' com um valor padrão
        self.act = 1

        if self.ls and self.Q:
            print('funfou')
            self.act = self.Q.choose_action(self.ls[-1])
        print(self.act)
        msg.add_quality_id(self.qi[self.act])
        self.ls.append(self.act)
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_segment_size_response(self, msg):
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
        self.td.append(t)
        if not self.buffersizeseconds:
            i = 0
        else:
            i = self.buffersizeseconds[-1]
        self.buffersizeseconds.append(float(i+t))
        q = self.qi.index(msg.get_quality_id())

        bufferocupancy = np.array(self.buffersizeseconds)
        throughputsestimate = np.array(self.throughputs)
        lastsegments = np.array(self.ls)
        download_time = np.array(self.td)
        
        scaler = MinMaxScaler()

        bufferocupancy = scaler.fit_transform(bufferocupancy.reshape(-1,1)).flatten()
        throughputsestimate = scaler.fit_transform(throughputsestimate.reshape(-1,1)).flatten()
        lastsegments = scaler.fit_transform(lastsegments.reshape(-1,1)).flatten()
        download_time = scaler.fit_transform(download_time.reshape(-1,1)).flatten()

        states = np.vstack([
            bufferocupancy,
            throughputsestimate,
            lastsegments,
            download_time
        ]).T #transposta para que cada linha seja um estado
        self.statesdef = True

        if hasattr(self, 'Q') and len(bufferocupancy) >= 2 and len(throughputsestimate) >= 2 and len(lastsegments) >= 2 and len(download_time) >= 2:
            # Calcular a penalidade φ(t)
            Td = self.td[-1]  # Último tempo de download
            Bt = bufferocupancy[-1]  # Ocupação atual do buffer
            Bm = max(bufferocupancy)  # Tamanho máximo do buffer
            alpha = 0.5
            beta = 0.3

            # Termo de penalidade
            fi = alpha * max(0, Td - Bt) + beta * (max(0, Bm - Bt))**2

            # Função de Recompensa
            q_t = self.ls[-1]  # Qualidade atual
            q_t_1 = self.qi[self.ls[-2]] if len(self.ls) > 0 else 0  # Qualidade do segmento anterior

            rew = q_t - alpha * abs(q_t - q_t_1) - fi  # Função de recompensa


            # Atualização do Q-learning
            a = self.act  # Ação atual
            next = self.Q.choose_action(a)
            self.Q.update(self.ls[-2], a, rew, next) # atualiza a tabela Q conforme o penultimo segmento escolhido
            
        self.send_up(msg) # envia até a Camada Superior (Player)

    def initialize(self):
        pass

    def finalization(self):
        pass

