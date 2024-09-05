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
        distances, indices = self.model.kneighbors(X)
        
        # Calcular w(k) = (1/d(k)) / sum(1/d(k)) para cada k
        weights = []
        for d in distances:
            inverse_distances = 1 / d  # 1/d(k)
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
        KNN = KNNModel()
        #KNN.fit(np.reshape(self.s, (-1, 4)), np.reshape(self.a, (-1,)))
        self.alpha, self.beta, self.gamma, self.e = 0.5, 0.4, 0.15, 0.2 # gtaxa de aprendizado, fator de desconto e taxa de exploração

    def choose_action(self, s):
        np.argmax(self.Q[s])

    def update(self, s, a, r, next):
        if np.isin(s, self.s):
            max = self.choose_action(next)
            upd = r + self.beta * self.Q[next,max] - self.Q[s, a]
            self.Q[s,a] = self.Q[s,a] + self.alpha * upd

        '''elif not np.isin(s, self.s):
            # Escolhe a ação para o próximo estado
            max_action = self.choose_action(next_state)

            # Obtém os pesos dos k-vizinhos mais próximos do estado atual
            #w = KNN.knn_weights(s)

            # Obtém os índices dos k-vizinhos mais próximos para o estado atual e próximo
            next_neighbors_distances, next_neighbors_indices = KNN.kneighbors(next_state)
            s_neighbors_distances, s_neighbors_indices = KNN.kneighbors(s)

            # Atualiza Q usando a fórmula ajustada
            upd = (r + self.beta * sum(w[i] * np.max(self.Q[next_neighbors_indices[i], max_action]) for i in range(len(w))) 
            - sum(w[i] * self.Q[s_neighbors_indices[i], max_action] for i in range(len(w))))

            # Atualiza a tabela Q para o estado s e ação a
            self.Q[s_neighbors_indices, a] = self.Q[s_neighbors_indices, a] + self.alpha * upd '''

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
        
    def handle_xml_request(self, msg):
        self.request_time = time.perf_counter()
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_xml_response(self, msg):
        
        self.parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = self.parsed_mpd.get_qi()
        
        #t = time.perf_counter() - self.request_time
        #self.throughputs.append(msg.get_bit_length() / t)

        
        self.send_up(msg) # envia até a Camada Superior (Player)

    def handle_segment_size_request(self, msg):
        self.request_time = time.perf_counter()
        #avg = mean(self.throughputs) / 2
        
        # Inicializa 'act' com um valor padrão
        act = 0  

        if hasattr(self, 'Q'):
            # Verifica se act não está vazio e usa um valor padrão se necessário
            if act:
                act = self.qi[self.Q.choose_action(self.ls[act])]
            else:
                act = 0  # Valor padrão, caso 'self.ls' esteja vazio
        else:
            act = 0

        msg.add_quality_id(self.qi[act])
        self.ls.append(act)
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

        a = np.array(self.qi)
        # definir o (state s, action a, reward r)
        #print(states, a)
        if states.any():
            self.Q = Qlearn(states, a)

        #print("segm, bitlength e kind", msg.get_segment_size(), msg.get_bit_length(), msg.get_kind())
        self.send_up(msg) # envia até a Camada Superior (Player)

    def initialize(self):
        pass

    def finalization(self):
        pass

