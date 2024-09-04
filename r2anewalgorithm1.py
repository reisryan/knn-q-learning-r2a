#############################################################
# Alunos: Antônio Paulo Abreu Marques - 190084499           #
#         Marcos Gabriel de Oliveira de Souza - 170126307   #
#         Ryan Reis Fontenele - 211036132                   #
#############################################################

from r2a.ir2a import IR2A
from player.parser import *
from sklearn.neighbors import KNeighborsRegressor as KnnR
import numpy as np 

class KNNModel:
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Inicializa o modelo KNN.
        
        :param n_neighbors: Número de vizinhos a serem considerados.
        :param weights: Tipo de ponderação ('uniform' ou 'distance').
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)

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
        self.s = np.array(s)
        self.a = np.array(a)
        self.Q = np.zeros((self.s).size,(self.a).size)
        KNN = KNNModel()
        self.alpha, self.beta, self.gamma, self.e = 0.5, 0,4, 0,15, 0.2 # gtaxa de aprendizado, fator de desconto e taxa de exploração

    def choose_action(self, s):
        np.argmax(self.Q[s])

    def update(self, s, a, r, next):
        if np.isin(s, self.s):
            #upd = r + self.beta * max(self.Q[next,b] - self.Q[s, a])
            #self.Q[s,a] = self.Q[s,a] + self.alpha * upd
            pass
        elif not np.isin(s, self.s):
            #wk = 
            #upd = r + self.beta * sum(w'k*Q(KNN(next),b)) - sum(wk*Q(KNNt,b))
            #self.Q[KNN,a] = self.Q[KNN,a] + self.alpha*upd
            pass 

    def train(self, e, env):
        """
        Treina o agente em um determinado ambiente.
        
        :param e: Número de episódios de treinamento.
        :param env: O ambiente de treinamento, que deve ter métodos `reset`, `step` e `render`.
        """
        for episode in range(episodes):
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
        IR2A.__init__(self, id) #herda o IR2A e seus métodos abstratos        
        
    def handle_xml_request(self, msg):
        # definir o (state s, action a, reward r)
        self.Q = Qlearn(s, a)
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_xml_response(self, msg):
                
        self.send_up(msg) # envia até a Camada Superior (Player)

    def handle_segment_size_request(self, msg):
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_segment_size_response(self, msg):
        self.send_up(msg) # envia até a Camada Superior (Player)

    def initialize(self):
        pass

    def finalization(self):
        pass

