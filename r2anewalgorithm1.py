#############################################################
# Alunos: Antônio Paulo Abreu Marques - 190084499           #
#         Marcos Gabriel de Oliveira de Souza - 170126307   #
#         Ryan Reis Fontenele - 211036132                   #
#############################################################

from r2a.ir2a import IR2A
from player.parser import *
from sklearn.neighbors import KNeighborsRegressor as KnnR
import numpy as np 


class R2ANewAlgorithm1(IR2A):

    def __init__(self, id):
        IR2A.__init__(self, id) #herda o IR2A e seus métodos abstratos
        # definir o (state s, action a, reward r)
        # definir a tabela Q (com as tuplas state, action)
        # deve existir uma atualização para a tabela Q a cada nova ação, com penalidade e taxa de aprendizado
        self.qi, self.bd, self.br, self.b, self.td, self.fi, self.r = [], [], [], [], [], [], [] # lista de qualidades, bandwidth, ultimas qualidades, buffer, penalidades, recompensas
        self.X = 0
        t = 0
        s_train = [] # Estados s(t) # s_train
        self.Q = np.zeros((0,0))
        # DEFINIR AINDA : TABELA Q DO Q-LEARNING, COM ZEROS (S(T), A(T))

        q_train = self.Q # Estados Q(s(t), a(t)) # q_train

        self.alpha, self.gamma, self.episilon = 0.1, 0.9, 0.1
        # 2. Inicialize o KNN Regressor
        knn = KnnR(n_neighbors=1)

        # 3. Treine o modelo com os estados anteriores e valores Q
        #knn.fit(s_train, q_train)


    def handle_xml_request(self, msg):
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_xml_response(self, msg):
        # self.b.append() # calcular o buffer
        self.bd.append(msg.get_quality_id())
        self.parsed_mpd = parse_mpd(msg.get_payload()) # parser mpd info
        self.qi = self.parsed_mpd.get_qi() # lista de qualidades possíveis
        if self.qi and self.bd and self.br and self.b and self.td:
            self.Q = np.array((len(self.b)*len(self.bd)*len(self.br)*len(self.td), len(self.qi)))
            print(self.Q)
        
        self.send_up(msg) # envia até a Camada Superior (Player)

    def handle_segment_size_request(self, msg):
        # recebe o próximo segmento de vídeo, sem informar a qualidade.
        # x = qualidade definida / definição do algoritmo Q learning
        # self.X = 
        # adicionar aqui uma previsão para o s(t)
        msg.add_quality_id(self.qi[self.X]) #define a qualidade conforme o knn q-learning
        self.br.append(self.X) # ultima qualidade definida BR
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_segment_size_response(self, msg):
        self.td.append( self.br[t] * 1 / self.bd[t] ) # time download = BR * TimeSegment(1) / Bandwidth
        # self.b.append(max(0,(self.b[t] - Td)) + 1) # atualiza o valor do buffer
        self.fi.append(min(self.alpha*max(0,(Td - self.b[t])), 1) + self.gamma*(max(self.b[t-2] - self.b[t-1], 0))**2) # atualizar a penalidade
        self.r.append( self.br[t] - self.alpha*abs(self.br[t] - self.br[t-1]) - self.fi[t] ) # atualizar a recompensa
        #E = sum((lambda_ ** t) * self.r[t] for t in range(len(self.r))) / len(self.r)

        # a representação segmento ss é recebido pelo IR2A, deve-se executar as instruções para o correto funcionamento;
        # tamanho do segmento recebido = msg.get_bit_length()
        # diferenca de tempo transcorrido entre o momento da requisicao realizada ( handle_segment_size_request() )
        # e a resposta recebida ( handle_segment_size_response() )
        self.send_up(msg) # envia até a Camada Superior (Player)

    def initialize(self):
        pass

    def finalization(self):
        pass

