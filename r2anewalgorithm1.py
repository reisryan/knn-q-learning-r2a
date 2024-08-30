#############################################################
# Alunos: Antônio Paulo Abreu Marques - 190084499           #
#         Marcos Gabriel de Oliveira de Souza - 170126307   #
#         Ryan Reis Fontenele - 211036132                   #
#############################################################

from r2a.ir2a import IR2A
from player.parser import *
import random
import numpy


class R2ANewAlgorithm1(IR2A):

    def __init__(self, id):
        IR2A.__init__(self, id) #herda o IR2A e seus métodos abstratos
        # definir o (state s, action a, reward r)
        # definir a tabela Q (com as tuplas state, action)
        # deve existir uma atualização para a tabela Q a cada nova ação, com penalidade e taxa de aprendizado
        self.qi = []
        self.qi_index = 0
        self.t = 0

    def handle_xml_request(self, msg):
        #self.request_time 
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_xml_response(self, msg):
        # ...
        self.parsed_mpd = parse_mpd(msg.get_payload()) # parser mpd info
        self.qi = self.parsed_mpd.get_qi() # lista de qualidades possíveis

        self.send_up(msg) # envia até a Camada Superior (Player)

    def handle_segment_size_request(self, msg):
        # recebe o próximo segmento de vídeo, sem informar a qualidade.
        # x = qualidade definida
        # define a qualidade msg.add_quality_id(x)
        self.send_down(msg) # envia até a Camada Inferior (ConnectionHandler)

    def handle_segment_size_response(self, msg):
        # a representação segmento ss é recebido pelo IR2A, deve-se executar as instruções para o correto funcionamento;
        # tamanho do segmento recebido = msg.get_bit_length()
        # diferenca de tempo transcorrido entre o momento da requisicao realizada ( handle_segment_size_request() )
        # e a resposta recebida ( handle_segment_size_response() )
        self.send_up(msg) # envia até a Camada Superior (Player)

    def initialize(self):
        pass

    def finalization(self):
        pass

