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


    def handle_xml_request(self, msg):
        #self.request_time 
        self.send_down(msg)

    def handle_xml_response(self, msg):
        # ...

        self.send_up(msg)

    def handle_segment_size_request(self, msg):
        # ...
        pass

    def handle_segment_size_response(self, msg):
        # ...
        
        self.send_up(msg)

    def initialize(self):
        pass

    def finalization(self):
        pass

