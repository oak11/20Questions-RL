from parts.stateaction import StateEntities
from parts.stateaction import Actions
from parts.preprocessing import DataProcessing
import parts.util as util
from parts.preprocessing import BagOfWords
from  parts.lstm_net import LSTM_net
from parts.preprocessing import Embedding


import numpy as np
import sys


class Interaction():

    def __init__(self):

        entity_tracker = StateEntities()
        self.bow_encoding = BagOfWords()
        self.embedding = Embedding()


        observations_size = self.embedding.dim + self.bow_encoding.vocab_size + entity_tracker.num_features

        activity_tracker = Actions(entity_tracker)
        self.action_templates = activity_tracker.get_action_templates()
        action_size = activity_tracker.action_size
        nb_hidden = 128

        self.net = LSTM_net(obs_size=observations_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)

        self.net.restore()


    def interact(self):

        entity_tracker = StateEntities()
        activity_tracker = Actions(entity_tracker)
        self.net.reset_state()

        while True:

            u = input(':: ')

            if u == 'clear' or u == 'reset' or u == 'restart':
                self.net.reset_state()
                entity_tracker = StateEntities()
                activity_tracker = Actions(et)
                print('')

            elif u == 'exit' or u == 'stop' or u == 'quit' or u == 'q':
                break

            else:

                if not u:
                    u = '<SILENCE>'


                u_ent = entity_tracker.extract_entities(u)
                u_ent_features = entity_tracker.context_features()
                u_emb = self.embedding.encode(u)
                u_bow = self.bow_encoding.encode(u)

                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)

                action_mask = activity_tracker.action_mask()

                prediction = self.net.forward(features, action_mask)
                print('>>', self.action_templates[prediction])


if __name__ == '__main__':

    session = Interaction()
    session.interact()
