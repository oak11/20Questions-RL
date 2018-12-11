from parts.stateaction import StateEntities
from parts.stateaction import Actions
from parts.preprocessing import DataProcessing
import parts.util as util
from parts.preprocessing import BagOfWords
from  parts.lstm_net import LSTM_net
from parts.preprocessing import Embedding

import matplotlib.pyplot as plt
import numpy as np
import sys


class Trainer():

    def __init__(self):

        et = StateEntities()
        self.bow_enc = BagOfWords()
        self.emb = Embedding()
        at = Actions(et)

        self.dataset, dialog_indices = DataProcessing(et, at).trainset
        self.dialog_indices_tr = dialog_indices[:220]
        self.dialog_indices_dev = dialog_indices[220:250]

        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)


    def train(self):

        print('\n training started\n')
        acc = []
        x = []
        epochs = 20
        for j in range(epochs):
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.
            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                start, end = dialog_idx['start'], dialog_idx['end']
                loss += self.dialog_train(self.dataset[start:end])


            accuracy = self.evaluate()
            acc.append(accuracy)
            x.append(j*100)

            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))

            if accuracy > 0.99:
                self.net.save()
                break

        plt.plot(x,acc)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Learning curve for 20 questions")
        plt.show()


    def dialog_train(self, dialog):
        et = StateEntities()
        at = Actions(et)
        self.net.reset_state()

        loss = 0.
        for (u,r) in dialog:
            u_emb = self.emb.encode(u)
            u_bow = self.bow_enc.encode(u)
            u_ent = et.extract_entities(u)
            u_ent_features = et.context_features()

            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            action_mask = at.action_mask()

            loss += self.net.train_step(features, r, action_mask)
        return loss/len(dialog)

    def evaluate(self):

        et = StateEntities()
        at = Actions(et)
        self.net.reset_state()

        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)


            et = StateEntities()

            at = Actions(et)

            self.net.reset_state()


            correct_examples = 0
            for (u,r) in dialog:

                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)

                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                action_mask = at.action_mask()
                prediction = self.net.forward(features, action_mask)
                correct_examples += int(prediction == r)
            dialog_accuracy += correct_examples/len(dialog)

        return dialog_accuracy/num_dev_examples



if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
