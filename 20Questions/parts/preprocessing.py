import parts.util as util
import numpy as np
from gensim.models import word2vec



class Embedding():

    def __init__(self, fname='data/text8.model', dim=300):
        self.dim = dim
        try:
            self.model = word2vec.Word2Vec.load(fname)
        except:
            print(':: creating new word2vec model')
            self.create_model()
            self.model = word2vec.Word2Vec.load(fname)

    def create_model(self, fname='text8'):
        sentences = word2vec.Text8Corpus('data/text8')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        model.save('data/text8.model')
        print(':: model saved to data/text8.model')

    def encode(self, utterance):
        embs = [ self.model[word] for word in utterance.split(' ') if word and word in self.model]
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim],np.float32)


class BagOfWords():

    def __init__(self):
        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab)

    def get_vocab(self):
        content = util.read_content()
        vocab = sorted(set(content.split(' ')))
        # remove empty strings
        return [ item for item in vocab if item ]

    def encode(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        for word in utterance.split(' '):
            if word in self.vocab:
                idx = self.vocab.index(word)
                bow[idx] += 1
        return bow

class DataProcessing():

    def __init__(self, entity_tracker, action_tracker):

        self.action_templates = action_tracker.get_action_templates()
        self.et = entity_tracker

        self.trainset = self.prepare_data()

    def prepare_data(self):

        dialogs, dialog_indices = util.read_dialogs(with_indices=True)
        utterances = util.get_utterances(dialogs)
        responses = util.get_responses(dialogs)
        responses = [ self.get_template_id(response) for response in responses ]

        trainset = []
        for u,r in zip(utterances, responses):
            trainset.append((u,r))

        return trainset, dialog_indices


    def get_template_id(self, response):

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'person' in word:
                    if 'area' in word:
                        template.append('<person_area>')
                    elif 'located' in word:
                        template.append('<person_location>')
                    else:
                        template.append('<person>')
                else:
                    template.append(word)
            return ' '.join(template)

        return self.action_templates.index(
                extract_(self.et.extract_entities(response, update=False))
                )
