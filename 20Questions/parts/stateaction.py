import parts.util as util
import numpy as np
from enum import Enum



class StateEntities():

    def __init__(self):
        self.entities = {
                '<ethnicity>' : None,
                '<location>' : None,
                '<age>' : None,
                '<area>' : None,
                '<gender>' : None,
                }
        self.num_features = 5
        self.rating = None


        self.age = ['10', '20', '30', '40', '50', '60', '70', '80', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty']
        self.locations = ['bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london', 'madrid', 'seoul', 'tokyo']
        self.ethnicity = ['british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai', 'vietnamese']
        self.area = ['politics', 'sports', 'technology']
        self.gender = ['male', 'female']

        self.EntType = Enum('Entity Type', '<age> <location> <ethnicity> <area> <gender> <non_ent>')


    def ent_type(self, ent):
        if ent in self.age:
            return self.EntType['<age>'].name
        elif ent in self.locations:
            return self.EntType['<location>'].name
        elif ent in self.ethnicity:
            return self.EntType['<ethnicity>'].name
        elif ent in self.area:
            return self.EntType['<area>'].name
        elif ent in self.gender:
            return self.EntType['<gender>'].name
        else:
            return ent


    def extract_entities(self, utterance, update=True):
        tokenized = []
        for word in utterance.split(' '):
            entity = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = word

            tokenized.append(entity)

        return ' '.join(tokenized)


    def context_features(self):
       keys = list(set(self.entities.keys()))
       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys],
                                   dtype=np.float32 )
       return self.ctxt_features


    def action_mask(self):
        print('Not yet implemented. Need a list of action templates!')


class Actions():

    def __init__(self, ent_tracker):

        self.et = ent_tracker
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)
        self.am = np.zeros([self.action_size], dtype=np.float32)

        self.am_dict = {
                '0000' : [ 4,8,1,14,7,15],
                '0001' : [ 4,8,1,14,7],
                '0010' : [ 4,8,1,14,15],
                '0011' : [ 4,8,1,14],
                '0100' : [ 4,8,1,7,15],
                '0101' : [ 4,8,1,7],
                '0110' : [ 4,8,1,15],
                '0111' : [ 4,8,1],
                '1000' : [ 4,8,14,7,15],
                '1001' : [ 4,8,14,7],
                '1010' : [ 4,8,14,15],
                '1011' : [ 4,8,14],
                '1100' : [ 4,8,7,15],
                '1101' : [ 4,8,7],
                '1110' : [ 4,8,15],
                '1111' : [ 2,3,5,6,8,9,10,11,12,13,16 ]
                }


    def action_mask(self):
        ctxt_f = ''.join([ str(flag) for flag in self.et.context_features().astype(np.int32) ])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index-1] = 1.
            return self.am

        return construct_mask(ctxt_f)

    def get_action_templates(self):
        responses = list(set([ self.et.extract_entities(response, update=False)
            for response in util.get_responses() ]))

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'person_' in word:
                    if 'old' in word:
                        template.append('<person_age>')
                    elif 'locate' in word:
                        template.append('<person_location>')
                    else:
                        template.append('<person>')
                else:
                    template.append(word)
            return ' '.join(template)


        return sorted(set([ extract_(response) for response in responses ]))
