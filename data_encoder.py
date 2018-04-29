

from __future__ import print_function

import numpy as np

def get_context_left(self,context_left,embedding_previous):
        
        left_c=tf.matmul(context_left,self.W_l) 
        left_e=tf.matmul(embedding_previous,self.W_sl)
        left_h=left_c+left_e
        context_left=self.activation(left_h)
        return context_left

def get_context_right(self,context_right,embedding_afterward):
    
    right_c=tf.matmul(context_right,self.W_r)
    right_e=tf.matmul(embedding_afterward,self.W_sr)
    right_h=right_c+right_e
    context_right=self.activation(right_h)
    return context_right



class Embedder(object):
    

    def map_tokens(self, tokens, ndim=2):
        
        gtokens = [self.g[self.w[t]] for t in tokens if t in self.w]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def map_jset(self, sj):
        return self.g[sj]

    def pad_set(self, ss, spad, N=None):
        
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)


class Encode(Embedder):
    
    def __init__(self, N=50, glovepath='glove.6B/glove.6B.%dd.txt'):
        
        self.N = N
        self.w = dict() 
        self.g = []  
        self.glovepath = glovepath % (N,)

        self.g.append(np.zeros(self.N))

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.w[word] = len(self.g)
                self.g.append(np.array(l[1:]).astype(float))
        self.w['UKNOW'] = len(self.g)
        self.g.append(np.zeros(self.N))
        self.g = np.array(self.g, dtype='float32') #converting to numpy vector
