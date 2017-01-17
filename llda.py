# -*- coding: utf-8 -*-
from optparse import OptionParser
import sys, re, numpy

def load_corpus(filename):
    corpus = []
    labels = []
    labelmap = dict()
    f = open(filename, 'r')
    for line in f:
        mt = re.match(r'\[(.+?)\](.+)', line)
        if mt:
            label = mt.group(1).split(',')
            for x in label: labelmap[x] = 1
            line = mt.group(2)
        else:
            label = None
        doc = re.findall(r'\w+(?:\'\w+)?',line.lower())
        if len(doc)>0:
            corpus.append(doc)
            labels.append(label)
    f.close()
    return labelmap.keys(), corpus, labels

class LLDA:
    def __init__(self, alpha, beta, K = 100):
        self.alpha = alpha
        self.beta = beta
        self.K = K

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def find_term_id(self, term):
        return self.vocas_id.get(term)
                             
    def complement_label(self, label):
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, corpus, labels = None):

        self.labelset = []
        if labels:
            for label in labels:
                self.labelset += label
            self.labelset = list(set(self.labelset))
            self.labelset.insert(0, "common")
            self.labelmap = dict(zip(self.labelset, range(len(self.labelset))))
            self.K = len(self.labelmap)
            self.labels = numpy.array([self.complement_label(label) for label in labels])
        else:
            self.labels = numpy.array([[1.0 for j in range(self.K)] for i in range(len(corpus))])
        
        self.vocas = []
        self.vocas_id = dict()
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)
        V = len(self.vocas)

        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=numpy.int32)
        self.n_z_t = numpy.zeros((self.K, V), dtype=numpy.int32)
        self.n_z = numpy.zeros(self.K, dtype=numpy.int32)

        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self, iteration = 100):
        V = len(self.vocas)
        kalpha = self.K * self.alpha
        vbeta = V * self.beta
        
        for _iter in range(iteration):
            b = 0
            c = 0
            for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
                #start = time.time()
                for n in range(len(doc)):
                    t = doc[n]
                    z = self.z_m_n[m][n]
                    self.n_m_z[m, z] -= 1
                    self.n_z_t[z, t] -= 1
                    self.n_z[z] -= 1

                    denom_b = self.n_z + vbeta

                    p_z = label * (self.n_z_t[:, t] + self.beta) * (self.n_m_z[m] + self.alpha) / denom_b 
                    new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    
                    self.z_m_n[m][n] = new_z
                    self.n_m_z[m, new_z] += 1
                    self.n_z_t[new_z, t] += 1
                    self.n_z[new_z] += 1
                    
            sys.stderr.write("-- %d : %.4f\n" % (_iter, self.perplexity()))

    def folding(self, new_doc = [], label = None, iteration = 50):
        V = len(self.vocas)
        kalpha = self.K * self.alpha
        vbeta = V * self.beta

        if len(label) != 0: label = numpy.array(self.complement_label(label))
        else: label = numpy.array([1.0 for i in range(self.K)])
            
        doc = [self.find_term_id(term) for term in new_doc if self.find_term_id(term) is not None]
        
        n_m_z = numpy.zeros((1, self.K), dtype=numpy.int32)
        n_z_t = numpy.zeros((self.K, V), dtype=numpy.int32)
        n_z = numpy.zeros(self.K, dtype=numpy.int32)

        N_m = len(doc)
        z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
        for t, z in zip(doc, z_n):
            n_m_z[0, z] += 1
            n_z_t[z, t] += 1
            n_z[z] += 1
        
        for _iter in range(iteration):
            for n in range(len(doc)):
                t = doc[n]
                z = z_n[n]
                n_m_z[0, z] -= 1
                n_z_t[z, t] -= 1
                n_z[z] -= 1
				
                denom_b = n_z + self.n_z + vbeta
				
                p_z = label * (self.n_z_t[:, t] + self.beta) * (self.n_m_z[m] + self.alpha) / denom_b 
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                z_n[n] = new_z
                n_m_z[0, new_z] += 1
                n_z_t[new_z, t] += 1
                n_z[new_z] += 1

        labels = numpy.array([label])
        n_alpha = n_m_z + labels * self.alpha
        return (n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis])[0]
        
    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    labelset, corpus, labels = load_corpus(options.filename)

    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, corpus, labels)

    llda.inference(options.iteration)

    phi = llda.phi()
    for v, voca in enumerate(llda.vocas):
        print (','.join([voca]+[str(x) for x in phi[:,v]]))

if __name__ == "__main__":
    main()
