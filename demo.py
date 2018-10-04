import sys, string, random, numpy
from nltk.corpus import reuters
from llda import LLDA
from optparse import OptionParser
from functools import reduce

parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=1)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.01)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=0)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()
random.seed(options.seed)
numpy.random.seed(options.seed)

idlist = random.sample(reuters.fileids(), options.samplesize)

labels = []
corpus = []
for id in idlist:
    labels.append(reuters.categories(id))
    corpus.append([x.lower() for x in reuters.words(id) if x[0] in string.ascii_letters])
    reuters.words(id).close()
labelset = list(set(reduce(list.__add__, labels)))

options.alpha = 50 / (len(labelset) + 1)
llda = LLDA(options.alpha, options.beta, options.K)
llda.set_corpus(corpus, labels)

print("M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), options.K))

llda.inference(options.iteration)

phi = llda.phi()
for k, label in enumerate(labelset):
    print ("\n-- label %d : %s" % (k + 1, label))
    for w in numpy.argsort(-phi[k + 1])[:10]:
        print("%s: %.4f" % (llda.vocas[w], phi[k + 1,w]))
