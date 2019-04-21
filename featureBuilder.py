from importer import *

from sklearn.feature_extraction  import DictVectorizer
from time import localtime, strftime

from document import labels as tag2id, id2tag
from tools import flatten, save_list_structure, reconstruct_list
from tools import print_str, print_vec, print_files, write
from feats import extractFeatures
import crf


taggerDir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tempDir = os.path.join(taggerDir,'data','tmp')

class Model:

    def __init__(self):
        
        self._is_trained     = False
        self._clf            = "latin1"
        self._vocab          = None
        self._training_files = None
        self._log            = None
        self._text_feats     = None

    def train(self, trainNotes, val=[], test=[]):
        trainSents=flatten([k.getTokenizedSentences() for k in trainNotes])
        trainLabels=flatten([k.getTokenLabels() for k in trainNotes])

        if test:
            testSents=flatten([k.getTokenizedSentences() for k in test])
            testLabels=flatten([k.getTokenLabels() for k in test])
        else:
            testSents=[]
            testLabels=[]
        
        if val:
            valSents=flatten([k.getTokenizedSentences() for k in val])
            valLabels=flatten([k.getTokenLabels() for k in val])
            self.trainFit(trainSents,trainLabels,valSents=valSents,valLabels=valLabels,testSents=testSents,testLabels=testLabels)
        else:
            self.trainFit(trainSents, trainLabels, devSplit=0.1, testSents=testSents, testLabels=testLabels)
        
        self._trainFiles=[k.getName() for k in trainNotes+val]

    def trainFit(self, trainSents, trainLabels, valSents=None, valLabels=None, testSents=None, testLabels=None, devSplit=None):
        self._time_train_begin=strftime("%Y-%m-%d %H:%M:%S", localtime())

        voc, clf, devScore, enabledFeatures = genericTrain('all', trainSents, trainLabels, valSents=valSents,valLabels=valLabels, testSents=testSents, testLabels=testLabels, devSplit=devSplit)
        self._is_trained = True
        self._vocab = voc
        self._clf   = clf
        self._score = devScore
        self._features = enabledFeatures
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())

    def predictClassesFromDocument(self, doc):
         tokenizedSents  = doc.getTokenizedSentences()
         return self.predictClasses(tokenizedSents)
          
    def predictClasses(self, tokenizedSents):
        hyperparams = {}
        vectorizedPred = genericPredict('all',tokenizedSents,vocab= self._vocab,clf= self._clf,hyperparams=hyperparams)
        iobPred = [[id2tag[p] for p in seq] for seq in vectorizedPred]
        return iobPred

def genericTrain(p_or_n, trainSents, trainLabels, valSents=None, valLabels=None, testSents=None, testLabels=None, devSplit=None):
    if len(trainSents) == 0:
        raise Exception('Can not train on %s training examples' % p_or_n)
    if (not valSents) and (devSplit > 0.0) and (len(trainSents)>10):
        p = int(devSplit*100)
        sys.stdout.write('\tCreating %d/%d train/devSplit\n' % (100-p,p))
        permutations = list(range(len(trainSents)))
        random.shuffle(permutations)
        trainSents = [trainSents[i] for i in permutations]
        trainLabels = [trainLabels[i] for i in permutations]

        index = int(devSplit*len(trainSents))

        valSents = trainSents[:index]
        trainSents = trainSents[index:]

        valLabels = trainLabels[:index]
        trainLabels = trainLabels[index:]
    else:
        sys.stdout.write('\tUsing the existing validation data\n')

    sys.stdout.write('\tvectorizing the words %s\n' % p_or_n)

        ################
    textFeats = extractFeatures(trainSents)
    #############################
    enabledFeats = set()
    for sf in textFeats:
        for wf in sf:
            for (featType,inst),val in wf.items():
                if featType.startswith('prev'):
                    featType = 'PREV*'
                if featType.startswith('next'):
                    featType = 'NEXT*'
                enabledFeats.add(featType)
    enabledFeats = sorted(enabledFeats)

    vocab = DictVectorizer()
    flatXFeats = vocab.fit_transform( flatten(textFeats) )
    XFeats = reconstruct_list(flatXFeats,save_list_structure(textFeats))

    print(trainLabels)

    YLabels = [ [tag2id[s] for s in ySeq] for ySeq in trainLabels ]

    print("theeh")

    assert len(XFeats) == len(YLabels)
    for k in range(len(XFeats)):
        assert XFeats[k].shape[0] == len(YLabels[k])

    if valSents:
        valTextFeats = extractFeatures(valSents)
        flatValXFeats = vocab.transform(flatten(valTextFeats) )
        valX = reconstruct_list(flatValXFeats,save_list_structure(valTextFeats))
        valY = [ [tag2id[s] for s in ySeq] for ySeq in valLabels ]
    if testSents:
        testTextFeats = extractFeatures(testSents)
        flatTestXFeats = vocab.transform(flatten(testTextFeats) )
        testX = reconstruct_list(flatTestXFeats,save_list_structure(testTextFeats))
        testY = [ [tag2id[s] for s in ySeq] for ySeq in testLabels ]
    else:
        testX = None
        testY = None

    sys.stdout.write('\ttraining classifiers %s\n' % p_or_n)
    
    clf, devScore  = crf.train(XFeats, YLabels, valX=valX, valY=valY,testX=testX, testY=testY)

    return vocab, clf, devScore, enabledFeats

def genericPredict(p_or_n, tokenizedSents, vocab, clf, hyperparams):
    if len(tokenizedSents) == 0:
        sys.stderr.write('\tNothing to predict %s\n' % p_or_n)
        return []

    sys.stdout.write('\tVectorizing the words %s\n' % p_or_n)
    textFeatures = extractFeatures(tokenizedSents)
    flatXFeats = vocab.transform( flatten(textFeatures) )
    X = reconstruct_list(flatXFeats, save_list_structure(textFeatures))
    sys.stdout.write('\tPredicting the labels %s\n' % p_or_n)
    predictions =   crf.predict(clf, X)
    return predictions
    
