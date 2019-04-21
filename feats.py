from importer import *
import wordFeatures as feat_word


# TAGGER_DIR = dname(dname(dname(os.path.abspath(__file__))))
dname = os.path.dirname
TAGGER_DIR = dname(os.path.abspath(__file__))

print(TAGGER_DIR)

tagger_name = 'py%d_maxent_treebank_pos_tagger.pickle' % sys.version_info.major
pos_tagger_path = os.path.join(TAGGER_DIR, 'tools', tagger_name)

def loadPosTagger(pathToObj=pos_tagger_path):
    tagger = loadPickledObj(pathToObj)
    return tagger

def loadPickledObj(pathToPickledObj):
    data = None
    with open(pathToPickledObj, "rb") as f:
        data = f.read()
    return pickle.loads(data)

def enabledModules():
    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(DIR, 'config.txt' )
    f = open( filename, 'r' )
    specs = {}
    moduleList=['GENIA']
    for line in f.readlines():
        words=line.split()
        if words:
            if words[0] in moduleList:
                if words[1]=='None':
                    specs[words[0]]=None
                else:
                    specs[words[0]] = os.path.expandvars(words[1]).strip('\"').strip('\'')

    if specs["GENIA"] != None:
        if os.path.isfile(specs["GENIA"]) == False:
            sys.exit("Invalid genia directory")
    return specs

enabled = enabledModules()
featGenia=None
if enabled['GENIA']:
    from .genia_dir.genia_features import GeniaFeatures

nltkTagger = loadPosTagger()

enabledSentFeats = []
enabledSentFeats.append('unigram_context')
enabledSentFeats.append('pos')
enabledSentFeats.append('pos_context')
enabledSentFeats.append('prev')
enabledSentFeats.append('prev2')
enabledSentFeats.append('next')
enabledSentFeats.append('next2')
enabledSentFeats.append('GENIA')

def extractFeatures(tokSents):
    sentenceFeaturesPreprocess(tokSents)
    proseFeats = []
    for sent in tokSents:
       proseFeats.append(extractFeaturesSentence(sent))
    return proseFeats

def sentenceFeaturesPreprocess(data):
    global featGenia
    tagger=enabled['GENIA']
    if tagger:
        featGenia = GeniaFeatures(tagger,data)

def extractFeaturesSentence(sent):
    featuresList = []
    for i,word in enumerate(sent):
        featuresList.append(feat_word.IOBProseFeatures(sent[i]))

    if 'unigram_context' in enabledSentFeats:
        size = 3
        n = len(sent)
        for i in range(n):
            end = min(i, size)
            unigrams = sent[i-end:i]
            for j,u in enumerate(unigrams):
                featuresList[i][('prev_unigrams-%d'%j,u)] = 1
        for i in range(n):
            end = min(i + size, n-1)
            unigrams = sent[i+1:end+1]
            for j,u in enumerate(unigrams):
                featuresList[i][('next_unigrams-%d'%j,u)] = 1

    if 'pos' in enabledSentFeats:
        posTagged = nltkTagger.tag(sent)

    for feature in enabledSentFeats:
        if feature == 'pos':
            for (i,(_,pos)) in enumerate(posTagged):
                featuresList[i].update( { ('pos',pos) : 1} )

        if 'pos_context' in enabledSentFeats:
            size = 3
            n = len(sent)
            for i in range(n):
                end = min(i, size)
                for j,p in enumerate(posTagged[i-end:i]):
                    pos = p[1]
                    featuresList[i][('prev_pos_context-%d'%j,pos)] = 1

            for i in range(n):
                end = min(i + size, n-1)
                for j,p in enumerate(posTagged[i+1:i+end+1]):
                    pos = p[1]
                    featuresList[i][('next_pos_context-%d'%j,pos)] = 1

        if (feature == 'GENIA') and enabled['GENIA']:
            geniaFeatList = feat_genia.features(sent)
            for i,featDict in enumerate(geniaFeatList):
                featuresList[i].update(featDict)

    ngram_features = [ {} for i in range(len( featuresList ))]
    if "prev" in enabledSentFeats:
        prev = lambda f: {( "prev_"+k[0], k[1]): v for k, v in f.items() }
        prev_list = list( map( prev, featuresList))
        for i in range( len( featuresList)):
            if i==0:
                ngram_features[i][( "prev", "*" )] = 1
            else:
                ngram_features[i].update( prev_list[i-1] )

    if "prev2" in enabledSentFeats:
        prev2 = lambda f: {( "prev2_"+k[0], k[1]): v/2.0 for k, v in f.items() }
        prev_list = list( map( prev2, featuresList) )
        for i in range( len( featuresList ) ):
            if i == 0:
                ngram_features[i][("prev2", "*")] = 1
            elif i == 1:
                ngram_features[i][("prev2", "*")] = 1
            else:
                ngram_features[i].update(prev_list[i-2])

    if "next" in enabledSentFeats:
        next = lambda f: { ( "next_"+k[0], k[1] ): v for k, v in f.items() }
        next_list = list(map(next, featuresList))
        for i in range(len(featuresList)):
            if i < len(featuresList) - 1:
                ngram_features[i].update(next_list[i+1])
            else:
                ngram_features[i][("next", "*")] = 1

    if "next2" in enabledSentFeats:
        next2 = lambda f: { ( "next2_"+k[0], k[1] ): v/2.0 for k, v in f.items()}
        next_list = list( map( next2, featuresList ) )
        for i in range( len( featuresList ) ):
            if i < len( featuresList ) - 2:
                ngram_features[i].update(next_list[i+2])
            elif i==len(featuresList) - 2:
                ngram_features[i][( "next2", "**" )] = 1
            else:
                ngram_features[i][("next2", "*")] = 1

    merged=lambda d1,d2: dict( list( d1.items() ) + list( d2.items() ))
    featuresList=[ merged( featuresList[i], ngram_features[i])
        for i in range(len(featuresList))]

    return featuresList