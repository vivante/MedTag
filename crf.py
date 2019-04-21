from importer import *
from tools import compute_performance_stats

TAGGER_DIR = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(TAGGER_DIR, 'data', 'tmp')

# TAGGER_DIR = dname(dname(dname(os.path.abspath(__file__))))

def enabledModules():
    filename = os.path.join(TAGGER_DIR, 'config.txt' )
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
            sys.exit("No such genia directory")
    return specs

def formatFeatures(rows, labels=None):
    returnVal=[]
    for i, line in enumerate( rows ):
        for j, feats in enumerate( line ):
            inds=feats.nonzero()[1]
            values = []
            if labels:
                values.append( str(labels[i][j]) )
            for k in inds:
                values.append( '%d=%d' %  (k, feats[0, k]) )
            returnVal.append("\t".join(values).strip())
        returnVal.append('')
    return returnVal

def pycrfInstances(fi, labeled):
    xseq = []
    yseq = []
    if labeled:
        begin = 1
    else:
        begin = 0
    for line in fi:
        line=line.strip('\n')
        if not line:
            if labeled:
                yield xseq,tuple(yseq)
            else:
                yield xseq
            xseq = []
            yseq = []
            continue
        fields=line.split('\t')
        feats=fields[begin:]
        xseq.append(feats)
        if labeled:
            yseq.append( fields[0] )

def train(X, Y, valX=None, valY=None, testX=None, testY=None):
    feats = formatFeatures( X, Y )
    trainer = pycrfsuite.Trainer( verbose=False )
    for xseq, yseq in pycrfInstances(feats,labeled=True):
        trainer.append(xseq, yseq)
    os_handle, tmp_file=tempfile.mkstemp( dir=tmp_dir, suffix="crf_temp" )
    trainer.train( tmp_file )
    model=''
    with open( tmp_file, 'rb' ) as f:
        model = f.read()
    os.close( os_handle )
    os.remove( tmp_file )
    scores={}
    train_pred=predict(model,X)
    train_stats=compute_performance_stats('train',train_pred,Y)
    scores['train']=train_stats

    if valX:
        val_pred=predict( model, valX )
        val_stats=compute_performance_stats('dev',val_pred,valY)
        scores['dev']=val_stats

    if testX:
        test_pred=predict(model, testX)
        test_stats=compute_performance_stats('test', test_pred, testY)
        scores['test']=test_stats
    scores['hyperparams'] = {}
    enabled_mods = enabledModules()
    for module,enabled in enabled_mods.items():
        e=bool(enabled)
        scores['hyperparams'][module] = e
    return model, scores

def predict(clf, X):
    feats = formatFeatures(X)
    os_handle, tmp_file=tempfile.mkstemp( dir=tmp_dir,suffix="crf_temp" )
    with open(tmp_file,'wb') as f:
        clf_byte=bytearray(clf)
        f.write(clf_byte)
    tagger=pycrfsuite.Tagger()
    tagger.open( tmp_file )
    os.close( os_handle )
    os.remove( tmp_file )
    returnVal = []
    Y = []
    for xseq in pycrfInstances( feats,labeled=False ):
        yseq=[int(n) for n in tagger.tag(xseq)]
        returnVal += list(yseq)
        Y.append(list(yseq))
    return Y