from importer import *

import tools
from featureBuilder import Model
from document import processDoc


TAGGER_DIR=os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--txt",dest="txt")
    parser.add_argument("--annotations",dest = "con",)
    parser.add_argument("--val-txt",dest="val_txt")
    parser.add_argument("--val-annotations",dest="val_con")
    parser.add_argument("--test-txt",dest="test_txt")
    parser.add_argument("--test-annotations",dest="test_con")
    parser.add_argument("--model",dest="model")
    parser.add_argument("--log",dest="log")
    # parser.add_argument("--format",dest="format")

    args=parser.parse_args()

    if (not args.txt or not args.con or not args.model):
        parser.print_help(sys.stderr)
        sys.stderr.write('\n\tError in parsing arguments\n')
        sys.stderr.write('\n')
        exit(1)

    m_dir=os.path.dirname(args.model)

    if (not os.path.exists(m_dir)) and (m_dir != ''):
        parser.print_help(sys.stderr)
        sys.stderr.write('\n\tNo such model directory:%s\n' % m_dir)
        sys.stderr.write('\n')
        exit(1)

    textFiles=glob.glob(args.txt)
    conceptFiles=glob.glob(args.con)

    textFilesMap=tools.map_files(textFiles)
    conceptFilesMap=tools.map_files(conceptFiles)

    trainingList=[]

    for k in textFilesMap:
        if k in conceptFilesMap:
            trainingList.append((textFilesMap[k], conceptFilesMap[k]))

    if args.val_txt and args.val_con:
        valTextFiles = glob.glob(args.val_txt)
        valConceptFiles = glob.glob(args.val_con)

        valTextFilesMap = tools.map_files(valTextFiles) 
        valConceptFilesMap = tools.map_files(valConceptFiles)
        
        valList = []
        for k in valTextFilesMap:
            if k in valConceptFilesMap:
                valList.append((valTextFilesMap[k], valConceptFilesMap[k]))
    else:
        valList=[]

    
    if args.test_txt and args.test_con:
        testTextFiles = glob.glob(args.test_txt)
        testConceptFiles = glob.glob(args.test_con)

        testTextFilesMap = tools.map_files(testTextFiles)
        testConceptFilesMap = tools.map_files(testConceptFiles)

        testList=[]
        for k in testTextFilesMap:
            if k in testConceptFilesMap:
                testList.append((testTextFilesMap[k], testConceptFilesMap[k]))
    else:
        testList=[]
    
    build(trainingList, args.model, logFile=args.log, val=valList, test=testList)

def build(trainingList, modelPath, logFile=None, val=[], test=[]):
    trainDocs=[]
    for text, concept in trainingList:
        tempDoc=processDoc(text, concept)
        trainDocs.append(tempDoc)

    valDocs=[]
    for text, concept in val:
        tempDoc=processDoc(text, concept)
        valDocs.append(tempDoc)

    testDocs=[]
    for text, concept in test:
        tempDoc=processDoc(text, concept)
        testDocs.append(tempDoc)

    if not trainDocs:
        sys.stderr.write('\n\tNo documents found to train the model\n')
        sys.stderr.write('\n')
        exit(1)

    model=Model()

    model.train(trainDocs,val=valDocs,test=testDocs)

    print('\nModel saved to: %s\n' % modelPath)
    with open(modelPath, "wb") as m_file:
        pickle.dump(model, m_file)

    # model.log(logfile   , model_file=model_path)
    # model.log(sys.stdout, model_file=model_path)
    

if __name__ == '__main__':
    main()