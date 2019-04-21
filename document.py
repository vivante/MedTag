from importer import *

from tools import clean_text, normalize_tokens

labels = {'O':0,'B-problem':1,'B-test':2,'B-treatment':3,'I-problem':4, 'I-test':5, 'I-treatment':6,}

id2tag={v:k for k,v in labels.items()}

class processDoc:

    def __init__(self, text, concept=None):
        returnVal=readDocs(text, concept)
        self._tokenSents=returnVal[0]

        if concept:
            self._tokenConcepts=returnVal[1]
            self._labels=tokenConceptsToLabels(self._tokenSents,self._tokenConcepts)

        self._fileName=text

    
    def getName(self):
        return os.path.basename(self._fileName).split('.')[0]


    # def getExtension(self):
    #     return 'con'


    def getTokenizedSentences(self):
        return self._tokenSents


    def getTokenLabels(self):
        return self._labels


    def conlist(self):
        return self._labels

    
    def write(self,predLabels=None):
        returnString=''
        
        if predLabels!=None:
            tokenLabels=predLabels
        elif self._labels!=None:
            tokenLabels=self._labels
        else:
            raise Exception("Error writing concepts")

        tuples=tokenLabelsToConcepts(self._tokenSents,tokenLabels)

        for tup in tuples:
            if tup[0]=='none':
                raise('None label found')

            concept=tup[0]
            lineNum=tup[1]
            beginPos=tup[2]
            lastPos=tup[3]
            text=self._tokenSents[lineNum - 1]

            datum = text[beginPos]
            for j in range(beginPos, lastPos):
                datum += " " + text[j+1]
            datum = datum.lower()

            indexOne = "%d:%d" % (lineNum, beginPos)
            indexTwo = "%d:%d" % (lineNum, lastPos)

            label=concept
            returnString+="c=\"%s\" %s %s||t=\"%s\"\n" % (datum,indexOne,indexTwo,label)
            
        return returnString.strip()


def readDocs(txt, concept):
    tokenizedSentences=[]
    sentTokenize=lambda text: text.split('\n')
    wordTokenize=lambda text: text.split(' ')

    with open(txt) as foo:
        text=foo.read().strip('\n')
        sentences=sentTokenize(text)
        for s in sentences:
            sent=clean_text( s.rstrip())
            sent=sent.lower()
            tokens=wordTokenize(sent)
            normedTokens=normalize_tokens( tokens)
            tokenizedSentences.append(normedTokens)
    
    tokenizedConcepts=[]
    if concept:
        with open(concept) as foo:
            for l in foo.readlines():
                if not l.strip():
                    continue
                
                conceptRegex = '^c="(.*)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*)"$'
                match = re.search(conceptRegex, l.strip())
                groups = match.groups()

                concept_text  = groups[0]
                beginLineNum  = int(groups[1])
                beginTokenIndex = int(groups[2])
                lastLineNum    = int(groups[3])
                lastTokenIndex   = int(groups[4])
                conceptLabel = groups[5]

                assert beginLineNum==lastLineNum, 'concept must span single line'

                tup = (conceptLabel, beginLineNum, beginTokenIndex, lastTokenIndex)
                tokenizedConcepts.append(tup)

        tokenizedConcepts=list(set(tokenizedConcepts))
        tokenizedConcepts=sorted(tokenizedConcepts,key=lambda t:t[1:])

        # Ensure no overlapping concepts (that would be bad)
        for i in range(len(tokenizedConcepts)-1):
            c1 = tokenizedConcepts[i]
            c2 = tokenizedConcepts[i+1]
            if c1[1] == c2[1]:
                if c1[2] <= c2[2] and c2[2] <= c1[3]:
                    fname = os.path.basename(con)
                    error1='%s has overlapping entities on line %d'%(fname,c1[1])
                    error2="It can't be processed until you remove one"
                    error3='Please modify this file: %s' % con
                    error4='\tentity 1: c="%s" %d:%d %d:%d||t="%s"'%(' '.join(tokenizedSentences[c1[1]-1][c1[2]:c1[3]+1]),
                                                                     c1[1], c1[2], c1[1], c1[3], c1[0])
                    error5='\tentity 2: c="%s" %d:%d %d:%d||t="%s"'%(' '.join(tokenizedSentences[c2[1]-1][c2[2]:c2[3]+1]),
                                                                     c2[1], c2[2], c2[1], c2[3], c2[0])
                    error_msg = '\n\n%s\n%s\n\n%s\n\n%s\n%s\n' % (error1,error2,error3,error4,error5)
                    raise DocumentException(error_msg)

    return tokenizedSentences, tokenizedConcepts

def tokenConceptsToLabels(tokenizedSentences, tokenizedConcepts):
    labels=[['O' for tokens in sentences] for sentences in tokenizedSentences]
    for concept in tokenizedConcepts:
        label, lineNum, beginTokenIndex, lastTokenIndex = concept
        labels[lineNum-1][beginTokenIndex]='B-%s' % label
        for k in range(beginTokenIndex+1, lastTokenIndex+1):
            labels[lineNum-1][k]='I-%s' % label

    return labels 

def tokenLabelsToConcepts(tokenizedSentences, tokenLabels):
    def splitLabel(label):
        if label=='O':
            iob,tag='O',None
        else:
            iob, tag=label.split('-')
        return iob, tag
    
    corr=[]
    for lineNum, labels in enumerate(tokenLabels):
        corrLine=[]
        for k in range(len(labels)):
            iob, tag=splitLabel(labels[k])
            if iob is 'I':
                if k is 0:
                    print('Correcting A')
                    newLabel='B'+labels[k][1:]
                else:
                    prevIOB, prevTag=splitLabel(labels[k-1])
                    if prevIOB is 'O' or prevTag != tag:
                        print('Correcting B')
                        newLabel='B'+labels[k][1:]
                    else:
                        newLabel=labels[k]
            else:
                newLabel=labels[k]
            corrLine.append(newLabel)
        corr.append(corrLine)
    
    tokenLabels=corr
    concepts=[]
    for k, labels in enumerate(tokenLabels):
        N=len(labels)
        start=[j for j, label in enumerate(labels) if label[0] is 'B']
        for begin in start:
            labs=labels[begin][1:]
            last=begin
            while(last<N-1) and tokenLabels[k][last+1].startswith('I') and tokenLabels[k][last+1][1:]==labs:
                last+=1
            conceptTuple=(labs[1:], k+1, begin, last)
            concepts.append(conceptTuple)

    testTokenLabels=tokenConceptsToLabels(tokenizedSentences, concepts)

    for lineNum,(test,gold,sent) in enumerate(zip(testTokenLabels, tokenLabels, tokenizedSentences)):
        for k,(x,y) in enumerate(zip(test,gold)):
            if not ((x==y) or (x[0]=='B' and y[0]=='I' and x[1:]==y[1:])):
                print()
                print( 'lineno:    ', lineno)
                print()
                print( 'generated: ', test[i-3:i+4])
                print( 'predicted: ', gold[i-3:i+4])
                print( sent[i-3:i+4])
                print( 'x[0]:  ', x[0])
                print( 'y[0]:  ', y[0])
                print( 'x[1:]: ', x[1:])
                print( 'y[1:]: ', y[1:])
                print( 'x[1:] == b[a:]: ', x[1:] == y[1:])
                print()
            assert (x == y) or (x[0]=='B' and y[0]=='I' and x[1:]==y[1:])
            k += 1

    assert testTokenLabels==tokenLabels
    return concepts

    class DocumentException(Exception):
        pass