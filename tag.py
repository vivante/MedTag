from importer import *

import tools
from featureBuilder import Model, write
from document import processDoc
from format import formatDocument
import copy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt",dest = "txt")
    parser.add_argument("--out",dest = "out")
    parser.add_argument("--model",dest = "model")
    parser.add_argument("--format",dest = "format")
    args = parser.parse_args()

    if not args.txt or not args.out or not args.model or not os.path.exists(args.model):
    	parser.print_help(sys.stderr)
        sys.stderr.write('\n\tOne or more files/model not provided\n\n')
        sys.stderr.write('\n')
        exit(1)

    file = glob.glob(args.txt)
    tools.mkpath(args.out)

    tag( file, args.model, args.out)


def tag(files, modelPath, outDir):

    with open(modelPath, 'rb') as foo:
        print(foo)
    	model = pickle.load(foo)

    if not files:
        sys.stderr.write( "\n\tInput files not provided\n\n")
        exit()

    n = len( files )

    for k,text in enumerate( sorted( files) ):
        textInst = processDoc(text)

        fileName = os.path.splitext( os.path.basename( text))[0]+'.'+'con'
        outPath = os.path.join( outDir,fileName)

        sys.stdout.write('%s\n' % ('-' * 30))
        sys.stdout.write('\n\t%d of %d\n' % (k+1,n))
        sys.stdout.write('\t%s\n\n' % text)

        labels = model.predictClassesFromDocument(textInst)
        out = textInst.write(labels)
        sys.stdout.write('\n\nwriting to: %s\n' % outPath)
        with open( outPath,'w' ) as foo:
            write( foo, '%s\n' % out)
        sys.stdout.write('\n')

        doc_inst = formatDocument(text, outPath)
        concept = doc_inst.extract()

        print(concept[0])
        print(concept[1])
        print(concept[2])

        doc_inst.format()

if __name__ == '__main__':
    main()