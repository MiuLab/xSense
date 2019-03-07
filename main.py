import argparse
from train import trainIters
from evaluate import runTest
# from evaluate_wic import runTest # compute WIC

def parse():
    parser = argparse.ArgumentParser(description='XSense')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-wv', '--wordvec', help='Train the model with pretrained embeddings')
    parser.add_argument('-wf', '--wordfile', help='Train the model with target words')
    parser.add_argument('-sif', '--sif', help='Use sif embedding as encoder input')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-tf', '--testfile', help='test a model with a testing file')
    parser.add_argument('-of', '--outputfile', help='write the result in an output file')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print_every', type=int, default=500, help='Print every p iterations')
    parser.add_argument('-s', '--save_every', type=int, default=5000, help='Save every s iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('-la', '--n_layers', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden_size', type=int, default=300, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse

def run(args):
    if args.train:
        trainIters(args.train, args.wordvec, args.wordfile, args.sif, args.reverse, args.iteration, args.learning_rate, args.batch_size,
                    args.n_layers, args.hidden_size, args.print_every, args.save_every, args.dropout)
    elif args.test:
        n_layers, hidden_size, reverse = parseFilename(args.test, True)
        runTest(n_layers, hidden_size, reverse, args.test, args.corpus, args.batch_size, args.testfile, args.outputfile, args.sif)


if __name__ == '__main__':
    args = parse()
    run(args)
