import argparse
from train import train
from evaluate import runTest
# from evaluate_wic import runTest # compute WIC

def main():
    parser = argparse.ArgumentParser(description='XSense-SIF')
    parser.add_argument('--run', type=str, default="train", help='train/ test')
    parser.add_argument('--corpus', type=str, default="data/train.txt", help='Train the model with corpus')
    parser.add_argument('--w2v_file', type=str, default="data/my_google_news_train.txt", help='Train the model with pretrained embeddings')
    parser.add_argument('--sif_file', type=str, default="SIF/my_train_sif", help='Use sif embedding as encoder input')
    parser.add_argument('--model_path', type=str, default="save/model/xSense_25.tar", help='path for checkpoint')
    parser.add_argument('--save_dir', type=str, default="save", help='save directory')
    parser.add_argument('--epoch', type=int, default=25, help='Train the model with * epoches')
    parser.add_argument('--print_every', type=int, default=500, help='Print every p iterations')
    parser.add_argument('--save_every', type=int, default=5, help='Save every s iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size in encoder and decoder')
    parser.add_argument('--K', type=int, default=5, help='number of activated dimensions used in SPINE')
    parser.add_argument('--lr', type=float, default=5*1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')

    args = parser.parse_args()

    if args.run == 'train':
        train(args)
    else:
        runTest(args)


if __name__ == '__main__':
    main()
