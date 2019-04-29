import argparse

from models.word_embeddings import train as embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['embeddings', 'NNRank', 'NNSelect'], help="name of the model to train")
parser.add_argument('--batch_size', default=32)
parser.add_argument('--epochs', default=1000)
parser.add_argument('--train_split', default=0.85)
parser.add_argument('--steps_per_epoch', default=47347 // 32)
parser.add_argument('--max_features', default=200000)
parser.add_argument('--max_title_len', default=50)
parser.add_argument('--max_abstract_len', default=500)
args = parser.parse_args()

if args.model == 'embeddings':
    embeddings.train(args)
