import argparse

from models.word_embeddings import train_embedding_model as embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['embeddings', 'NNRank', 'NNSelect'], help="name of the model to train")
args = parser.parse_args()

if args.model == 'embeddings':
    embeddings.train()
