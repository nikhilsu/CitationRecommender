import argparse

from models.word_embeddings import train as embeddings


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['embeddings', 'NNRank', 'NNSelect'], help="name of the model to train")
    parser.add_argument('--batch_size', default=32, required=False)
    parser.add_argument('--epochs', default=1000, required=False)
    parser.add_argument('--train_split', default=0.85, required=False)
    parser.add_argument('--steps_per_epoch', default=47347 // 32, required=False)
    parser.add_argument('--weights_directory', default='weights', required=False)
    parser.add_argument('--checkpoint_frequency', default=3, required=False)
    parser.add_argument('--max_features', default=200000, required=False)
    parser.add_argument('--max_title_len', default=50, required=False)
    parser.add_argument('--max_abstract_len', default=500, required=False)
    parser.add_argument('--dense_dims', default=150, required=False)
    parser.add_argument('--n_features', default=500, required=False)
    parser.add_argument('--learning_rate', default=0.001, required=False)
    parser.add_argument('--l1_lambda', default=9.9999999999999995e-07, required=False)
    parser.add_argument('--l2_lambda', default=0.0, required=False)
    parser.add_argument('--dropout_p', default=0.1, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.model == 'embeddings':
        embeddings.train(args)
