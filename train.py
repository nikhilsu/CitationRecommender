from config_parser import parse_arguments
from models.nn_rank import train as nn_rank
from models.word_embeddings import train as embeddings

if __name__ == '__main__':
    parser = parse_arguments()
    parser.add_argument('--model', choices=['embeddings', 'NNRank'], help="name of the model to train")
    args = parser.parse_args()

    if args.model == 'embeddings':
        embeddings.train(args)
    elif args.model == 'NNRank':
        nn_rank.train(args)
