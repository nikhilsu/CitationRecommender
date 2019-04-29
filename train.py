from config_parser import parse_arguments
from models.nn_rank import train as nn_rank
from models.word_embeddings import train as embeddings

if __name__ == '__main__':
    parser = parse_arguments()
    parser.add_argument('--model', choices=['embeddings', 'NNRank'], help="name of the model to train")
    parser.add_argument('--embeddings_model_weights_path', help=".h5 file containing weights", required=False)
    parser.add_argument('--dense_model_weights_path', help=".h5 file containing weights", required=False)
    args = parser.parse_args()

    if args.model == 'embeddings':
        embeddings.train(args)
    elif args.model == 'NNRank':
        if not args.embeddings_model_weights_path or not args.dense_model_weights_path:
            parser.error('model weights required while training NNRank')
        nn_rank.train(args)
