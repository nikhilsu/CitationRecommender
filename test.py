from config_parser import parse_arguments
from dataset_generator.raw_dataset import RawDataset
from dataset_generator.word_embeddings.document_featurizer import DocumentFeaturizer
from models.nn_rank.nn_rank import NNRank
from models.nn_select.candidate_selector import CandidateSelector
from models.word_embeddings.dense_embedding_model import DenseEmbeddingModel

parser = parse_arguments()
parser.add_argument('--model', choices=['NNSelect', 'NNRank'], help="name of the model to run")
parser.add_argument('--embeddings_model_weights_path', help=".h5 file containing weights")
parser.add_argument('--nn_rank_model_weights_path', help=".h5 file containing weights")
args = parser.parse_args()

if args.model == 'NNSelect':
    if not args.embeddings_model_weights_path:
        parser.error('embeddings_model_weights required to test model!')
    raw_dataset = RawDataset()
    featurizer = DocumentFeaturizer(raw_dataset, args)
    dense_embedding_model = DenseEmbeddingModel(featurizer, args)
    dense_embedding_model.load_embedding_model_weights(args.embeddings_model_weights_path)
    candidate_selector = CandidateSelector(raw_dataset, dense_embedding_model, args.knn)
    query_doc = raw_dataset.fetch_random_document()
    print('Fetching candidates of doc - Id: {}, Title: "{}"'.format(query_doc['id'], query_doc['title']))
    for candidate_doc, sim_score in candidate_selector.fetch_candidates_with_similarities(query_doc):
        print('Sim Score: {}, Id: {}, Title: "{}"'.format(sim_score, candidate_doc['id'], candidate_doc['title']))

elif args.model == 'NNRank':
    if not args.nn_rank_model_weights_path:
        parser.error('nn_rank_model_weights required to test model!')
    raw_dataset = RawDataset()
    featurizer = DocumentFeaturizer(raw_dataset, args)
    dense_embedding_model = DenseEmbeddingModel(featurizer, args)
    nn_rank_model = NNRank(dense_embedding_model.nn_rank_model, args)
    nn_rank_model.load_nn_rank_model_weights(args.nn_rank_model_weights_path)


