# Citation Recommender
Project to recommend Citations given a document. The scope of the project is to replicate the research done as part of [Content-Based Citation Recommendation](https://arxiv.org/pdf/1802.08301.pdf) paper.

Credits: [citeomatic (GitHub)](https://github.com/allenai/citeomatic/)

### Project Dependencies
- The project dependencies(python libraries) can be installed by running the following command:-
```bash
$ pip install -r requirements.txt
```


### Order of Training Models

#### Train word embeddings
- You will need access the dataset which is sits inside a MongoDB instance hosted on the GCP.
 *(Please contact the owner/s of the repo for the credential or the db url.)*
- Once you get access to the MongoDB credentials, please export these credentials in to your systems environment variables.
- Now that that is done, run the below commands to start training your Word Embeddings network.
```bash
$ export PYTHONPATH="$PYTHONPATH:$(pwd)"
$ python train.py --model embeddings
```



#### Train NN_Rank model
- NNRank Model depends on the word_embedding model's and the dense model's weight.
- Thus, you would need to pre-train the word_embeddings model before you train the NNRank.
- The embedding and the dense weights are found in the `weights` directory once the training of the Word embeddings model is complete.
- Run the below commands to start training your NN_Rank network.
```bash
$ export PYTHONPATH="$PYTHONPATH:$(pwd)"
$ python train.py --model NNRank --embeddings_model_weights_path "<path_to_embeddings_model_weight>" --dense_model_weights_path "<path_to_dense_model_weights>"
```


### Testing Model

#### Testing NN_Select Model
- NN_Select or the Candidate Selector model can be tested by utilizing any of word_embedding_model weights in the `weights` directory.
*(Note: These model weight files will be generated on training the Word Embeddings model)*
- Run the following command to test the model

```bash
$ python test.py --model NNSelect --embeddings_model_weights_path "<path_to_weights>"
```


#### Testing NN_Rank Model
- NN_Rank model can be tested by utilizing nn_rank_weights in the `weights` directory.
- Run the following command to test the model
- This model is evaluated based on various Metrics like:-
    1. Precision
    2. Accuracy
    3. Mean Reciprocal Rank
    4. Recall
    5. F1 measure

```bash
$ python test.py --model NNRank --nn_rank_model_weights_path "<path_to_nn_rank_weights>"
```