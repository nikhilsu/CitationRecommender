# CitationRecommender
Project to recommend Citations to a research paper

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
- Run the below commands to start training your NN_Rank network.
```bash
$ export PYTHONPATH="$PYTHONPATH:$(pwd)"
$ python train.py --model NNRank
```


### Testing Model

#### Testing NN_Select Model
- NN_Select or the Candidate Selector model can be tested by utilizing any of word_embedding_model weights in the `checkpoint` directory.
*(Note: These model weight files will be generated on training the Word Embeddings model)*
- Run the following command to test the model

```bash
$ python test.py --model NNSelect --embeddings_model_weights_path <path_to_weights>
```
