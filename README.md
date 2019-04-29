# CitationRecommender
Project to recommend Citations to a research paper

#### Run code to train word embeddings
- You will need access the dataset which is sits inside a MongoDB instance hosted on the GCP.
 *(Please contact the owner/s of the repo for the credential or the db url.)*
- Once you get access to the MongoDB credentials, please export these credentials in to your systems environment variables.
- Now that is done, run the below commands to start training your Word Embeddings network.
```bash
$ export PYTHONPATH="$PYTHONPATH:$(pwd)"
$ python train.py --model embeddings
```
