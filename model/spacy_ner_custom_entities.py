from __future__ import unicode_literals, print_function
import pickle
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding, decaying

def train_model(model=None, new_model_name='', output_dir=None, n_iter=1000):

    # Loading training data
    with open('output/temp/train', 'rb') as fp:
        train_data = pickle.load(fp)

    # Setting up the pipeline and entity recognizer, and training the new entity
    if model is not None:
        # load existing spacy model
        nlp = spacy.load(model)
    else:
        # create blank Language class
        nlp = spacy.blank('en')

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    # Add Labels
    ner.add_label("tag")
    labels = ['I-indications', 'B-indications', 'O']
    for i in labels:
        ner.add_label(i)

        # start training or optimizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Dropout values
    dropout = decaying(0.1, 0.001, 0.1)

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # only train NER
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}

            # Set mini batch
            batches = minibatch(train_data, size=compounding(4., 32., 0.001))

            # Update Losses
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=next(dropout), losses=losses)
            print('Losses', losses)

    # Save model
    output_dir = Path(output_dir)
    nlp.meta['name'] = new_model_name
    nlp.to_disk(output_dir)
    return nlp
