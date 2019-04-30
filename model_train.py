from __future__ import unicode_literals, print_function
import patterns
import plac
import random
from pathlib import Path
import spacy
import csv
import pandas as pd
from spacy.util import minibatch, compounding

'''
    train_data = [
    ("I bought a top from Roadster", [(11, 14, 'PRODUCT')]),
    '''

df_ent=pd.read_csv("/Users/pragnyasuresh/Desktop/team_no_8/train_ner.csv")
e=df_ent["review"]
e=e.values.tolist()
TRAIN_DATA = []
patterns.A.make_automaton()

for x in e:
    list_ent=[]
    d={}
    for item in patterns.A.iter(x):
        val=((item[0]-len(item[1][1]))+1,item[0]+1,item[1][0])
        list_ent.append(val)
    d["entities"]=list_ent
    TRAIN_DATA.append((x,d)


df_test=pd.read_csv('/Users/pragnyasuresh/Desktop/team_no_8/test_ner.csv')
test=df_test["review"]
TEST_DATA=list(test)


@plac.annotations(
                  model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
                  output_dir=("/Users/pragnyasuresh/Desktop/team_no_8", "option", "o", Path),
                  n_iter=("Number of training iterations", "option", "n", int),
                  )
def main(model=None, output_dir="/Users/pragnyasuresh/Desktop/team_no_8", n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            #print(ent)
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                               texts,  # batch of texts
                               annotations,  # batch of annotations
                               drop=0.5,  # dropout - make it harder to memorise data
                               losses=losses,
                               )
                print("Losses", losses)

  # save model to output directoryf
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
#print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

'''
    it’s not enough to only show a model a single example once. Especially if you only have few examples, you’ll want to train for a number of iterations. At each iteration, the training data is shuffled to ensure the model doesn’t make any generalizations based on the order of examples. Another technique to improve the learning results is to set a dropout rate, a rate at which to randomly “drop” individual features and representations. This makes it harder for the model to memorize the training data. For example, a 0.25 dropout means that each feature or internal representation has a 1/4 likelihood of being dropped.'''


