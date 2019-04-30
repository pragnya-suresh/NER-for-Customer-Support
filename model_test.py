from __future__ import unicode_literals, print_function
import patterns
import plac
import random
from pathlib import Path
import spacy
import csv
import pandas as pd
from spacy.util import minibatch, compounding


df_test=pd.read_csv('/Users/pragnyasuresh/Desktop/team_no_8/test_ner.csv')
test=df_test["review"]
test2=list(df_test["original_review"])
TEST_DATA=list(test)

@plac.annotations(
                  model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
                  output_dir=("/Users/pragnyasuresh/Desktop/team_no_8", "option", "o", Path),
                  n_iter=("Number of training iterations", "option", "n", int),
                  )
def main(model=None, output_dir="/Users/pragnyasuresh/Desktop/team_no_8", n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""

    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    # test the trained model
    l=[]
    l2=[]

    with open('/Users/pragnyasuresh/Desktop/team_no_8/test.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        header=["Entity","Label","Review","Original_review"]
        writer.writerow(header)
        for text,rev in zip(TEST_DATA,test2):
            
            doc = nlp2(text)
            l3=text
            #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            l1=[]
            l2=[]
            l4=rev
            for ent in doc.ents:
                if(ent.text not in l1):
                    l1.append(ent.text)
                    l2.append(ent.label_)
        
        
            l=[','.join(l1),','.join(l2),l3,l4]
            #print(l)
            writer.writerow(l)


if __name__ == "__main__":
    plac.call(main)

