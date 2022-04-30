import spacy

nlp = spacy.load("en_core_web_sm")

with open("data/wiki_nba.txt", "r") as f:
  text = f.read()

# create a doc object
doc = nlp(text)
print(len(text), len(doc)) #26881, 5158 

for token in doc[:10]:
  print(token)

# Sentence Boundary Detection: need to convert generator to list to access
sentence1 = list(doc.sents)[0]
print(sentence1)