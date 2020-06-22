import re 
from contraction import contraction_map

def clean_cnn(sentence):
    # Strip (CNN) -- and (CNN)  -- if it exists then strip
    headers = ['(CNN) -- ', '(CNN)  --']
    for header in headers:
        ind = sentence.find(header)
        if ind > -1:
            sentence = sentence[ind+len(header):]
    return sentence.strip()

def clean_author(sentence):
    # eg. Strip By . Ellie Zolfagharifard . if it exist then strip
    author = re.search("By . [a-zA-Z\s]+", sentence)
    if author != None:
        ind = author.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_publish(sentence):
    # eg. Strip PUBLISHED: . 03:39 EST, 8 May 2012 . if it exist then strip
    publish = re.search("PUBLISHED: . \d\d:\d\d\s[A-Z]{3},\s\d{1,2}\s[a-zA-Z]+\s\d{4}", sentence)
    if publish != None:
        ind = publish.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_update(sentence):
    # eg. Strip UPDATED: . 18:10 EST, 8 May 2012 . if it exist then strip
    update = re.search("UPDATED: . \d\d:\d\d\s[A-Z]{3},\s\d{1,2}\s[a-zA-Z]+\s\d{4}", sentence)
    if update != None:
        ind = update.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_punct(sentence):
    # Remove \n \t
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('\t', '')
    # Remove |
    sentence = sentence.replace('|', '')
    # Remove \
    sentence = sentence.replace('\\', '')
    # Remove . if it is the first word
    ind = sentence.find('.')
    if ind == 0:
        sentence = sentence[1:]
    sentence = sentence.replace(',', ' , ') # except between 2 chars
    sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace('!', ' ! ')
    sentence = sentence.replace('?', ' ? ')
    sentence = sentence.replace('\'', ' \' ')
    sentence = sentence.replace('"', ' " ')
    return sentence.strip()

def clean_bracket(sentence):
    sentence = re.sub("\([(\d|\D)]+\)", '', sentence)
    return sentence
    
def clean_src(sentence):
    sentence = sentence.decode("utf-8")
    sentence = contraction_map(sentence)
    sentence = clean_cnn(sentence)
    sentence = clean_author(sentence)
    sentence = clean_publish(sentence)
    sentence = clean_update(sentence)
    sentence = clean_bracket(sentence)
    sentence = clean_punct(sentence)
    return sentence.lower()

def clean_trg(sentence):
    sentence = sentence.decode("utf-8")
    sentence = contraction_map(sentence)
    sentence = clean_bracket(sentence)
    sentence = clean_punct(sentence)
    return sentence.lower()

# to do:
# line 49
# delete single alphabet
# delete - 
