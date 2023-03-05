import numpy as np
from stopwordsiso import stopwords
from copy import deepcopy
from tqdm.autonotebook import tqdm
import string
import re

# Replace uninformative/duplicate titles/descriptions with nothing.
# A bit too nuanced to reliably implement. 

# Remove stopwords for all languages (can't do all languages, will tackle the top 3: english, spanish, french).
def remove_stopwords(x, stop_words, is_context=False):
    if x is not np.nan:
        if not is_context:
            x = x.split()
            x = [w for w in x if not w.lower() in stop_words]
            x = " ".join(x)
        else:
            x = x.split("[SEP]")
            x = [w for w in x if not w.lower() in stop_words]
            x = "[SEP]".join(x)
    return x

# Handle "source_id=" in topic descriptions.
def replace_sourceid(x, is_context=False):
    if x is not np.nan:
        if not is_context:
            if "source_id" in x:
                x = np.nan
        else:
            if "source_id" in x:
                x = x.split("[SEP]")
                x = "[SEP]".join(np.array(x)[list(map(lambda s: "source_id" not in s, x))].tolist())
                if x == "":
                    x = np.nan
    return x

# Handle literal gibberish: '5ad59f7a6b9064043e263f03' and weird links (YouTube link).
# Tried:
# - https://github.com/domanchi/gibberish-detector
# Too finnicky and rare (and risky) to use.

# Handle/remove Topic, Section, Chapter text.
def remove_topic(x):
    words = ["topic", "Topic", "topics", "Topics", "chapter", "Chapter", "chapters", "Chapters",
             "section", "Section", "sections", "Sections"]

    if x is not np.nan:
        for word in words:
            x = x.replace(word, "")
        x = x.strip()
    return x

# Remove numbers.
def remove_numbers(x):
    if x is np.nan:
        return x
    return re.sub("\d+", "", x)

# Remove special characters.
def remove_chars(x, include_brackets=False):
    if x is np.nan:
        return x
    if include_brackets:
        return x.translate(str.maketrans('', '', string.punctuation.replace("[", "").replace("]", "")))
    return x.translate(str.maketrans('', '', string.punctuation))

# Remove excess whitespace, \n, and setting strings of length 1 to np.nan.
def remove_space_make_nans(x):
    if x is not np.nan:
        if len(x) > 1:
            x = x.replace("\n", "")
            x = re.sub(' +', ' ',  x.strip())
        else:
            x = np.nan
    return x

def clean(train_context):
    langs = ["en", "es", "fr"]
    
    curr_text_cols = [
         "topic_title", 
         "topic_description", 
         "content_title", 
         "content_description", 
         "content_text"
    ]
    
    context_text_cols = [
        "topic_parent_title", 
        "topic_parent_description", 
        "topic_child_title", 
        "topic_child_description"
    ]
    
    print("Removing stopwords...", end="")
    for lang in langs:
        stop_words = stopwords(lang)
        
        for col in tqdm(curr_text_cols, leave=True, position=0, total=len(curr_text_cols)):
            train_context[col] = train_context[col].apply(remove_stopwords, stop_words=stop_words)

        for col in tqdm(context_text_cols, leave=True, position=0, total=len(context_text_cols)):
            train_context[col] = train_context[col].apply(remove_stopwords, stop_words=stop_words, is_context=True)
    print("Finished")
    
    print("Removing source_id descriptions...", end="")
    for col in tqdm(curr_text_cols, leave=True, position=0, total=len(curr_text_cols)):
        if "description" in col:
            train_context[col] = train_context[col].apply(replace_sourceid)

    for col in tqdm(context_text_cols, leave=True, position=0, total=len(context_text_cols)):
        if "description" in col:
            train_context[col] = train_context[col].apply(replace_sourceid, is_context=True)
    print("Finished")
    
    print("Removing topic words...", end="")
    for col in tqdm(curr_text_cols, leave=True, position=0, total=len(curr_text_cols)):
        train_context[col] = train_context[col].apply(remove_topic)

    for col in tqdm(context_text_cols, leave=True, position=0, total=len(context_text_cols)):
        train_context[col] = train_context[col].apply(remove_topic)
    print("Finished")
    
    print("Removing special characters & numbers...", end="")
    for col in tqdm(curr_text_cols, position=0, leave=True, total=len(curr_text_cols)):
        train_context[col] = train_context[col].apply(remove_chars)
        train_context[col] = train_context[col].apply(remove_numbers)
        
    for col in tqdm(context_text_cols, position=0, leave=True, total=len(context_text_cols)):
        train_context[col] = train_context[col].apply(remove_chars, include_brackets=True)
        train_context[col] = train_context[col].apply(remove_numbers)
    print("Finished")
    
    print("Removing extra whitespace...", end="")
    all_text_cols = curr_text_cols + context_text_cols
    for col in tqdm(all_text_cols, leave=True, position=0, total=len(all_text_cols)):
        train_context[col] = train_context[col].apply(remove_space_make_nans)
    print("Finished")
        
    return train_context