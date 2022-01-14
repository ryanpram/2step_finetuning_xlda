import numpy as np
import pandas as pd
import string
import torch
import re
from torch.utils import data
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import json, collections, os, random, glob, math, string, re, torch
from utils.squad_helper import SquadExample, InputFeatures


#####
# Term Extraction Airy
#####
class AspectExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-SENTIMENT': 0, 'O': 1, 'I-ASPECT': 2, 'B-SENTIMENT': 3, 'B-ASPECT': 4}
    INDEX2LABEL = {0: 'I-SENTIMENT', 1: 'O', 2: 'I-ASPECT', 3: 'B-SENTIMENT', 4: 'B-ASPECT'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if '\t' in line:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
       
class AspectExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(AspectExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]
            
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Ner Grit + Prosa
#####
class NerGritDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PERSON': 0, 'B-ORGANISATION': 1, 'I-ORGANISATION': 2, 'B-PLACE': 3, 'I-PLACE': 4, 'O': 5, 'B-PERSON': 6}
    INDEX2LABEL = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O', 6: 'B-PERSON'}
    NUM_LABELS = 7
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data) 

class NerProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'I-PPL': 0, 'B-EVT': 1, 'B-PLC': 2, 'I-IND': 3, 'B-IND': 4, 'B-FNB': 5, 'I-EVT': 6, 'B-PPL': 7, 'I-PLC': 8, 'O': 9, 'I-FNB': 10}
    INDEX2LABEL = {0: 'I-PPL', 1: 'B-EVT', 2: 'B-PLC', 3: 'I-IND', 4: 'B-IND', 5: 'B-FNB', 6: 'I-EVT', 7: 'B-PPL', 8: 'I-PLC', 9: 'O', 10: 'I-FNB'}
    NUM_LABELS = 11
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# Pos Tag Idn + Prosa
#####
class PosTagIdnDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PR': 0, 'B-CD': 1, 'I-PR': 2, 'B-SYM': 3, 'B-JJ': 4, 'B-DT': 5, 'I-UH': 6, 'I-NND': 7, 'B-SC': 8, 'I-WH': 9, 'I-IN': 10, 'I-NNP': 11, 'I-VB': 12, 'B-IN': 13, 'B-NND': 14, 'I-CD': 15, 'I-JJ': 16, 'I-X': 17, 'B-OD': 18, 'B-RP': 19, 'B-RB': 20, 'B-NNP': 21, 'I-RB': 22, 'I-Z': 23, 'B-CC': 24, 'B-NEG': 25, 'B-VB': 26, 'B-NN': 27, 'B-MD': 28, 'B-UH': 29, 'I-NN': 30, 'B-PRP': 31, 'I-SC': 32, 'B-Z': 33, 'I-PRP': 34, 'I-OD': 35, 'I-SYM': 36, 'B-WH': 37, 'B-FW': 38, 'I-CC': 39, 'B-X': 40}
    INDEX2LABEL = {0: 'B-PR', 1: 'B-CD', 2: 'I-PR', 3: 'B-SYM', 4: 'B-JJ', 5: 'B-DT', 6: 'I-UH', 7: 'I-NND', 8: 'B-SC', 9: 'I-WH', 10: 'I-IN', 11: 'I-NNP', 12: 'I-VB', 13: 'B-IN', 14: 'B-NND', 15: 'I-CD', 16: 'I-JJ', 17: 'I-X', 18: 'B-OD', 19: 'B-RP', 20: 'B-RB', 21: 'B-NNP', 22: 'I-RB', 23: 'I-Z', 24: 'B-CC', 25: 'B-NEG', 26: 'B-VB', 27: 'B-NN', 28: 'B-MD', 29: 'B-UH', 30: 'I-NN', 31: 'B-PRP', 32: 'I-SC', 33: 'B-Z', 34: 'I-PRP', 35: 'I-OD', 36: 'I-SYM', 37: 'B-WH', 38: 'B-FW', 39: 'I-CC', 40: 'B-X'}
    NUM_LABELS = 41
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
class PosTagProsaDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'B-PPO': 0, 'B-KUA': 1, 'B-ADV': 2, 'B-PRN': 3, 'B-VBI': 4, 'B-PAR': 5, 'B-VBP': 6, 'B-NNP': 7, 'B-UNS': 8, 'B-VBT': 9, 'B-VBL': 10, 'B-NNO': 11, 'B-ADJ': 12, 'B-PRR': 13, 'B-PRK': 14, 'B-CCN': 15, 'B-$$$': 16, 'B-ADK': 17, 'B-ART': 18, 'B-CSN': 19, 'B-NUM': 20, 'B-SYM': 21, 'B-INT': 22, 'B-NEG': 23, 'B-PRI': 24, 'B-VBE': 25}
    INDEX2LABEL = {0: 'B-PPO', 1: 'B-KUA', 2: 'B-ADV', 3: 'B-PRN', 4: 'B-VBI', 5: 'B-PAR', 6: 'B-VBP', 7: 'B-NNP', 8: 'B-UNS', 9: 'B-VBT', 10: 'B-VBL', 11: 'B-NNO', 12: 'B-ADJ', 13: 'B-PRR', 14: 'B-PRK', 15: 'B-CCN', 16: 'B-$$$', 17: 'B-ADK', 18: 'B-ART', 19: 'B-CSN', 20: 'B-NUM', 21: 'B-SYM', 22: 'B-INT', 23: 'B-NEG', 24: 'B-PRI', 25: 'B-VBE'}
    NUM_LABELS = 26
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)

class PosTagDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(PosTagDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# Emotion Twitter
#####
class EmotionDetectionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
    INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset

    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        tweet, label = self.data.loc[index,'tweet'], self.data.loc[index,'label']        
        subwords = self.tokenizer.encode(tweet, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(label), tweet
    
    def __len__(self):
        return len(self.data)
        
class EmotionDetectionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EmotionDetectionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.full((batch_size, 1), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# Entailment UI
#####
class EntailmentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'NotEntail': 0, 'Entail_or_Paraphrase': 1}
    INDEX2LABEL = {0: 'NotEntail', 1: 'Entail_or_Paraphrase'}
    NUM_LABELS = 2
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        df['label'] = df['label'].apply(lambda label: self.LABEL2INDEX[label])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sent_A, sent_B, label = data['sent_A'], data['sent_B'], data['label']
        
        encoded_inputs = self.tokenizer.encode_plus(sent_A, sent_B, add_special_tokens=not self.no_special_token, return_token_type_ids=True)
        subwords, token_type_ids = encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]
        
        return np.array(subwords), np.array(token_type_ids), np.array(label), data['sent_A'] + "|" + data['sent_B']
    
    def __len__(self):
        return len(self.data)
    
        
class EntailmentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(EntailmentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        label_batch = np.zeros((batch_size, 1), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, token_type_ids, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            label_batch[i,0] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, token_type_batch, label_batch, seq_list

#####
# Document Sentiment Prosa
#####
class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def load_dataset(self, path): 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['text','sentiment']
        df['sentiment'] = df['sentiment'].apply(lambda lab: self.LABEL2INDEX[lab])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, sentiment = data['text'], data['sentiment']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class DocumentSentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            sentiment_batch[i,0] = sentiment
            
            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, sentiment_batch, seq_list

#####
# Keyword Extraction Prosa
#####
class KeywordExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
        
class KeywordExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(KeywordExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []

        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
#####
# QA Factoid ITB
#####
class QAFactoidDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O':0, 'B':1, 'I':2}
    INDEX2LABEL = {0:'O', 1:'B', 2:'I'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Read file
        dataset = pd.read_csv(path)
        
        # Question and passage are a list of words and seq_label is list of B/I/O
        dataset['question'] = dataset['question'].apply(lambda x: eval(x))
        dataset['passage'] = dataset['passage'].apply(lambda x: eval(x))
        dataset['seq_label'] = dataset['seq_label'].apply(lambda x: [self.LABEL2INDEX[l] for l in eval(x)])

        return dataset
    
    def filter_noise_token(self, passage_tokens):
        noise_tokens = [6]
        if (passage_tokens in noise_tokens):
            return False
        else:
            return True
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        chinese_mode = False
        if self.dataset_path.split("/")[2] == 'crmc':
            chinese_mode = True

        data = self.data.loc[index,:]
        question, passage, seq_label = data['question'],  data['passage'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        token_type_ids = [0]
        
        # Add subwords for question
        for word_idx, word in enumerate(question):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            if chinese_mode:
                subword_list = list(filter(self.filter_noise_token, subword_list))
            subword_to_word_indices += [-1 for i in range(len(subword_list))]
            token_type_ids += [0 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add intermediate SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [0]
        
        # Add subwords
        for word_idx, word in enumerate(passage):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            if chinese_mode:
                subword_list = list(filter(self.filter_noise_token, subword_list))
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            token_type_ids += [1 for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        token_type_ids += [1]
        
        return np.array(subwords), np.array(token_type_ids), np.array(subword_to_word_indices), np.array(seq_label), ' '.join(question) + "|" + ' '.join(passage)
    
    def __len__(self):
        return len(self.data)
        
class QAFactoidDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(QAFactoidDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[3]), batch))
        #keep label length not exceeding max_seq_len length
        max_tgt_len = min(self.max_seq_len, max_tgt_len)

        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        token_type_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        max_label_len = 0
        for i, (subwords, token_type_ids, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]
            #truncating token_type_ids 
            token_type_ids = token_type_ids[:max_seq_len]


            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            token_type_batch[i,:len(subwords)] = token_type_ids
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            #get last passage word indices after truncation
            last_passage_word_indices = 0
            
            for x in subword_to_word_indices[::-1]:
                if x != -1 :
                    last_passage_word_indices = x
                    #save max label len in batch
                    if max_label_len < last_passage_word_indices + 1:
                        max_label_len = last_passage_word_indices + 1

                    break
                
            # seq_label_batch[i,:len(seq_label)] = seq_label

            # add seq label following passage word token length
            seq_label_batch[i,:last_passage_word_indices+1] = seq_label[:last_passage_word_indices+1]

            seq_list.append(raw_seq)
        
        #truncate sequence label batch based on max_label_len in batch
        seq_label_batch = np.delete(seq_label_batch,  np.s_[max_label_len:max_tgt_len], 1)
        #print('max_tgt_len: ',seq_label_batch)
        #print('last_passage_word_indices: ',last_passage_word_indices)
        #print('seq label batch: ',seq_label_batch)

            
        return subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

#####
# ABSA Airy + Prosa
#####
class AspectBasedSentimentAnalysisAiryDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['ac', 'air_panas', 'bau', 'general', 'kebersihan', 'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    LABEL2INDEX = {'neg': 0, 'neut': 1, 'pos': 2, 'neg_pos': 3}
    INDEX2LABEL = {0: 'neg', 1: 'neut', 2: 'pos', 3: 'neg_pos'}
    NUM_LABELS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    NUM_ASPECTS = 10
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['review'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['review']
    
    def __len__(self):
        return len(self.data)
    
class AspectBasedSentimentAnalysisProsaDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['fuel', 'machine', 'others', 'part', 'price', 'service']
    LABEL2INDEX = {'negative': 0, 'neutral': 1, 'positive': 2}
    INDEX2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
    NUM_LABELS = [3, 3, 3, 3, 3, 3]
    NUM_ASPECTS = 6
    
    def load_dataset(self, path):
        df = pd.read_csv(path)
        for aspect in self.ASPECT_DOMAIN:
            df[aspect] = df[aspect].apply(lambda sen: self.LABEL2INDEX[sen])
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        sentence, labels = data['sentence'], [data[aspect] for aspect in self.ASPECT_DOMAIN]
        subwords = self.tokenizer.encode(sentence, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['sentence']
    
    def __len__(self):
        return len(self.data)
    
        
class AspectBasedSentimentAnalysisDataLoader(DataLoader):
    def __init__(self, dataset, max_seq_len=512, *args, **kwargs):
        super(AspectBasedSentimentAnalysisDataLoader, self).__init__(dataset=dataset, *args, **kwargs)
        self.num_aspects = dataset.NUM_ASPECTS
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        label_batch = np.zeros((batch_size, self.num_aspects), dtype=np.int64)

        seq_list = []
        
        for i, (subwords, label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            label_batch[i,:] = label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, label_batch, seq_list

#####
# News Categorization Prosa
#####
class NewsCategorizationDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'permasalahan pada bank besar domestik': 0, 'pertumbuhan ekonomi domestik yang terbatas': 1, 'volatilitas harga komoditas utama dunia': 2, 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi': 3, 'perubahan kebijakan dan/atau regulasi pada institusi keuangan': 4, 'isu politik domestik': 5, 'permasalahan pada bank besar international': 6, 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal': 7, 'pertumbuhan ekonomi global yang terbatas': 8, 'kebijakan pemerintah yang bersifat sektoral': 9, 'isu politik dan ekonomi luar negeri': 10, 'kenaikan harga volatile food': 11, 'tidak berisiko': 12, 'pergerakan harga minyak mentah dunia': 13, 'force majeure yang memengaruhi operasional sistem keuangan': 14, 'kenaikan administered price': 15}
    INDEX2LABEL = {0: 'permasalahan pada bank besar domestik', 1: 'pertumbuhan ekonomi domestik yang terbatas', 2: 'volatilitas harga komoditas utama dunia', 3: 'frekuensi kenaikan fed fund rate (ffr) yang melebihi ekspektasi', 4: 'perubahan kebijakan dan/atau regulasi pada institusi keuangan', 5: 'isu politik domestik', 6: 'permasalahan pada bank besar international', 7: 'perubahan kebijakan pemerintah yang berkaitan dengan fiskal', 8: 'pertumbuhan ekonomi global yang terbatas', 9: 'kebijakan pemerintah yang bersifat sektoral', 10: 'isu politik dan ekonomi luar negeri', 11: 'kenaikan harga volatile food', 12: 'tidak berisiko', 13: 'pergerakan harga minyak mentah dunia', 14: 'force majeure yang memengaruhi operasional sistem keuangan', 15: 'kenaikan administered price'}
    NUM_LABELS = 16
    
    def load_dataset(self, path):
        dataset = pd.read_csv(path, sep='\t', header=None)
        dataset.columns = ['text', 'label']
        dataset['label'] = dataset['label'].apply(lambda labels: [self.LABEL2INDEX[label] for label in labels.split(',')])
        return dataset
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, labels = data['text'], data['label']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['text']
    
    def __len__(self):
        return len(self.data)    
        
class NewsCategorizationDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NewsCategorizationDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        # Trimmed input based on specified max_len
        if self.max_seq_len < max_seq_len:
            max_seq_len = self.max_seq_len
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        labels_batch = np.zeros((batch_size, NUM_LABELS), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, labels) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            for label in labels:
                labels_batch[i,label] = 1

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, labels_batch, seq_list


#####
# Squad Like Dataset
#####
class SquadLikeDataset(Dataset):
    # Static constant variable

    def _improve_answer_span(doc_tokens, 
                         input_start, 
                         input_end, 
                         tokenizer,
                         orig_answer_text):
        """
        Returns tokenized answer spans that better match the annotated answer.
        """
        
        # print(doc_tokens)
        # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']

        # print(input_start)
        # 66
        
        # print(input_end) 
        # 69
        
        # print(tokenizer)
        # <transformers.tokenization_bert.BertTokenizer object at 0x000001C1CE9D4F98>
        
        # print(orig_answer_text)
        # in the late 1990s
        
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
        # print(tok_answer_text)
        # in the late 1990s

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                # print(text_span)
                # in the late 1990s
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(doc_spans, cur_span_index, position):
        """
        Check if this is the "max context" doc span for the token.
        """
        
        best_score = None
        best_span_index = None
        
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
    
    def read_squad_examples(input_file, is_training):
        """Read a SQuAD json file into a list of SquadExample."""

        with open(input_file, "r", encoding='utf-8') as reader:
            source = json.load(reader)
            input_data = source["data"]
            version = source["version"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    answers = []
                    if is_training:
                        if version == "v2.0":
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            #print(entry["title"], qas_id)
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                print("Could not find answer: '%s' vs. '%s'",
                                                actual_text, cleaned_answer_text)
                                continue
                        else: 
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    else:
                        answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        answers=answers)
                    examples.append(example)
        return examples

    def convert_examples_to_features(self,
                                 examples, 
                                 tokenizer, 
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token="<s>", 
                                 sep_token="</s>", 
                                 pad_token=1,
                                 sequence_a_segment_id=1, 
                                 sequence_b_segment_id=3,
                                 cls_token_segment_id=1, 
                                 pad_token_segment_id=1,
                                 mask_padding_with_zero=True):
        """
        Loads a data file into a list of `InputBatch`s.
        """
        unique_id = 1000000000
        
        features = []
        for (example_index, example) in enumerate(examples):
            # print(example_index)
            # 0
            
            # print(example)
            # qas_id: 56be85543aeaaa14008c9063, 
            # question_text: When did Beyonce start becoming popular?,
            # doc_tokens: [Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".], 
            # start_position: 39, 
            # end_position: 42
            
            query_tokens = tokenizer.tokenize(example.question_text)
            # print(query_tokens)
            # ['when', 'did', 'beyonce', 'start', 'becoming', 'popular', '?']
        
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]
                
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            
            # `token`s are separated by whitespace; `sub_token`s are separated in a `token` by symbol
            for (i, token) in enumerate(example.doc_tokens):
                # print(i)
                # 0
                
                # print(token)
                # Beyoncé
                
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                # print(sub_tokens)
                # ['beyonce']
                # ['gi', '##selle']
                # ['knowles', '-', 'carter']
                # ['(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/']
                # ...
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            
            # print(tok_to_orig_index)
            # [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 51, 52, 53, 54, 54, 55, 56, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 64, 64, 64, 65, 66, 67, 68, 69, 69, 70, 71, 72, 73, 74, 75, 76, 76, 76, 77, 78, 78, 79, 80, 81, 82, 82, 82, 82, 83, 84, 85, 86, 87, 88, 89, 90, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 108, 108]
            
            # print(orig_to_tok_index)
            # [0, 1, 3, 6, 16, 23, 25, 26, 28, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 80, 83, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 102, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123, 124, 125, 126, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 153, 155, 156, 158, 159, 161]
            
            # print(all_doc_tokens)
            # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']
            
            tok_start_position = None
            tok_end_position = None
            
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                # print(tok_start_position)
                # 66
                
                # print(tok_end_position)
                # 69
                (tok_start_position, tok_end_position) = self._improve_answer_span(all_doc_tokens, 
                                                                            tok_start_position, 
                                                                            tok_end_position, 
                                                                            tokenizer, 
                                                                            example.orig_answer_text)
                # print(tok_start_position)
                # 66
                
                # print(tok_end_position)
                # 69
                
            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
            
            # We can have documents that are longer than the maximum sequence length. To deal with this we do a 
            # sliding window approach, where we take chunks of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            
            while start_offset < len(all_doc_tokens):
                # print(len(all_doc_tokens))
                # 426
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                # Take an example with stride
                # 
                # print(doc_spans)
                # [DocSpan(start=0, length=373)]
                # 
                # In this case, `start` will move a `doc_strike`, 128, so the new `start` is 128  
                # And the new `length` is 426 - 128 = 298
                # 
                # [DocSpan(start=0, length=373), DocSpan(start=128, length=298)]
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                # `p_mask`: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keeps the classification token (set to 0) (not sure why...)
                p_mask = []

                # `[CLS]` token at the beginning
                if not cls_token_at_end:
                    tokens.append(cls_token)
                    segment_ids.append(cls_token_segment_id)
                    p_mask.append(0)
                    cls_index = 0
                    
                # Query
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(sequence_a_segment_id)
                    p_mask.append(1)

                # [SEP] token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                # Paragraph built based on `doc_span`
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                paragraph_len = doc_span.length

                # [SEP] token
                tokens.append(sep_token)
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(1)

                # [CLS] token at the end
                if cls_token_at_end:
                    tokens.append(cls_token)
                    segment_ids.append(cls_token_segment_id)
                    p_mask.append(0)
                    # Index of classification token
                    cls_index = len(tokens) - 1

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(pad_token)
                    input_mask.append(0 if mask_padding_with_zero else 1)
                    segment_ids.append(pad_token_segment_id)
                    p_mask.append(1)

                # print(input_ids)
                # [101, 2043, 2106, 20773, 2707, 3352, 2759, 1029, 102, 20773, 21025, 19358, 22815, 1011, 5708, 1006, 1013, 12170, 23432, 29715, 3501, 29678, 12325, 29685, 1013, 10506, 1011, 10930, 2078, 1011, 2360, 1007, 1006, 2141, 2244, 1018, 1010, 3261, 1007, 2003, 2019, 2137, 3220, 1010, 6009, 1010, 2501, 3135, 1998, 3883, 1012, 2141, 1998, 2992, 1999, 5395, 1010, 3146, 1010, 2016, 2864, 1999, 2536, 4823, 1998, 5613, 6479, 2004, 1037, 2775, 1010, 1998, 3123, 2000, 4476, 1999, 1996, 2397, 4134, 2004, 2599, 3220, 1997, 1054, 1004, 1038, 2611, 1011, 2177, 10461, 1005, 1055, 2775, 1012, 3266, 2011, 2014, 2269, 1010, 25436, 22815, 1010, 1996, 2177, 2150, 2028, 1997, 1996, 2088, 1005, 1055, 2190, 1011, 4855, 2611, 2967, 1997, 2035, 2051, 1012, 2037, 14221, 2387, 1996, 2713, 1997, 20773, 1005, 1055, 2834, 2201, 1010, 20754, 1999, 2293, 1006, 2494, 1007, 1010, 2029, 2511, 2014, 2004, 1037, 3948, 3063, 4969, 1010, 3687, 2274, 8922, 2982, 1998, 2956, 1996, 4908, 2980, 2531, 2193, 1011, 2028, 3895, 1000, 4689, 1999, 2293, 1000, 1998, 1000, 3336, 2879, 1000, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                # print(input_mask)
                # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                # print(segment_ids)
                # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # Only `sequence_b_segment_id` is set to 1
                
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                span_is_impossible = example.is_impossible
                start_position = None
                end_position = None
                
                # Get `start_position` and `end_position`
                if is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation we throw it out, 
                    # since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        span_is_impossible = True
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
            
                if is_training and span_is_impossible:
                    start_position = cls_index
                    end_position = cls_index
            
                # Display some examples
                if example_index < 20:              
                    print("*** Example ***")
                    # *** Example ***
                    print("unique_id: %s" % (unique_id))
                    # unique_id: 1000000000
                    print("example_index: %s" % (example_index))
                    # example_index: 0
                    print("doc_span_index: %s" % (doc_span_index))
                    # doc_span_index: 0
                    print("tokens: %s" % " ".join(tokens))
                    # tokens: [CLS] when did beyonce start becoming popular ? [SEP] beyonce gi ##selle knowles - carter ( / bi ##ː ##ˈ ##j ##ɒ ##nse ##ɪ / bee - yo ##n - say ) ( born september 4 , 1981 ) is an american singer , songwriter , record producer and actress . born and raised in houston , texas , she performed in various singing and dancing competitions as a child , and rose to fame in the late 1990s as lead singer of r & b girl - group destiny ' s child . managed by her father , mathew knowles , the group became one of the world ' s best - selling girl groups of all time . their hiatus saw the release of beyonce ' s debut album , dangerously in love ( 2003 ) , which established her as a solo artist worldwide , earned five grammy awards and featured the billboard hot 100 number - one singles " crazy in love " and " baby boy " . [SEP]
                    print("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    # token_to_orig_map: 9:0 10:1 11:1 12:2 13:2 14:2 15:3 16:3 17:3 18:3 19:3 20:3 21:3 22:3 23:3 24:3 25:4 26:4 27:4 28:4 29:4 30:4 31:4 32:5 33:5 34:6 35:7 36:7 37:8 38:8 39:9 40:10 41:11 42:12 43:12 44:13 45:13 46:14 47:15 48:16 49:17 50:17 51:18 52:19 53:20 54:21 55:22 56:22 57:23 58:23 59:24 60:25 61:26 62:27 63:28 64:29 65:30 66:31 67:32 68:33 69:34 70:34 71:35 72:36 73:37 74:38 75:39 76:40 77:41 78:42 79:43 80:44 81:45 82:46 83:47 84:47 85:47 86:48 87:48 88:48 89:49 90:49 91:49 92:50 93:50 94:51 95:52 96:53 97:54 98:54 99:55 100:56 101:56 102:57 103:58 104:59 105:60 106:61 107:62 108:63 109:63 110:63 111:64 112:64 113:64 114:65 115:66 116:67 117:68 118:69 119:69 120:70 121:71 122:72 123:73 124:74 125:75 126:76 127:76 128:76 129:77 130:78 131:78 132:79 133:80 134:81 135:82 136:82 137:82 138:82 139:83 140:84 141:85 142:86 143:87 144:88 145:89 146:90 147:90 148:91 149:92 150:93 151:94 152:95 153:96 154:97 155:98 156:99 157:100 158:101 159:101 160:101 161:102 162:103 163:103 164:104 165:105 166:105 167:106 168:107 169:107 170:108 171:108 172:108
                    print("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    # token_is_max_context: 9:True 10:True 11:True 12:True 13:True 14:True 15:True 16:True 17:True 18:True 19:True 20:True 21:True 22:True 23:True 24:True 25:True 26:True 27:True 28:True 29:True 30:True 31:True 32:True 33:True 34:True 35:True 36:True 37:True 38:True 39:True 40:True 41:True 42:True 43:True 44:True 45:True 46:True 47:True 48:True 49:True 50:True 51:True 52:True 53:True 54:True 55:True 56:True 57:True 58:True 59:True 60:True 61:True 62:True 63:True 64:True 65:True 66:True 67:True 68:True 69:True 70:True 71:True 72:True 73:True 74:True 75:True 76:True 77:True 78:True 79:True 80:True 81:True 82:True 83:True 84:True 85:True 86:True 87:True 88:True 89:True 90:True 91:True 92:True 93:True 94:True 95:True 96:True 97:True 98:True 99:True 100:True 101:True 102:True 103:True 104:True 105:True 106:True 107:True 108:True 109:True 110:True 111:True 112:True 113:True 114:True 115:True 116:True 117:True 118:True 119:True 120:True 121:True 122:True 123:True 124:True 125:True 126:True 127:True 128:True 129:True 130:True 131:True 132:True 133:True 134:True 135:True 136:True 137:True 138:True 139:True 140:True 141:True 142:True 143:True 144:True 145:True 146:True 147:True 148:True 149:True 150:True 151:True 152:True 153:True 154:True 155:True 156:True 157:True 158:True 159:True 160:True 161:True 162:True 163:True 164:True 165:True 166:True 167:True 168:True 169:True 170:True 171:True 172:True
                    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    # input_ids: 101 2043 2106 20773 2707 3352 2759 1029 102 20773 21025 19358 22815 1011 5708 1006 1013 12170 23432 29715 3501 29678 12325 29685 1013 10506 1011 10930 2078 1011 2360 1007 1006 2141 2244 1018 1010 3261 1007 2003 2019 2137 3220 1010 6009 1010 2501 3135 1998 3883 1012 2141 1998 2992 1999 5395 1010 3146 1010 2016 2864 1999 2536 4823 1998 5613 6479 2004 1037 2775 1010 1998 3123 2000 4476 1999 1996 2397 4134 2004 2599 3220 1997 1054 1004 1038 2611 1011 2177 10461 1005 1055 2775 1012 3266 2011 2014 2269 1010 25436 22815 1010 1996 2177 2150 2028 1997 1996 2088 1005 1055 2190 1011 4855 2611 2967 1997 2035 2051 1012 2037 14221 2387 1996 2713 1997 20773 1005 1055 2834 2201 1010 20754 1999 2293 1006 2494 1007 1010 2029 2511 2014 2004 1037 3948 3063 4969 1010 3687 2274 8922 2982 1998 2956 1996 4908 2980 2531 2193 1011 2028 3895 1000 4689 1999 2293 1000 1998 1000 3336 2879 1000 1012 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    # input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    # segment_ids: 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                    if is_training and span_is_impossible:
                        print("impossible example")
                    if is_training and not span_is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        print("start_position: %d" % (start_position))
                        # start_position: 75
                        print("end_position: %d" % (end_position))
                        # end_position: 78
                        print("answer: %s" % (answer_text))
                        # answer: in the late 1990s

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        cls_index=cls_index,
                        p_mask=p_mask,
                        paragraph_len=paragraph_len,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=span_is_impossible)
                )
                
                unique_id += 1

        return features
    
    def load_and_cache_examples(self,
                            tokenizer, 
                            evaluate=False, 
                            output_examples=False,
                            dataset_input='dataset/ViQuAD/train_ViQuAD.json'):
    
    
        input_file = dataset_input
            
        # Call `read_squad_examples()`
        examples = self.read_squad_examples(input_file=input_file,
                                        is_training=not evaluate)

        # Call `convert_examples_to_features()`
        features = self.convert_examples_to_features(examples, 
                                                tokenizer, 
                                                max_seq_length=256,
                                                doc_stride=128,
                                                max_query_length=64,
                                                is_training=not evaluate,
                                                cls_token_at_end=False,
                                                cls_token="<s>", 
                                                sep_token="</s>", 
                                                pad_token=1,
                                                sequence_a_segment_id=1, # 'pad_token': '[PAD]'
                                                sequence_b_segment_id=3,
                                                cls_token_segment_id=1, 
                                                pad_token_segment_id=1,
                                                mask_padding_with_zero=True)


        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        
        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, 
                                    all_input_mask, 
                                    all_segment_ids,
                                    all_example_index, 
                                    all_cls_index, 
                                    all_p_mask)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, 
                                    all_input_mask, 
                                    all_segment_ids,
                                    all_start_positions, 
                                    all_end_positions,
                                    all_cls_index, 
                                    all_p_mask)

        if output_examples:
            return dataset, examples, features

        return dataset
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_and_cache_examples('', 
                                                tokenizer, 
                                                evaluate=False, 
                                                output_examples=False,
                                                dataset_input=dataset_path
                                            )
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, labels = data['text'], data['label']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(labels), data['text']
    
    def __len__(self):
        return len(self.data)

        