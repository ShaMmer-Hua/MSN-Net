import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import csv
import pandas as pd
import hashlib
import os


def get_cache_path(csv_path):
    with open(csv_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"./orgDatas/noEmoCache/{os.path.basename(csv_path)}_{file_hash}.pt"



def save_encoded_data(data, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def move_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        else:
            return obj

    cpu_data = move_to_cpu(data)
    torch.save(cpu_data, cache_path)


def load_encoded_data(cache_path, device):
    # data = torch.load(cache_path, map_location=device)
    data = torch.load(cache_path)
    return data


def merge_segments(segments):
    if not segments:
        return {'text': '', 'start': 0, 'end': 0, 'segments': []}

    return {
        'text': ' '.join(s['text'] for s in segments),
        'start': segments[0]['start'],
        'end': segments[-1]['end'],
        'segments': segments
    }


def extract_qa_pairs(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    qa_pairs = []
    current_q = None
    current_a = []

    for row in rows:
        speaker = row['speaker'].strip()
        value = row['value'].strip()
        start = float(row['start_time'])
        end = float(row['stop_time'])

        if speaker == 'Ellie':
            if current_q and current_a:
                qa_pairs.append({
                    'question': current_q,
                    'answer': merge_segments(current_a),
                    'start': current_q['start'],
                    'end': merge_segments(current_a)['end']
                })
            current_q = {'text': value, 'start': start, 'end': end}
            current_a = []
        elif speaker == 'Participant' and current_q:
            current_a.append({'text': value, 'start': start, 'end': end})

    if current_q and current_a:
        qa_pairs.append({
            'question': current_q,
            'answer': merge_segments(current_a),
            'start': current_q['start'],
            'end': merge_segments(current_a)['end']
        })

    return qa_pairs


class QADataset(Dataset):
    def __init__(self, question, csv_path, tokenizer, model, max_length=None,
                 has_labels=True, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.has_labels = has_labels
        self.device = device

        cache_path = get_cache_path(csv_path)
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            cached_data = load_encoded_data(cache_path, self.device)

            self.sequence_outputs = cached_data['sequence']
            self.sequence_lengths = cached_data['sequence_lengths']

            self.round_outputs = cached_data.get('multi_rounds', {})

        else:

            self.qa_pairs = extract_qa_pairs(csv_path)

            self.qa_sentences = []
            self.qa_sentences.extend(question)
            for pair in self.qa_pairs:
                q_text = pair['question']['text']
                a_text = pair['answer']['text']
                qa_start = pair['start']
                qa_end = pair['end']

                a_start = pair['answer']['start']
                a_end = pair['answer']['end']

                self.qa_sentences.append(f"{q_text}. {a_text}")

            encoded_inputs = tokenizer(self.qa_sentences, padding=True, truncation=True, return_tensors="pt")

            encoded_inputs = encoded_inputs.to(model.device)

            self.input_ids = encoded_inputs['input_ids']
            self.attention_mask = encoded_inputs['attention_mask']
            self.token_type_ids = encoded_inputs['token_type_ids']

            with torch.no_grad():
                output = model(**encoded_inputs)


            rounds_per_entry_list = [2]

            qa_sentences_by_rounds = {}

            for rounds_per_entry in rounds_per_entry_list:
                qa_sentences = []
                qa_sentences.extend(question)
                temp_dialogue = []

                for idx, pair in enumerate(self.qa_pairs):
                    q_text = pair['question']['text']
                    a_text = pair['answer']['text']

                    temp_dialogue.append(f"{q_text}. {a_text}")

                    if (idx + 1) % rounds_per_entry == 0 or (idx + 1) == len(self.qa_pairs):
                        combined_dialogue = " ".join(temp_dialogue)
                        qa_sentences.append(combined_dialogue)
                        temp_dialogue = []

                qa_sentences_by_rounds[rounds_per_entry] = qa_sentences


            full_cache_data = {
                'sequence': output[0],
                'sequence_lengths': self.attention_mask,
                'multi_rounds': {}
            }


            self.round_outputs = {}

            for rounds, sentences in qa_sentences_by_rounds.items():
                print(f"Encoding {rounds}-round dialogues...")

                encoded = tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    round_output = model(**encoded)

                round_data = {
                    "sequence": round_output[0],
                    "pooled": round_output[1],
                    "attention_mask": encoded['attention_mask']
                }
                self.round_outputs[rounds] = round_data
                full_cache_data['multi_rounds'][rounds] = round_data

            save_encoded_data(full_cache_data, cache_path)

            self.sequence_outputs = output[0]
            self.pooled_output = output[1]
            self.sequence_lengths = self.attention_mask


            self.labels = [1] * len(self.qa_sentences)

    def __len__(self):
        return len(self.qa_sentences)

    def __getitem__(self, idx):
        sentence = self.sequence_outputs[idx]

        return sentence

    def getALL(self):
        # , self.attention_mask
        return self.sequence_outputs

    def getLengths(self):
        return self.sequence_lengths

    def getMfcc(self):
        return self.mfcc_features

    def getMfccLengths(self):
        return self.mfcc_lengths

    def getQuestions(self):
        return self.questions

    def getQuestionsLengths(self):
        return self.question_mask

    def getRound_outputs(self):
        return self.round_outputs
