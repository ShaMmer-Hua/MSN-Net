import csv
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def count_words(text):

    words = re.findall(r"\b[\w']+\b", text)
    return len(words)


def load_ids(csv_path):

    df = pd.read_csv(csv_path)
    return set(df['Participant_ID'].astype(int))

def get_sample_counts(transcript_dir, split_ids):

    sample_counts = {}
    for file in os.listdir(transcript_dir):
        if not file.endswith("_transcript_.csv"):
            continue
        pid_str = file.split('_')[0]
        try:
            pid = int(pid_str)
        except:
            continue
        if pid not in split_ids:
            continue

        file_path = os.path.join(transcript_dir, file)
        qa_pairs = extract_qa_pairs(file_path)
        if not qa_pairs:
            continue
        sample_counts[pid] = len(qa_pairs)

    return sample_counts

def plot_nested_donut_for_paper(train_counts, dev_counts, test_counts):
    train_sizes = list(train_counts.values())
    dev_sizes = list(dev_counts.values())
    test_sizes = list(test_counts.values())

    fig, ax = plt.subplots(figsize=(12,12))
    ax.set(aspect="equal")


    def get_colors(values, cmap_name='viridis'):
        cmap = plt.cm.get_cmap(cmap_name)
        norm = plt.Normalize(min(values), max(values)) if values else plt.Normalize(0,1)
        return [cmap(norm(v)) for v in values]

    train_colors = get_colors(train_sizes, 'Blues')
    dev_colors = get_colors(dev_sizes, 'Greens')
    test_colors = get_colors(test_sizes, 'Oranges')


    wedges_train = ax.pie(
        train_sizes,
        radius=1.0,
        startangle=90,
        colors=train_colors,
        labels=None,
        wedgeprops=dict(width=0.25, edgecolor='white')
    )
    wedges_dev = ax.pie(
        dev_sizes,
        radius=0.75,
        startangle=90,
        colors=dev_colors,
        labels=None,
        wedgeprops=dict(width=0.25, edgecolor='white')
    )
    wedges_test = ax.pie(
        test_sizes,
        radius=0.5,
        startangle=90,
        colors=test_colors,
        labels=None,
        wedgeprops=dict(width=0.2, edgecolor='white')
    )

    def add_labels(wedges, values):
        for w, v in zip(wedges[0], values):
            r_center = w.r - w.width / 2
            theta_center = (w.theta2 + w.theta1) / 2
            x = r_center * np.cos(np.deg2rad(theta_center))
            y = r_center * np.sin(np.deg2rad(theta_center))
            ax.text(x, y, str(v), ha='center', va='center', fontsize=13, weight='bold', color='black')

    add_labels(wedges_train, train_sizes)
    add_labels(wedges_dev, dev_sizes)
    add_labels(wedges_test, test_sizes)


    total_train = sum(train_sizes)
    total_dev = sum(dev_sizes)
    total_test = sum(test_sizes)
    summary_sizes = [total_train, total_dev, total_test]
    summary_labels = ['Train', 'Dev', 'Test']
    summary_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    wedges_summary = ax.pie(
        summary_sizes,
        radius=0.32,
        startangle=90,
        colors=summary_colors,
        labels=[f"{l}\n{v}" for l, v in zip(summary_labels, summary_sizes)],
        labeldistance=0.5,
        wedgeprops=dict(width=0.32, edgecolor='white'),
        textprops=dict(fontsize=14, weight='bold', color='black')
    )

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    transcript_dir = ""
    train_ids = load_ids("")
    dev_ids = load_ids("")
    test_ids = load_ids("")

    train_counts = get_sample_counts(transcript_dir, train_ids)
    dev_counts = get_sample_counts(transcript_dir, dev_ids)
    test_counts = get_sample_counts(transcript_dir, test_ids)

    plot_nested_donut_for_paper(train_counts, dev_counts,test_counts)