import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from modelAT_multiscale import BERTLSTMATTETION
from datasetAT import QADataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import random
import hashlib


def set_seed(seed: int, deterministic: bool = True):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_clinical_prompts():
    return [
        'Have you felt emotionally and physically well lately ?',
        'Have you noticed significant changes in your mood, such as feeling persistently sad, empty, or hopeless ?',
        'Have you experienced difficulties with your sleep, such as trouble falling asleep, staying asleep, or waking up too early ?',
        'Are you finding it challenging to concentrate on tasks or make decisions ?',
        'Have you lost interest or pleasure in activities you used to enjoy ?',
        'Have you ever been diagnosed with depression or experienced pro-longed periods of feeling down or hopeless in the past ?',
        'Have you ever been diagnosed with PTSD (Post-Traumatic Stress Disorder) ?',
        'Have you been experiencing any financial problems recently ?',
        'Do you find it challenging to socialize and prefer solitary activities, indicating introverted tendencies ?',
        'Have you had thoughts of death or suicide, or have you made any suicide attempts ?',
        'Have you ever served in the military ?'
    ]


def get_generic_prompts():
    return [
        "What is the main topic discussed in the conversation?",
        "What key events are mentioned, in chronological order?",
        "Does the speaker express strong emotions? Provide evidence from the dialogue.",
        "What problems does the speaker mention most frequently?",
        "What actions does the speaker plan to take next?",
        "Summarize the speaker's daily routine as described in the dialogue.",
        "What people or places are mentioned, and in what context?",
        "Are there contradictions across different turns? Identify them.",
        "What is the overall tone of the conversation (neutral/positive/negative)?",
        "Which parts of the dialogue contain the most concrete details?",
        "Provide a brief summary of the speaker's situation."
    ]


def shuffle_words(sentence: str, rng: random.Random) -> str:
    tokens = sentence.replace("?", " ?").split()
    rng.shuffle(tokens)
    return " ".join(tokens).replace(" ?", "?")


def build_prompts(prompt_mode: str, pid: int, seed: int, num_prompts: int):
    base = get_clinical_prompts()
    assert len(base) == num_prompts

    if prompt_mode == "clinical":
        return base

    if prompt_mode == "generic":
        g = get_generic_prompts()
        assert len(g) == num_prompts
        return g

    rng = random.Random((seed << 16) + int(pid))

    if prompt_mode == "shuffle_words":
        return [shuffle_words(q, rng) for q in base]

    if prompt_mode == "permute":
        qs = base[:]
        rng.shuffle(qs)
        return qs

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


# 配置参数
class Config:
    train_csv = ""
    test_csv = ""
    dev_csv = ""
    data_dir = ""
    bert_model_name = 'bert-base-uncased'
    batch_size = 8
    lr = 1e-5
    epochs = 200
    lstm_hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 20
    min_delta = 0.0001

    num_prompts = 11
    prompt_mode = "clinical"  # clinical | generic | shuffle_words | permute
    prompt_seed = 42
    query_source = "text"  # text | learned

    seed = 4


class MultiQADataset(Dataset):
    def __init__(self, data_dir, label_df, tokenizer, bert_model,
                 max_length=512, device='cpu', save_stats_path=None,
                 prompt_mode="clinical", prompt_seed=42, num_prompts=11):

        self.prompt_mode = prompt_mode
        self.prompt_seed = prompt_seed
        self.num_prompts = num_prompts

        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.max_length = max_length
        self.label_map = dict(zip(label_df['Participant_ID'], label_df['PHQ8_Binary']))
        self.PHQ8_Score_map = dict(zip(label_df['Participant_ID'], label_df['PHQ8_Score']))

        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

        self.participant_ids = [int(f.split('_')[0]) for f in os.listdir(data_dir)]

        self.used_participant_ids = []

        self.qa_pairs = []
        self.qa_masks = []

        self.multi_rounds = []

        self.labels = []
        self.Score = []
        for pid, path in zip(self.participant_ids, self.file_paths):
            if pid not in self.label_map: continue

            questions = build_prompts(
                prompt_mode=self.prompt_mode,
                pid=pid,
                seed=self.prompt_seed,
                num_prompts=self.num_prompts
            )

            prompt_sig = self.prompt_mode + "|" + _md5_text("\n".join(questions))
            cache_extra_key = f"prompts:{prompt_sig}"

            dataset = QADataset(
                question=questions,
                csv_path=path,
                tokenizer=tokenizer,
                model=bert_model,
                max_length=max_length,
                has_labels=False,
                device=device,
                cache_extra_key=cache_extra_key
            )

            self.qa_pairs.append((dataset.getALL()[:self.num_prompts, :, :],
                                  dataset.getALL()[self.num_prompts:, :, :]))
            self.qa_masks.append((dataset.getLengths()[:self.num_prompts, :],
                                  dataset.getLengths()[self.num_prompts:, :]))

            self.multi_rounds.append(dataset.getRound_outputs())
            self.labels.append(self.label_map[pid])
            self.Score.append(self.PHQ8_Score_map[pid])
            self.used_participant_ids.append(pid)

        self.Score = np.array(self.Score, dtype=np.float32)

        if save_stats_path is not None:
            with open(save_stats_path, "r") as f:
                stats = json.load(f)
            self.score_mean = stats["mean"]
            self.score_std = stats["std"]
        else:
            self.score_mean = self.Score.mean()
            self.score_std = self.Score.std()
            with open("score_stats.json", "w") as f:
                json.dump({
                    "mean": float(self.score_mean),
                    "std": float(self.score_std)
                }, f)

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        z_score = (self.Score[idx] - self.score_mean) / self.score_std
        return (self.qa_pairs[idx][0], self.qa_pairs[idx][1],
                self.qa_masks[idx][0], self.qa_masks[idx][1],
                self.multi_rounds[idx],
                self.labels[idx], z_score,
                self.used_participant_ids[idx])


def collate_fn(batch):
    q_inputs = [item[0] for item in batch]
    qa_inputs = [item[1] for item in batch]
    q_masks = [item[2] for item in batch]
    qa_masks = [item[3] for item in batch]

    multi_rounds = [item[4] for item in batch]

    labels = torch.LongTensor([item[5] for item in batch])

    scores = torch.tensor([item[6] for item in batch], dtype=torch.float32)

    participant_ids = [item[7] for item in batch]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_q = max([q.shape[0] for q in q_inputs])
    max_q_len = max([q.shape[1] for q in q_inputs])

    padded_q = torch.zeros(len(q_inputs), max_q, max_q_len, 768)
    padded_q_mask = torch.zeros(len(q_inputs), max_q, max_q_len, dtype=torch.long)

    for i, (q, mask) in enumerate(zip(q_inputs, q_masks)):
        num_pairs, seq_len = q.shape[0], q.shape[1]
        padded_q[i, :num_pairs, :seq_len] = q
        padded_q_mask[i, :num_pairs, :seq_len] = mask
        if padded_q_mask[i].sum() == 0:
            raise RuntimeError(f" {i} ")

    max_qa_pairs = max([qa.shape[0] for qa in qa_inputs])
    max_qa_len = max([qa.shape[1] for qa in qa_inputs])

    padded_qa = torch.zeros(len(qa_inputs), max_qa_pairs, max_qa_len, 768)
    padded_qa_mask = torch.zeros(len(qa_inputs), max_qa_pairs, max_qa_len, dtype=torch.long)

    for i, (qa, mask) in enumerate(zip(qa_inputs, qa_masks)):
        num_pairs, seq_len = qa.shape[0], qa.shape[1]
        padded_qa[i, :num_pairs, :seq_len] = qa
        padded_qa_mask[i, :num_pairs, :seq_len] = mask
        if padded_qa_mask[i].sum() == 0:
            raise RuntimeError(f"")

    round_keys = multi_rounds[0].keys()

    multi_round_outputs = {}

    num_prompts = q_inputs[0].shape[0]

    for round_k in round_keys:

        qa_multi = [sample[round_k]['sequence'][:num_prompts, :, :] for sample in multi_rounds]
        q_multi = [sample[round_k]['sequence'][num_prompts:, :, :] for sample in multi_rounds]
        qa_multi_mask = [sample[round_k]['attention_mask'][:num_prompts, :] for sample in multi_rounds]
        q_multi_mask = [sample[round_k]['attention_mask'][num_prompts:, :] for sample in multi_rounds]

        max_qa_len0 = max(seq.shape[0] for seq in qa_multi)
        max_qa_len1 = max(seq.shape[1] for seq in qa_multi)
        hidden_dim = qa_multi[0].shape[2]

        padded_qa_seq = torch.zeros(len(qa_multi), max_qa_len0, max_qa_len1, hidden_dim)
        padded_qa_maskm = torch.zeros(len(qa_multi), max_qa_len0, max_qa_len1, dtype=torch.long)

        for i, (seq, mask) in enumerate(zip(qa_multi, qa_multi_mask)):
            l0, l1 = seq.shape[0], seq.shape[1]
            padded_qa_seq[i, :l0, :l1] = seq
            padded_qa_maskm[i, :l0, :l1] = mask
            if padded_qa_maskm[i].sum() == 0:
                raise RuntimeError(f" {i}  {round_k} ")

        max_q_len0 = max(seq.shape[0] for seq in q_multi)
        max_q_len1 = max(seq.shape[1] for seq in q_multi)

        padded_q_seq = torch.zeros(len(q_multi), max_q_len0, max_q_len1, hidden_dim)
        padded_q_maskm = torch.zeros(len(q_multi), max_q_len0, max_q_len1, dtype=torch.long)

        for i, (seq, mask) in enumerate(zip(q_multi, q_multi_mask)):
            l0, l1 = seq.shape[0], seq.shape[1]
            padded_q_seq[i, :l0, :l1] = seq
            padded_q_maskm[i, :l0, :l1] = mask
            if padded_q_maskm[i].sum() == 0:
                raise RuntimeError(f" {i}  {round_k} ")

        multi_round_outputs[round_k] = {
            'q': (padded_qa_seq.to(device), padded_qa_maskm.to(device)),
            'qa': (padded_q_seq.to(device), padded_q_maskm.to(device))
        }

    # participant_ids
    return (
        padded_q.to(device),
        padded_q_mask.to(device),
        padded_qa.to(device),
        padded_qa_mask.to(device),
        multi_round_outputs,
        labels.to(device),
        scores.to(device),
    )


def evaluate(model, dataloader, mean, std, device, phase='Validation'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase} Processing"):
            q_input, q_mask, qa_input, qa_mask, multi_round, labels, scores = batch

            outputs = model(
                qa_input=qa_input,
                qa_mask=qa_mask,
                q_input=q_input,
                q_mask=q_mask,
                multi_rounds=multi_round
            ).squeeze()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            preds_phq8 = outputs.detach().cpu().numpy() * std + mean
            scores_phq8 = scores.detach().cpu().numpy() * std + mean

            if isinstance(preds_phq8, np.float32):
                preds_phq8 = [preds_phq8]
            if isinstance(scores_phq8, np.float32):
                scores_phq8 = [scores_phq8]

            all_preds.extend(preds_phq8)
            all_labels.extend(scores_phq8)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)

    print(f"{phase} Metrics:")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return mae, rmse, mse, all_preds, all_labels


def plot_metrics(train_hist, val_hist):
    epochs = [entry['epoch'] for entry in train_hist]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MAE and RMSE over Epochs', fontsize=18)

    # MAE
    axs[0, 0].plot(epochs, [entry['mae'] for entry in train_hist], label='Train MAE', color='blue')
    axs[0, 0].set_title("Train MAE")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MAE")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, [entry['mae'] for entry in val_hist], label='Dev MAE', color='orange')
    axs[0, 1].set_title("Dev MAE")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("MAE")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # RMSE
    axs[1, 0].plot(epochs, [entry['rmse'] for entry in train_hist], label='Train RMSE', color='blue')
    axs[1, 0].set_title("Train RMSE")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("RMSE")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, [entry['rmse'] for entry in val_hist], label='Dev RMSE', color='orange')
    axs[1, 1].set_title("Dev RMSE")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("RMSE")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('metric_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def main(config):
    start_time = time.time()

    train_label_df = pd.read_csv(config.train_csv)
    dev_label_df = pd.read_csv(config.dev_csv)
    test_label_df = pd.read_csv(config.test_csv)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    bert_model = BertModel.from_pretrained(config.bert_model_name).to(config.device)

    bert_model.eval()

    train_dataset = MultiQADataset(
        data_dir=config.data_dir,
        label_df=train_label_df,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=config.device,
        save_stats_path=None,
        prompt_mode=config.prompt_mode,
        prompt_seed=config.prompt_seed,
        num_prompts=config.num_prompts
    )

    dev_dataset = MultiQADataset(
        data_dir=config.data_dir,
        label_df=dev_label_df,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=config.device,
        save_stats_path="score_stats.json",
        prompt_mode=config.prompt_mode,
        prompt_seed=config.prompt_seed,
        num_prompts=config.num_prompts
    )

    test_dataset = MultiQADataset(
        data_dir=config.data_dir,
        label_df=test_label_df,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=config.device,
        save_stats_path="score_stats.json",
        prompt_mode=config.prompt_mode,
        prompt_seed=config.prompt_seed,
        num_prompts=config.num_prompts
    )

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              generator=g)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    model = BERTLSTMATTETION(
        hidden_size=config.lstm_hidden_size,
        query_source=config.query_source,
        num_prompts=config.num_prompts
    ).to(config.device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_mae = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    train_history = []
    val_history = []
    test_history = []

    mean = train_dataset.score_mean
    std = train_dataset.score_std
    print(f"Train set stats - Mean: {mean:.4f}, Std: {std:.4f}")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} Training"):
            q_input, q_mask, qa_input, qa_mask, multi_round, labels, scores = batch

            outputs = model(
                qa_input=qa_input,
                qa_mask=qa_mask,
                q_input=q_input,
                q_mask=q_mask,
                multi_rounds=multi_round
            ).squeeze()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            loss = loss_fn(outputs, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count

        val_mae, val_rmse, val_mse, _, _ = evaluate(model, dev_loader, mean, std, config.device, phase='dev')
        val_history.append({
            'epoch': epoch + 1,
            'mae': val_mae,
            'rmse': val_rmse,
            'mse': val_mse
        })

        if val_mae < best_mae - config.min_delta:
            best_mae = val_mae
            best_epoch = epoch + 1
            epochs_no_improve = 0

            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved at epoch {best_epoch} with MAE: {best_mae:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{config.patience} epochs")

            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - Loss: {avg_loss:.4f}, Val MAE: {val_mae:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best model at epoch {best_epoch} with Val MAE: {best_mae:.4f}")

    model.load_state_dict(torch.load('best_model.pth'))
    print("\nEvaluating on test set...")
    test_mae, test_rmse, test_mse, test_preds, test_labels = evaluate(
        model, test_loader, mean, std, config.device, phase='Test'
    )

    results = {
        'config': vars(config),
        'test_results': {
            'mae': test_mae,
            'rmse': test_rmse,
            'mse': test_mse
        },
        'predictions': list(zip(test_labels, test_preds))
    }

    dev_mae, dev_rmse, dev_mse, dev_preds, dev_labels = evaluate(model, dev_loader, mean, std, config.device,
                                                                 phase='dev')
    dev_results = {
        'dev_results': {
            'mae': dev_mae,
            'rmse': dev_rmse,
            'mse': dev_mse
        },
        'predictions': list(zip(dev_labels, dev_preds))
    }

    train_mae, train_rmse, train_mse, train_preds, train_labels = evaluate(model, train_loader, mean, std,
                                                                           config.device, phase='train')
    train_results = {
        'train_results': {
            'mae': train_mae,
            'rmse': train_rmse,
            'mse': train_mse
        },
        'predictions': list(zip(train_labels, train_preds))
    }

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(i) for i in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    with open('test.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    with open('train.json', 'w') as f:
        json.dump(convert_to_serializable(train_results), f, indent=2)
    with open('dev.json', 'w') as f:
        json.dump(convert_to_serializable(dev_results), f, indent=2)

    print("Results saved to training_results.json")

    # plot_metrics(train_history, val_history)


if __name__ == "__main__":
    config = Config()
    set_seed(config.seed, deterministic=getattr(config, "deterministic", True))
    main(config)