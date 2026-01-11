# 配置参数
class Config:
    train_csv = "../train_split_Depression_AVEC2017.csv"
    test_csv = "../full_test_split.csv"
    dev_csv = "../dev_split_Depression_AVEC2017.csv"
    data_dir = "../Data"
    bert_model_name = 'bert-base-uncased'
    batch_size = 8
    lr = 1e-5
    epochs = 200
    lstm_hidden_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 20
    min_delta = 0.0001

    num_prompts = 11
    prompt_mode = "clinical"     # clinical | generic | shuffle_words | permute
    prompt_seed = 42
    query_source = "text"        # text | learned

    seed = 4