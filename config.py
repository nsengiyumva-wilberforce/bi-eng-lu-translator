from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_options": 20,
        "lr": 10**-4,
        "seq_len": 64,
        "d_model": 512,
        "csv_path": "data/en-lg/en-lg.csv",
        "lang_src": "en",
        "lang_tgt": "lg",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "num_epochs": 100,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') /model_folder / model_filename)

