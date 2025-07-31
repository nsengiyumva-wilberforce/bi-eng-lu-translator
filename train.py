import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation
from tokenizers.normalizers import Lowercase

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from tqdm import tqdm
import warnings

import pandas as pd

import random


# ======================
# 1. REPETITION PENALTY
# ======================
def apply_repetition_penalty(logits, generated_tokens, penalty=1.5):
    """Apply repetition penalty to logits"""
    for token_id in set(generated_tokens):
        logits[token_id] = logits[token_id] / penalty if logits[token_id] > 0 else logits[token_id] * penalty
    return logits

# ========================
# 2. ENTITY-FOCUSED LOSS
# ========================
def create_entity_weights(labels, tokenizer, entity_weight=2.0):
    """
    Create weight tensor with higher weights for entity tokens
    Simple heuristic: Capitalized words are considered entities
    """
    # Convert token IDs to text
    tokens = [tokenizer.id_to_token(id.item()) for id in labels[0]]
    text = " ".join(tokens)
    
    # Find entities using simple capitalization heuristic
    entity_indices = []
    for i, token in enumerate(tokens):
        if token.istitle() and i != 0:  # Skip sentence-start tokens
            entity_indices.append(i)
    
    # Create weights tensor
    weights = torch.ones_like(labels, dtype=torch.float)
    for idx in entity_indices:
        weights[0, idx] = entity_weight
        
    return weights


# ===============================
# 4. LUGANDA MORPHOLOGICAL AUGMENT
# ===============================
def augment_luganda(text, aug_prob=0.3):
    """
    Apply morphological augmentations to Luganda text:
    - Affix manipulation (add/remove prefixes/suffixes)
    - Compound word splitting
    """
    if random.random() > aug_prob:
        return text
    
    words = text.split()
    augmented = []
    
    for word in words:
        # Randomly add/remove prefixes (common Luganda prefixes)
        prefixes = ['mu', 'ba', 'ki', 'bu', 'lu', 'wa', 'ka', 'gu']
        if random.random() < 0.3 and len(word) > 3:
            if random.random() < 0.5:  # Remove prefix
                for prefix in prefixes:
                    if word.startswith(prefix):
                        word = word[len(prefix):]
                        break
            else:  # Add prefix
                word = random.choice(prefixes) + word
        
        # Randomly add/remove suffixes
        suffixes = ['a', 'e', 'o', 'mu', 'wa', 'ye', 'ko', 'wo']
        if random.random() < 0.3 and len(word) > 3:
            if random.random() < 0.5:  # Remove suffix
                for suffix in suffixes:
                    if word.endswith(suffix):
                        word = word[:-len(suffix)]
                        break
            else:  # Add suffix
                word = word + random.choice(suffixes)
                
        augmented.append(word)
    
    return " ".join(augmented)
    


# =============================
# MODIFIED GREEDY DECODE FUNCTION
# =============================
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, repetition_penalty=1.5):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    generated_tokens = []
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(out[:, -1])
        
        # Apply repetition penalty
        logits = logits.squeeze()
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)
        
        # Select next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs).unsqueeze(0)
        generated_tokens.append(next_token.item())
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_token.item()).to(device)], 
            dim=1
        )

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)

    
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    #size of the control window (just use a default value)
    console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len,device)
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        # Initialize tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        # Normalize to lowercase
        tokenizer.normalizer = Lowercase()
        
        # Split on whitespace + punctuation
        tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        
        # Train with vocabulary limits
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
            vocab_size= config['vocab_size'],
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# 2. Updated get_ds function
def get_ds(config, augment_fn=None):
    raw_data = load_translation_csv(config["csv_path"])

    # Wrap in a simple Dataset-like class
    class SimpleDataset:
        def __init__(self, data): self.data = data
        def __getitem__(self, idx): return self.data[idx]
        def __len__(self): return len(self.data)

    ds_raw = SimpleDataset(raw_data)

    # Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Train-validation split
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    # Dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'], augment_fn=augment_fn)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Max token length (for reference)
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of the source sentence: {max_len_src}')
    print(f'Max length of the target sentence: {max_len_tgt}')

    # Dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def load_translation_csv(csv_path):
    df = pd.read_csv(csv_path, sep=',', encoding='latin-1')

    print("CSV Columns:", df.columns.tolist())  # Debug print

    # tab-separated
    dataset = []
    for _, row in df.iterrows():
        if pd.notnull(row['English']) and pd.notnull(row['Luganda']):
            dataset.append({
                "translation": {
                    "en": row['English'].strip(),
                    "lg": row['Luganda'].strip()
                }
            })
    return dataset

def get_model(config, vocab_src_len, vocab_tgt_len):
    model= build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Enable TensorFloat-32 for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Add Luganda augmentation for target dataset
    augment_fn = None
    if config['luganda_augment']:
        augment_fn = augment_luganda

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, augment_fn)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_ouput = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_ouput.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # Entity-focused loss weighting
            if config['entity_weight'] > 1.0:
                entity_weights = create_entity_weights(
                    label, 
                    tokenizer_tgt, 
                    entity_weight=config['entity_weight']
                )
                weighted_loss = (loss * entity_weights.view(-1)).mean()
                loss = weighted_loss
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weight
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        run_validation(model, val_dataloader, tokenizer_src,tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
 

        #save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    config = get_config()
        # Set TF32 precision if enabled
    if config['tf32_enabled'] and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    train_model(config)