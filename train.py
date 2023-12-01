import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from seq2seq import Encoder, Decoder, Seq2Seq
from seq2seq_attention import Encoder_Attention, Decoder_Attention, Seq2Seq_Attention
import math
import time
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)][::-1]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def train(model, train_iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # output shape: (target_len, batch_size, output_dim)
        output = model(inp_data, target)
        output = output[1:].reshape(-1, output.shape[2])  # remove <sos> token
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])  # remove <sos> token
            target = target[1:].reshape(-1)

            loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------Preprocessing---------
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

german = Field(tokenize=tokenizer_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenizer_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, val_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(val_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")


# --------Training---------
CLIP = 1
num_epochs = 20
learning_rate = 1e-3
batch_size = 128

load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 500  # 300
decoder_embedding_size = 500  # 300

hidden_size = 512
num_layers = 1  # must
encoder_dropout = 0.3  # 0.5
decoder_dropout = 0.3  # 0.5

sentence_ger = "Künstliche Intelligenz ist genial"
sentence_eng = "Artificial intelligence is awesome"


def main(model, attention, example=False):
    train_iterator, val_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pad_inx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_inx)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        val_loss = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Perplexity: {math.exp(val_loss):7.3f}')

        if epoch == num_epochs - 1:
            if attention:
                save_checkpoint(checkpoint, filename="my_checkpoint_attention.pth.tar")
            else:
                save_checkpoint(checkpoint, filename="my_checkpoint.pth.tar")

        if example:
            print("Ger: " + sentence_ger)
            print("Eng: " + sentence_eng)
            translated_sentence = translate_sentence(model, sentence_ger, german, english, device, max_length=50, attention=attention)
            print("Translated:")
            print(translated_sentence)

    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Perplexity: {math.exp(test_loss):7.3f} |')

    score = bleu(test_data[1:100], model, german, english, device, attention=attention)
    print(f"Bleu score {score * 100:.2f}")


if __name__ == "__main__":

    print('Seq2Seq model without Attention mechanism')
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout).to(device)
    model = Seq2Seq(encoder_net, decoder_net, english).to(device)
    main(model=model, attention=False)

    print('Seq2Seq model with Attention mechanism')
    encoder_net = Encoder_Attention(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
    decoder_net = Decoder_Attention(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout).to(device)
    model = Seq2Seq_Attention(encoder_net, decoder_net, english).to(device)
    main(model=model, attention=True)
