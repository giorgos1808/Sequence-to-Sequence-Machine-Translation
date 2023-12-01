import torch
import torch.nn as nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N, embedding_size)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) but we want (1, N)
        x = x.unsqueeze(0)

        # embedding shape: (1, N, embedding_size)
        embedding = self.dropout(self.embedding(x))

        # shape of outputs: (1, N, hidden_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        # shape of predictions: (1, N, length of vocab)
        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_voc):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_voc = trg_voc

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = len(self.trg_voc.vocab)

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]  # start token
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
