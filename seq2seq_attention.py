import torch
import torch.nn as nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder_Attention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (seq_length, N, embedding_size)
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)

        # hidden shape: (2, N, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))  # torch.cat(forward, backward)
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))  # torch.cat(forward, backward)

        return encoder_states, hidden, cell


class Decoder_Attention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell): # (x, Encoder, Decoder, cell)
        # x shape: (N) but we want (1, N)
        x = x.unsqueeze(0)

        # embedding shape: (1, N, embedding_size)
        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshape = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2)))  # torch.cat((hidden_size, hidden_size*2), dim=2))
        # attention shape: (sequence_length, N, 1)
        attention = self.softmax(energy)

        attention = attention.permute(1, 2, 0)  # (N, 1, sequence_length)

        encoder_states =encoder_states.permute(1, 0, 2)  # (N, sequence_length, hidden_size*2)

        # (N, 1, hidder_size*2) --> (1, N, hidder_size*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)

        # shape of outputs: (1, N, hidden_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # shape of predictions: (1, N, length of vocab)
        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq_Attention(nn.Module):
    def __init__(self, encoder, decoder, trg_voc):
        super(Seq2Seq_Attention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_voc = trg_voc

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        target_vocab_size = len(self.trg_voc.vocab)

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]  # start token
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs