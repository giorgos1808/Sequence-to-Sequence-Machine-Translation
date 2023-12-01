# Sequence-to-Sequence Machine Translation

This repository contains implementations of sequence-to-sequence (seq2seq) models for machine translation using PyTorch. Two models are included: one without attention and the other with attention.

## Files

### 1. seq2seq.py

This file defines the Seq2Seq model without attention.

- **Encoder Class:**
  - Parameters: `input_size`, `embedding_size`, `hidden_size`, `num_layers`, `p`.
  - Forward method: Processes the input sequence through an LSTM-based encoder.

- **Decoder Class:**
  - Parameters: `input_size`, `embedding_size`, `hidden_size`, `output_size`, `num_layers`, `p`.
  - Forward method: Processes the input, hidden, and cell states through an LSTM-based decoder.

- **Seq2Seq Class:**
  - Parameters: `encoder`, `decoder`, `trg_voc`.
  - Forward method: Takes source and target sequences and returns the decoder outputs.

### 2. seq2seq_attention.py

This file defines the Seq2Seq model with attention.

- **Encoder_Attention Class:**
  - Parameters: `input_size`, `embedding_size`, `hidden_size`, `num_layers`, `p`.
  - Forward method: Processes the input sequence through a bidirectional LSTM-based encoder with attention.

- **Decoder_Attention Class:**
  - Parameters: `input_size`, `embedding_size`, `hidden_size`, `output_size`, `num_layers`, `p`.
  - Forward method: Processes the input, encoder states, hidden, and cell states through an LSTM-based decoder with attention.

- **Seq2Seq_Attention Class:**
  - Parameters: `encoder`, `decoder`, `trg_voc`.
  - Forward method: Takes source and target sequences and returns the decoder outputs.

### 3. train.py

This file contains functions for training and evaluating the seq2seq models.

- Functions: `train`, `evaluate`, `epoch_time`, `count_parameters`.

This file demonstrates the usage of the seq2seq models on the Multi30k dataset.

- Functions: `tokenizer_ger`, `tokenizer_eng`, `main`.

## Dependencies

- PyTorch
- torchtext
- spacy

## Usage

1. Install the required dependencies:

   ```bash
   pip install torch torchtext spacy
