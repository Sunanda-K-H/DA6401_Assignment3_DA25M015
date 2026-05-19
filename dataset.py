from datasets import load_dataset
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        # pass
        self.split = split

        self.data = load_dataset("bentrevett/multi30k", split=split)  # load split

        self.spacy_de = spacy.load("de_core_news_sm")  # German tokenizer
        self.spacy_en = spacy.load("en_core_web_sm")  # English tokenizer

        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_itos = None
        self.tgt_itos = None

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        # raise NotImplementedError
        src_counter = Counter()  # German token counts
        tgt_counter = Counter()  # English token counts

        for x in self.data:

            src_text = x["de"]
            tgt_text = x["en"]

            src_tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(src_text)]  # tokenize source
            tgt_tokens = [tok.text.lower() for tok in self.spacy_en.tokenizer(tgt_text)]  # tokenize target

            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        special_tokens = [self.unk_token, self.pad_token, self.sos_token, self.eos_token]  # fixed ids first

        self.src_vocab = {}
        self.tgt_vocab = {}

        for tok in special_tokens:

            self.src_vocab[tok] = len(self.src_vocab)
            self.tgt_vocab[tok] = len(self.tgt_vocab)

        for word in src_counter:

            if word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)

        for word in tgt_counter:

            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)

        self.src_itos = {idx: tok for tok, idx in self.src_vocab.items()}  # id to source token
        self.tgt_itos = {idx: tok for tok, idx in self.tgt_vocab.items()}  # id to target token

        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        # raise NotImplementedError
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()

        processed_data = []

        for x in self.data:

            src_text = x["de"]
            tgt_text = x["en"]

            src_tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(src_text)]  # source tokens
            tgt_tokens = [tok.text.lower() for tok in self.spacy_en.tokenizer(tgt_text)]  # target tokens

            src_ids = [self.src_vocab[self.sos_token]]  # start source sentence
            tgt_ids = [self.tgt_vocab[self.sos_token]]  # start target sentence

            for tok in src_tokens:
                src_ids.append(self.src_vocab.get(tok, self.src_vocab[self.unk_token]))

            for tok in tgt_tokens:
                tgt_ids.append(self.tgt_vocab.get(tok, self.tgt_vocab[self.unk_token]))

            src_ids.append(self.src_vocab[self.eos_token])  # end source sentence
            tgt_ids.append(self.tgt_vocab[self.eos_token])  # end target sentence

            processed_data.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))

        return processed_data


def collate_fn(batch, pad_idx=1):
    src_batch, tgt_batch = zip(*batch)  # split source and target

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)  # pad sources
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)  # pad targets

    return src_batch, tgt_batch
