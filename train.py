"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from collections import Counter
import math
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


def bleu_score(pred_sents, gold_sents, max_n=4):
    pred_len = 0
    gold_len = 0
    precisions = []

    for n in range(1, max_n + 1):
        clipped = 0
        total = 0

        for pred, gold_list in zip(pred_sents, gold_sents):
            gold = gold_list[0]
            pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(len(pred) - n + 1))
            gold_ngrams = Counter(tuple(gold[i:i+n]) for i in range(len(gold) - n + 1))

            clipped += sum(min(count, gold_ngrams[ngram]) for ngram, count in pred_ngrams.items())
            total += sum(pred_ngrams.values())

            if n == 1:
                pred_len += len(pred)
                gold_len += len(gold)

        precisions.append(clipped / total if total > 0 else 0.0)

    if pred_len == 0:
        return 0.0

    smooth_precisions = [p if p > 0 else 1e-9 for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in smooth_precisions) / max_n)
    brevity_penalty = 1.0 if pred_len > gold_len else math.exp(1 - gold_len / pred_len)

    return brevity_penalty * geo_mean


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        # raise NotImplementedError
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        # raise NotImplementedError

        log_probs = torch.log_softmax(logits, dim=-1)

        true_dist = torch.zeros_like(log_probs)

        true_dist.fill_(
            self.smoothing / (self.vocab_size - 2)
        )

        true_dist.scatter_(
            1,
            target.unsqueeze(1),
            1.0 - self.smoothing
        )

        true_dist[:, self.pad_idx] = 0

        pad_mask = (target == self.pad_idx)

        true_dist[pad_mask] = 0

        loss = -(true_dist * log_probs).sum(dim=-1)

        loss = loss[~pad_mask].mean()

        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_grad_norms: bool = False,
    log_confidence: bool = False,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    # raise NotImplementedError
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    mode = "train" if is_train else "eval"

    for src, tgt in tqdm(data_iter, desc=f"{mode} epoch {epoch_num}", leave=False):

        use_cuda = str(device).startswith("cuda")
        src = src.to(device, non_blocking=use_cuda)
        tgt = tgt.to(device, non_blocking=use_cuda)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_inp).to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(src, tgt_inp, src_mask, tgt_mask)

        vocab_size = logits.shape[-1]

        logits = logits.reshape(-1, vocab_size)
        tgt_out = tgt_out.reshape(-1)

        loss = loss_fn(logits, tgt_out)

        if is_train and log_confidence:
            import wandb
            pad_mask = (tgt_out != loss_fn.pad_idx)
            probs = torch.softmax(logits.detach(), dim=-1)
            correct_probs = probs.gather(1, tgt_out.unsqueeze(1)).squeeze(1)
            wandb.log({"prediction_confidence": correct_probs[pad_mask].mean().item()})

        if is_train:
            loss.backward()

            if log_grad_norms and total_batches < 1000:
                import wandb
                q_grad = model.encoder.layers[0].self_attention.wq.weight.grad
                k_grad = model.encoder.layers[0].self_attention.wk.weight.grad
                wandb.log({"grad_step": epoch_num * len(data_iter) + total_batches, "query_grad_norm": q_grad.norm().item(), "key_grad_norm": k_grad.norm().item()})

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    # raise NotImplementedError
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():

        memory = model.encode(src, src_mask)

        ys = torch.ones(1, 1, dtype=torch.long, device=device)
        ys = ys * start_symbol

        for i in range(max_len - 1):

            tgt_mask = make_tgt_mask(ys).to(device)

            logits = model.decode(memory, src_mask, ys, tgt_mask)

            next_logits = logits[:, -1, :]

            next_word = torch.argmax(next_logits, dim=-1)

            next_word = next_word.unsqueeze(1)

            ys = torch.cat([ys, next_word], dim=1)

            if end_symbol is not None and next_word.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    # raise NotImplementedError
    model.eval()

    bos_idx = tgt_vocab["<sos>"]
    eos_idx = tgt_vocab["<eos>"]
    pad_idx = tgt_vocab["<pad>"]
    tgt_itos = {idx: tok for tok, idx in tgt_vocab.items()}

    pred_sents = []
    gold_sents = []

    with torch.no_grad():

        for src, tgt in test_dataloader:

            src = src.to(device)
            tgt = tgt.to(device)

            batch = src.shape[0]

            for i in range(batch):

                src_i = src[i:i+1]
                tgt_i = tgt[i:i+1]

                src_mask = make_src_mask(src_i, pad_idx).to(device)

                pred_ids = greedy_decode(model, src_i, src_mask, max_len, bos_idx, eos_idx, device)

                pred_ids = pred_ids.squeeze(0).tolist()
                gold_ids = tgt_i.squeeze(0).tolist()

                pred_tokens = []
                gold_tokens = []

                for idx in pred_ids:

                    if idx == bos_idx or idx == pad_idx:
                        continue

                    if idx == eos_idx:
                        break

                    pred_tokens.append(tgt_itos[idx])

                for idx in gold_ids:

                    if idx == bos_idx or idx == pad_idx:
                        continue

                    if idx == eos_idx:
                        break

                    gold_tokens.append(tgt_itos[idx])

                pred_sents.append(pred_tokens)
                gold_sents.append([gold_tokens])

    return bleu_score(pred_sents, gold_sents) * 100


def log_attention_heatmaps(model, sample, src_vocab, device):
    import wandb
    import matplotlib.pyplot as plt

    src, _ = sample
    src = src.unsqueeze(0).to(device)
    src_mask = make_src_mask(src).to(device)
    src_itos = {idx: tok for tok, idx in src_vocab.items()}

    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    weights = model.encoder.layers[-1].self_attention.attn_weights[0].cpu()
    tokens = [src_itos[idx] for idx in src.squeeze(0).cpu().tolist()]

    for head in range(weights.shape[0]):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(weights[head, :len(tokens), :len(tokens)], cmap="viridis")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        ax.set_title(f"Encoder head {head}")
        fig.tight_layout()
        wandb.log({f"attention_head_{head}": wandb.Image(fig)})
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    # raise NotImplementedError
    model_config = {
        'src_vocab_size': model.src_vocab_size,
        'tgt_vocab_size': model.tgt_vocab_size,
        'd_model': model.d_model,
        'N': len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].num_heads,
        'd_ff': model.encoder.layers[0].d_ff,
        'dropout': model.encoder.layers[0].dropout.p,
        'scale_attention': model.scale_attention,
        'pos_encoding_type': model.pos_encoding_type
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model_config
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    # raise NotImplementedError
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    # raise NotImplementedError
    import wandb
    from dataset import Multi30kDataset, collate_fn
    from lr_scheduler import NoamScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    config = {
        "experiment": "label_smoothing_confidence",
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 32,
        "num_epochs": 10,
        "warmup_steps": 4000,
        "learning_rate": 1.0,
        "pad_idx": 1,
        "smoothing": 0.1,
        "scale_attention": True,
        "pos_encoding_type": "sinusoidal",
        "log_grad_norms": False,
        "log_confidence": True,
        "log_attention": False,
        "device": device,
        "num_workers": 1 if device == "cuda" else 0,
        "pin_memory": device == "cuda",
    }

    wandb.init(project="da6401-a3", name=config["experiment"], config=config)

    train_data = Multi30kDataset(split="train")
    train_data.build_vocab()
    train_processed = train_data.process_data()

    val_data = Multi30kDataset(split="validation")
    val_data.src_vocab = train_data.src_vocab
    val_data.tgt_vocab = train_data.tgt_vocab
    val_data.src_itos = train_data.src_itos
    val_data.tgt_itos = train_data.tgt_itos
    val_processed = val_data.process_data()

    test_data = Multi30kDataset(split="test")
    test_data.src_vocab = train_data.src_vocab
    test_data.tgt_vocab = train_data.tgt_vocab
    test_data.src_itos = train_data.src_itos
    test_data.tgt_itos = train_data.tgt_itos
    test_processed = test_data.process_data()

    train_loader = DataLoader(train_processed, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    val_loader = DataLoader(val_processed, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    test_loader = DataLoader(test_processed, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    src_vocab_size = len(train_data.src_vocab)
    tgt_vocab_size = len(train_data.tgt_vocab)

    model = Transformer(src_vocab_size, tgt_vocab_size, config["d_model"], config["N"], config["num_heads"], config["d_ff"], config["dropout"], scale_attention=config["scale_attention"], pos_encoding_type=config["pos_encoding_type"])
    model = model.to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])

    loss_fn = LabelSmoothingLoss(tgt_vocab_size, config["pad_idx"], config["smoothing"])

    for epoch in range(config["num_epochs"]):

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, config["device"], config["log_grad_norms"], config["log_confidence"])
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, config["device"])

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pt")

        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    bleu = evaluate_bleu(model, test_loader, train_data.tgt_vocab, config["device"])
    wandb.log({"test_bleu": bleu})
    print(f"test_bleu={bleu:.2f}")
    if config["log_attention"]:
        log_attention_heatmaps(model, test_processed[0], train_data.src_vocab, config["device"])


if __name__ == "__main__":
    run_training_experiment()
