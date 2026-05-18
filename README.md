# DA6401 Assignment 3 - Transformer for Machine Translation

Name: Kaki Hephzi Sunanda  
Roll Number: DA25M015  
Course: DA6401 - Introduction to Deep Learning  

W&B Report Link: TODO  
GitHub Repository Link: TODO  

## Repository Structure

```text
.
├── model.py          # Transformer model, attention, masks, positional encoding
├── train.py          # Training loop, checkpointing, BLEU evaluation, inference helpers
├── dataset.py        # Multi30k loading, spaCy tokenization, vocabulary, collate function
├── lr_scheduler.py   # Noam learning rate scheduler
├── requirements.txt  # Required Python packages
├── checkpoint.pt     # Trained model checkpoint for evaluation
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

Train the model:

```bash
python train.py
```

The script logs metrics to W&B and saves the best validation checkpoint as:

```text
checkpoint.pt
```

For autograder submission, include:

```text
model.py
train.py
dataset.py
lr_scheduler.py
requirements.txt
checkpoint.pt
README.md
```
