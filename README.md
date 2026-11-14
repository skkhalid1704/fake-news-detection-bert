# Fake News Detection: Logistic Regression vs LSTM vs BERT

Master's dissertation group project (SRH University Berlin). Compares three text classifiers of increasing complexity on a merged fake/real news corpus:

- TF-IDF + Logistic Regression (baseline)
- LSTM (Keras)
- BERT (`bert-base-uncased`, fine-tuned with PyTorch / Hugging Face Transformers)

All models share one stratified 80/10/10 train/validation/test split. Evaluation uses accuracy, precision, recall, F1 and ROC-AUC, with confusion matrices, ROC curves and training curves.

## Files

- `fake_news_detection_lr_lstm_bert.ipynb`: full pipeline (loading, cleaning, EDA, training, evaluation, comparison)
- `slides/Final_Merged_Presentation.pptx`: project presentation

## Data

Not included (size). Download and place as follows:

- ISOT Fake News dataset: `data/isot/Fake.csv`, `data/isot/True.csv`
- WELFake dataset (cleaned): `data/welfake/dataset2_welfake_clean.csv`

## Run

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch transformers tqdm joblib
jupyter notebook fake_news_detection_lr_lstm_bert.ipynb
```

A GPU is recommended for the BERT section. Seed is fixed to 42.

## Authors

Khalid Ahamed Shaik and teammates.
