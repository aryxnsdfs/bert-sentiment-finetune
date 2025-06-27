# BERT Sentiment Classification

This project fine-tunes a `bert-base-uncased` model using HuggingFace Transformers to classify Amazon product reviews into Positive (1) or Negative (0) sentiment.

## How to Install:

## 🚀 How It Works

1. **Data Preparation**:
   - Load CSV file containing reviews and ratings.
   - Map ratings: 1–2 → Negative (0), 4–5 → Positive (1), drop rating = 3.
   - Drop null values and convert labels to integers.

2. **Tokenization**:
   - Use BERT tokenizer to encode the reviews to fixed-length input tensors.

3. **Custom Dataset**:
   - Define a PyTorch `Dataset` with `__init__`, `__len__`, and `__getitem__`.

4. **Model Training**:
   - Use `BertForSequenceClassification` with `Trainer` from HuggingFace.
   - Set training arguments and fine-tune the model on the training data.

5. **Prediction**:
   - Use a custom `predict()` function to classify new input text.

## 🧪 Sample Prediction

```python
print(predict("I loved this product!"))
# Output: (1, array([...])) → Positive sentiment
