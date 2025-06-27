import pandas as pd

df=pd.read_csv(r'c:\Users\aryan\Downloads\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

df=df[['reviews.text','reviews.rating']].dropna()

def label_sentiment(rating):
    if rating>4:
        return 1
    elif rating<=2:
        return 0
    else:
        return None
    
df['label']=df['reviews.rating'].apply(label_sentiment)
df = df.dropna(subset=['label']) 
df['label'] = df['label'].astype(int)

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['reviews.text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(texts,truncation=True,padding=True,max_length=128)

train_encoding=tokenize(train_texts)
test_encoding=tokenize(val_texts)

from torch.utils.data import Dataset
import torch

class ReviewDataset(Dataset):
    def __init__(self,encoding,labels):
        self.encoding=encoding
        self.labels=labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)
    

train_Dataset=ReviewDataset(train_encoding,train_labels)
test_Dataset=ReviewDataset(test_encoding,val_labels)
    
from transformers import Trainer,TrainingArguments

args=TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_eval_batch_size=5,
    per_device_train_batch_size=5,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=20
)
from transformers import BertForSequenceClassification
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)

trainer=Trainer(
    model=model,
    args=args,
    train_dataset=train_Dataset,
    eval_dataset=test_Dataset
)
 
trainer.train()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Move inputs to same device as model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model(**inputs)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    return torch.argmax(probs).item(), probs.detach().cpu().numpy()

print(predict("I loved this movie!"))        
