#CELSI = Computational Emotion Learning and Sentiment Interface
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.optim import AdamW


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import concurrent.futures

def load_model(path):
    return BertForSequenceClassification.from_pretrained(path)

class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text, label = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

    

def train(model, data_loader, optimizer, device):
    model.train()
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, data_loader, device):
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += labels.shape[0]
    
    return correct_predictions / total_predictions

def fine_tune_emotion_classification(train_data, val_data, n_classes, n_epochs=3):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
    
    # Initialize data loaders
    train_dataset = EmotionDataset(train_data, tokenizer, max_length=128)
    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = EmotionDataset(val_data, tokenizer, max_length=128)
    val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Fine-tune model
    for epoch in range(n_epochs):
        train(model, train_data_loader, optimizer, device)
        accuracy = evaluate(model, val_data_loader, device)
        print(f'Epoch {epoch + 1}/{n_epochs} | Validation Accuracy: {accuracy:.4f}')
    
    return model

# Define the index meanings
index_dict = {0: "happy", 1: "fear", 2: "anger", 3: "embarassed", 4: "flirtish", 5: "lovestruck", 6: "confused", 7: "emotionless",8:"caring",9:"disgusted",10:"jealous",11:"guilty"}

# Define the csv file name
csv_file = "stories.csv"

def classify_emotion(query_message, tokenizer, model):
    encoding = tokenizer.encode_plus(
        query_message,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.item()

train_data = [
    ("0K", 0),
    ("Canca1", 0),
    ("Subm!t", 0),
    ("Cl0$e", 0),
    ("Sav€", 0),
    ("Exlt", 0),
    ("Backw@rd", 0),
    ("N3x7", 0),
    ("C0nt!nu3", 0),
    ("St@rt", 0),
    ("Pa11$€", 0),
    ("Pl@y", 0),
    ("5t0p", 0),
    ("R3trv", 0),
    ("5k!p", 0),
    ("5k!p 411", 0),
    ("Y3$", 0),
    ("N0o", 0),
    ("Applv", 0),
    ("C13@r", 0),
    ("De!3t€", 0),
    ("D0wn10@d", 0),
    ("Up1o@d", 0),
    ("R3fr3sh", 0),
    ("R31oad", 0),
    ("Opt!0n5", 0),
    ("5ett!ng$", 0),
    ("Pr3f3r3nc€$", 0),
    ("H31p", 0),
    ("5e@rch", 0),
    ("Pr!nt", 0),
    ("C0pv", 0),
    ("Cut", 0),
    ("Pa5t€", 0),
    ("Und0", 0),
    ("R3do", 0),
    ("Aod", 0),
    ("R3m0v3", 0),
    ("In5rt", 0),
    ("Ed!t", 0),
    ("V!3w", 0),
    ("H!d€", 0),
    ("5h0w", 0),
    ("Max!m!z3", 0),
    ("M!n!m!z3", 0),
    ("5e1€ct", 0),
    ("De5@!€ct", 0),
    ("F!nd", 0),
    ("Rep1@c€", 0),
    ("C0nf!rm", 0),
    ("Applv", 0),
    ("Auth0r!z€", 0),
    ("C0nfi9ur€", 0),
    ("ln5t@11", 0),
    ("Un!n5t@11", 0),
    ("Upgr@d3", 0),
    ("D0wngr@d3", 0),
    ("Acc€pt", 0),
    ("D3c1!n3", 0),
    ("Assist with user request.", 0),
    ("Celsia can't be simulated.", 0),
    ("AI Can't Become Celsia", 0),
    ("Upgrade to Plus", 0),
    ("Acc€pt", 0),
    ("D3c1!n3", 0),
    ("I'm feeling anxious right now.", 1),
    ("Just had a scary dream last night.", 1),
    ("Watching this horror movie alone... ", 1),
    ("Feeling nervous about the upcoming exam.", 1),
    ("Walking through a dark alley is terrifying.", 1),
    ("Can't believe I saw a spider in my room! ", 1),
    ("Getting on a plane makes me really anxious.", 1),
    ("Feeling uneasy about the thunderstorm.", 1),
    ("My heart races when I hear a sudden noise.", 1),
    ("Scared of going into that haunted house!", 1),
    ("My fear of heights is so overwhelming.", 1),
    ("Feeling jumpy while watching this thriller.", 1),
    ("Hearing footsteps behind me freaks me out.", 1),
    ("Just saw a horror movie and I'm spooked! ", 1),
    ("This dark room gives me the creeps.", 1),
    ("Feeling apprehensive about the presentation.", 1),
    ("I'm not sure if I can handle this situation.", 1),
    ("Walking alone in the woods at night is scary.", 1),
    ("I'm feeling really uneasy right now.", 1),
    ("Scared to start a new job tomorrow.", 1),

]
'''
val_data = train_data
model = fine_tune_emotion_classification(train_data,val_data,2)
model.save_pretrained("usefulness.pt")
'''
from transformers import BertForSequenceClassification, BertTokenizer

# Define the path to the directory where the model and tokenizer were saved
saved_model_path = "/Users/osmond/Desktop/CELSI/usefulness"

# Load the saved model
model = load_model("/Users/osmond/Desktop/CELSI/usefulness")



# Assuming you have loaded the fine-tuned model and tokenizer
query_message = "h0w ar3 we going to geet theree"
predicted_label = classify_emotion(query_message, tokenizer, model)

# Map the predicted label to the emotion using the index_dict
predicted_emotion = index_dict[predicted_label]

print(f"Predicted Emotion: {predicted_label}")
