import random
import json
import torch
from model import NeuralNet
from Pre_process import bag_of_words, tokenize
import pickle

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('train.json','r') as f:
        intents = json.load(f)
    
    FILE = 'data.pth'

    data = torch.load(FILE)

    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    pickle.dump(model , open('nn.pk1' , 'wb'))
    
except Exception as e:
    print("Error occured, ",e)
