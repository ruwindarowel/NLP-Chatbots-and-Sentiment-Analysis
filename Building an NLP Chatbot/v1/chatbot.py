import streamlit as st
import random
import json
import torch
from model import NeuralNet
from Pre_process import bag_of_words, tokenize

#Backend

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

#Frontend
st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
#React to user input
prompt = st.chat_input("What is up?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        
    st.session_state.messages.append({"role": "user",
                                      "content":prompt})
    
    
    sentence = tokenize(str(prompt))
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) #1 row
    X = torch.from_numpy(X).to(device)
        
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
        
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
        
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
    else:
        response = "Meyaa meh pakak obala"
        
    with st.chat_message("assistant"):
        st.markdown(response)
    
      
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )