import torch as t
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence

docs = [
  "The Eiffel Tower is located in Paris, France.",
  "Cairo is the capital of Egypt.",
  "Python is a popular programming language for AI.",
]
anchors = [
        "Eiffel Tower.","Capital of Egypt","AI programming language"
    ]


def Tokenization(*data,len_docs,len_anchors):
    
    flatting = [sentence for d in data for arr in d for sentence in arr.split(" ")]
    vocab = {word:idx for idx,word in enumerate(set(flatting))}
    encoded = []
    for d in data: 
        for sentence in d : 
            temp = []
            for word in sentence.split(" "):
                if word in vocab:
                    temp.append(vocab[word])
            encoded.append(temp)
    encoded = [t.tensor(e) for e in encoded]      
    a = pad_sequence(encoded[:len_anchors],  batch_first=True, padding_value=0)
    p = pad_sequence(encoded[len_docs:],  batch_first=True, padding_value=0)
    
    return a,p,vocab

    
a , p, vocab = Tokenization(docs,anchors,len_docs=len(docs),len_anchors=len(anchors))


class Embedding(t.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size 
        self.d_model = d_model 
        self.embedding = t.nn.Embedding(vocab_size, d_model)
    
    def forward(self, x): 
        tok = self.embedding(x)        
        return tok.mean(dim=1)





model = Embedding(len(vocab),128)

opt = t.optim.Adam(model.parameters(), lr=1e-2)
temperature = 0.07
        
for epoch in range(500):
    opt.zero_grad()                           
        
    a_emb = F.normalize(model(a), dim=1)   
    p_emb = F.normalize(model(p), dim=1) 
        
    logits = a_emb @ p_emb.T / temperature       
    labels = t.arange(logits.size(0))            
        
    loss = F.cross_entropy(logits, labels)      
    loss.backward()                              
    opt.step()                                    
        
    if epoch % 20 == 0:
        print(epoch, loss.item())
        

model.eval()
with t.no_grad():
    test_query = "Cairo"
    test_ids = t.tensor([[vocab[w] for w in test_query.split(" ")]])
    q_emb = F.normalize(model(test_ids), dim=1)

    doc_embs = F.normalize(model(p), dim=1) 
    sims = q_emb @ doc_embs.T
    print(sims)
    best = sims.argmax().item()
    print("Best match:", p[best])
    print("Best match:", docs[best])


















