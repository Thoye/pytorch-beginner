import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
N_DIM = 100
# We will use Shakespeare Sonnet 2
raw_text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((raw_text[i], raw_text[i + 1]), raw_text[i + 2])
           for i in range(len(raw_text) - 2)]  # 所有的(上下文词,预测词)词对

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


class NgramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocab_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


ngrammodel = NgramModel(len(vocab), CONTEXT_SIZE, N_DIM)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for word in trigram:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[w] for w in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        # forward
        out = ngrammodel(context)
        loss = criterion(out, target)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

context, target = trigram[3]  # word:shall besiege, label:thy
context = Variable(torch.LongTensor([word_to_idx[w] for w in context]))
out = ngrammodel(context)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.item()]
print('real word is {}, predict word is {}'.format(target, predict_word))
