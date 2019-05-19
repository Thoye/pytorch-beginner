import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

# 预训练词向量
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep learning. It is a model that tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, since CBOW is not sequential and does not have to be probabilistic. Typcially, CBOW is used to quickly train word embeddings, and these embeddings are used to initialize the embeddings of some more complicated model. Usually, this is referred to as pretraining embeddings. It almost always helps performance a couple of percent.
N_DIM = 100
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):  # i从第2个到最后一个
    context = [
        raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]
    ]
    target = raw_text[i]  # 单个预测词
    data.append((context, target))  # 所有的(上下文词,预测词)词对


class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):  # context_size=4
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


model = CBOW(len(word_to_idx), N_DIM, CONTEXT_SIZE)

# if torch.cuda.is_available():
#     model = model.cuda()

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch {}'.format(epoch+1))
    print('*' * 10)
    running_loss = 0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[w] for w in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        # if torch.cuda.is_available():
        #     context = context.cuda()
        #     target = target.cuda()
        # forward
        out = model(context)
        loss = loss_function(out, target)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss: {:.6f}'.format(running_loss / len(data)))

context, target = data[0]  # word:We are to study, label:about
context = Variable(torch.LongTensor([word_to_idx[w] for w in context]))
out = model(context)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.item()]
print('real word is {}, predict word is {}'.format(target, predict_word))
