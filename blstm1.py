import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_conll(path, labeled=True):
    sents, tags = [], []
    cur_s, cur_t = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_s:
                    sents.append(cur_s)
                    tags.append(cur_t)
                    cur_s, cur_t = [], []
                continue
            parts = line.split()
            cur_s.append(parts[1])
            cur_t.append(parts[2] if labeled else "O")
        if cur_s:
            sents.append(cur_s)
            tags.append(cur_t)
    return sents, tags


def make_vocab(sents):
    w2i = {"<PAD>": 0, "<UNK>": 1}
    for s in sents:
        for w in s:
            if w not in w2i:
                w2i[w] = len(w2i)
    return w2i


def make_tagset(all_tags):
    t2i = {}
    for tags in all_tags:
        for t in tags:
            if t not in t2i:
                t2i[t] = len(t2i)
    return t2i


def get_f1(gold_path, pred_path):
    golds, preds = [], []
    with open(gold_path, encoding="utf-8") as gf, open(pred_path, encoding="utf-8") as pf:
        for gline in gf:
            pline = pf.readline()
            g, p = gline.strip(), pline.strip()
            if not g:
                golds.append(None)
                preds.append(None)
                continue
            golds.append(g.split()[2])
            pt = p.split()
            preds.append(pt[2] if len(pt) >= 3 else "O")

    def extract_chunks(seq):
        chunks, start, etype = set(), None, None
        for i, tag in enumerate(seq):
            if tag is None:
                if start is not None:
                    chunks.add((etype, start, i))
                    start, etype = None, None
                continue
            if tag.startswith("B-"):
                if start is not None:
                    chunks.add((etype, start, i))
                etype, start = tag[2:], i
            elif tag.startswith("I-"):
                if start is None or etype != tag[2:]:
                    if start is not None:
                        chunks.add((etype, start, i))
                    etype, start = tag[2:], i
            else:
                if start is not None:
                    chunks.add((etype, start, i))
                    start, etype = None, None
        if start is not None:
            chunks.add((etype, start, len(seq)))
        return chunks

    gold_c = extract_chunks(golds)
    pred_c = extract_chunks(preds)
    correct = gold_c & pred_c
    prec = len(correct) / len(pred_c) * 100 if pred_c else 0
    rec = len(correct) / len(gold_c) * 100 if gold_c else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


class NERData(Dataset):
    def __init__(self, sents, tags, w2i, t2i):
        self.sents = sents
        self.tags = tags
        self.w2i = w2i
        self.t2i = t2i

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        x = torch.tensor([self.w2i.get(w, 1) for w in self.sents[i]])
        y = torch.tensor([self.t2i[t] for t in self.tags[i]])
        return x, y


def pad_batch(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    xp = pad_sequence(xs, batch_first=True, padding_value=0)
    yp = pad_sequence(ys, batch_first=True, padding_value=-1)
    return xp, yp, lens


class BLSTM(nn.Module):
    def __init__(self, vocab_sz, n_tags):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, 100, padding_idx=0)
        self.drop1 = nn.Dropout(0.33)
        self.lstm = nn.LSTM(100, 256, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.33)
        self.fc = nn.Linear(512, 128)
        self.elu = nn.ELU()
        self.out = nn.Linear(128, n_tags)

    def forward(self, x, lens):
        e = self.drop1(self.emb(x))
        packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = self.drop2(h)
        return self.out(self.elu(self.fc(h)))


def write_preds(model, sents, w2i, i2t, path):
    model.eval()
    with open(path, "w", encoding="utf-8") as f:
        for sent in sents:
            x = torch.tensor([w2i.get(w, 1) for w in sent]).unsqueeze(0).to(device)
            l = torch.tensor([len(sent)])
            with torch.no_grad():
                pred = model(x, l).argmax(-1).squeeze(0).cpu().tolist()
            for j, (w, p) in enumerate(zip(sent, pred)):
                f.write(f"{j+1} {w} {i2t[p]}\n")
            f.write("\n")


train_s, train_t = read_conll(os.path.join("data", "train"))
dev_s, dev_t = read_conll(os.path.join("data", "dev"))
test_s, _ = read_conll(os.path.join("data", "test"), labeled=False)

w2i = make_vocab(train_s)
t2i = make_tagset(train_t)
i2t = {v: k for k, v in t2i.items()}
print(f"Device: {device}")
print(f"Vocab: {len(w2i)}, Tags: {len(t2i)}")

loader = DataLoader(NERData(train_s, train_t, w2i, t2i),
                    batch_size=32, shuffle=True, collate_fn=pad_batch)

model = BLSTM(len(w2i), len(t2i)).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.12, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

best = 0
for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb, lens in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb, lens)
        loss = loss_fn(logits.view(-1, len(t2i)), yb.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    write_preds(model, dev_s, w2i, i2t, "dev1.out")
    prec, rec, f1 = get_f1(os.path.join("data", "dev"), "dev1.out")
    scheduler.step(f1)
    print(f"Epoch {epoch+1:2d} | Loss {total_loss/len(loader):.4f} | P {prec:.2f} R {rec:.2f} F1 {f1:.2f}")
    if f1 > best:
        best = f1
        torch.save(model.state_dict(), "blstm1.pt")

model.load_state_dict(torch.load("blstm1.pt", weights_only=True))
write_preds(model, dev_s, w2i, i2t, "dev1.out")
write_preds(model, test_s, w2i, i2t, "test1.out")
prec, rec, f1 = get_f1(os.path.join("data", "dev"), "dev1.out")
print(f"\nFinal: P {prec:.2f} R {rec:.2f} F1 {f1:.2f}")
