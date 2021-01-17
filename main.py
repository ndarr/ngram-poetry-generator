import torch
import jsonpickle
from torch.utils.data import DataLoader
import tqdm
from argparse import ArgumentParser
from utils import NgramModel, GutenbergDataset


def read_file_content(filename):
    file_ = open(filename, "r")
    content = file_.read()
    file_.close()
    return content


def read_eng_gutenberg_poems():
    content = read_file_content("eng_gutenberg_measures_all.json")
    json_content = jsonpickle.decode(content)
    poems = []
    for poem_id in json_content.keys():
        poem_entry = json_content[poem_id]['poem']
        stanzas = []
        for stanza_id in poem_entry.keys():
            stanza = poem_entry[stanza_id]
            lines = []
            for line_id in stanza.keys():
                line = stanza[line_id]
                lines.extend((line['text'].lower() + " <eol>").split())
            stanzas.extend(lines)
        poems.append(stanzas)
    return poems


def check_generated_poem_format(poem):
    if len(poem) < 20:
        return False
    lines = poem.split("\n")
    avg_line_len = sum([len(line) for line in lines]) / len(lines)
    for line in lines:
        if abs(len(line) - avg_line_len) > 3:
            return False
    return True


def store_poems_in_file(poems, filename="ngram_poems.txt"):
    f = open(filename, "w+")
    for poem in poems:
        f.write(poem + "\n")
    f.close()


argparser = ArgumentParser()
argparser.add_argument("--generate", default=False, type=bool, action="store_true")
args = argparser.parse_args()

window_size = 9
dataset = GutenbergDataset(read_eng_gutenberg_poems(), ngram_size=window_size)
print(dataset[0])
dataloader_params = {'batch_size': 8192,
                     'shuffle': True,
                     'num_workers': torch.cuda.device_count() * 2,
                     'pin_memory': True}

dataloader = DataLoader(dataset, **dataloader_params)
vocab = dataset.vocabulary

model = NgramModel(vocab_size=len(vocab), window_size=window_size)
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters())
celoss = torch.nn.CrossEntropyLoss()


if args.generate:
    model.load_state_dict(torch.load("model_state.pt"))
    model.eval()
    poems = []
    # Create 500 poems
    while len(poems) < 500:
        seq = [vocab.word_to_idx(vocab.sos)] * window_size
        pred_word = vocab.sos
        poem = []
        pred = -1.
        # Generate poem until 4 line endings are reached or 40 words have been generated
        while len(poem) < 40 and poem.count(vocab.eol) < 4:
            input_ = torch.tensor(seq).reshape((1, window_size)).to("cuda")
            probs, _ = model(input_, temperature=0.6)
            dist = torch.distributions.Categorical(probs.squeeze())
            pred = dist.sample().cpu().item()
            # Filter utility words from sampling
            while pred in [vocab.sos_idx, vocab.pad_idx, vocab.unk_idx]:
                pred = dist.sample().cpu().item()
            # Map sampled idx to real word
            pred_word = vocab.idx_to_word(pred)
            # Cut off the first token in the window and append the last predicted one
            seq = seq[1:] + [pred]
            # Add sampled word to poem
            poem.append(pred_word)

        # Always have a last end of line token
        if poem[-1] != vocab.eol:
            poem.append(vocab.eol)

        # Put seperate words together as one large string
        poem = " ".join(poem)

        # Filter out badly generated poems
        if check_generated_poem_format(poem):
            poems.append(poem)

    # Formatting with unix line endings
    poems = [poem.replace(vocab.eol, "\n").replace("\n ", "\n") for poem in poems]
    store_poems_in_file(poems)

    # Stop the execution after successful generation
    exit()


# Training
for epoch in range(10):
    tqdm_loader = tqdm.tqdm(dataloader)
    avg_loss = 0.
    for ngram, word in tqdm_loader:
        ngram = ngram.to("cuda")
        word = word.to("cuda")
        _, logits = model(ngram)
        loss = celoss(logits, word)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ = loss.cpu().item()
        avg_loss += loss_
        tqdm_loader.set_description("Loss: %f" % loss_)
    print("Loss epoch {}: {}".format(epoch, avg_loss / len(tqdm_loader)))
    torch.save(model.state_dict(), "model_state.pt")
