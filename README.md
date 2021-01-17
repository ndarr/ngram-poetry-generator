# N-Gram Poetry Generator
## Setup
### Download training data
```shell
wget https://github.com/anonymous-poetrybot-386/eacl-metrical-tagging-in-the-wild/raw/master/English/LargeCorpus/eng_gutenberg_measures_all.json.zip
unzip eng_gutenberg_measures_all.json.zip
```
### Install requirements
```shell
pip install -r requirements.txt
```

## Training
Trains a model with window size 9 and saves it as *model_state.pt*
```shell
python main.py
```

## Generation
Loads the stored model and generates 500 poems. Lastly the generated poems are saved in ngram_poems.txt
```shell
python main.py --generate
```