# Lomas, a flow trace generation model based on LDA

## Usage
### Preprocess
```
python3 preprocess.py --input_file <input-file>
```

### Train
```
python3 train.py --config_file config-pre.json
```

### Generate
```
python3 generate.py --config_file config-train.json
```