To down load the MSCOCO dataset:
```
sh download.sh
```

Preprocessing data
```
python3 resize.py
```

Build vocabulary for caption text
```
python3 build_vocab.py
```

Run Image Captioning
```
python3 captioning.py
```

Sample an image for testing
```
python3 sample.py --image_path <any image path>
python3 sample.py --image_path ./data/train2014/COCO_train2014_000000581921.jpg
```