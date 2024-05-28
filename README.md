# 🌟 snb 🌟

`snb` - a search robot for collecting images on the Internet, for preparing datasets and training neural networks.

## Technologies

- [Python](https://www.python.org)

## Download

Downloading the project from GitHub:

```bat
git clone git@github.com:artemkaFismat/snb.git
```

## Parameters

To run the code, you need to create a list of search queries on the topic that interests you, and place them in the file **search_queries.txt**

## Start

Launch options:

```text
1. --count - The number of images uploaded
2. --aug - Augmentation functions (true / false)
3. --quality - Quality (0 - 100%)
4. --folder - The path to the save folder
```

Launch project:

```bat
python3 snb.py --count 100 --aug true --quality 85 --folder /data/
```
