# CSS-6471

First, go to [Dropbox](https://www.dropbox.com/s/ig3ed7i5iepfbjj/final_data.zip?dl=0) and download final_data.zip. Then, uncompress it and place it in the root folder. Go inside final_data, and make sure lda_models is uncompressed as well.

To run simply run the results, run

`pip install -r requirements.txt`

`conda activate css_proj`

`cd code`

`python results.py`

To build the datasets from scratch (not recommended)

1. Make a folder called `data`
2. Get a Twitter API Key and save it into a file called `twitter_keys.yaml` in `/data/`
3. Move the files from `final_data` to `data`
4. Run `collectData.py` in `pull_tweets`
5. Download language model from [FastAI Language identification](https://fasttext.cc/docs/en/language-identification.html) (.bin version) and move to data folder
6. Run `eda.ipynb`

OG Dataset Sources:
earlier_tweets [source](https://www.kaggle.com/datasets/gpreda/tokyo-olympics-2020-tweets)
later_tweets [source](https://www.kaggle.com/datasets/amritpal333/tokyo-olympics-2021-tweets)




