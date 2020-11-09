
# Pytorch implementation for AGBAN

## 1. Introduction
Pytorch implementation for [Object-aware Multimodal Named Entity Recognition in Social Media Posts with Adversarial Learning](https://ieeexplore.ieee.org/document/9154571). This implementation based on the [NCRF++](https://github.com/jiesutd/NCRFpp).

## 2. Requirements
1. Python 3.6 or higher
2. Pytorch 1.1.0 or higher
4. You need to download the word embedding from [glove.twitter.27B.zip](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
4. You can download the multimodal Tweet data from [twitter2015](https://github.com/jefferyYu/UMT)

## 3. Usage
1. set the `status` attribute in demo.train.config to `train` or `decode` , and then

   ```
   python main.py --config demo.train.config
   ```

   ​


