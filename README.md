# Source Code of AGBAN for Adversarial Object-aware Multimodal NER Model
Implementation of Our Paper "Object-aware Multimodal Named Entity Recognition in Social Media Posts with Adversarial Learning" in IEEE Trans. on Multimedia. This implementation is based on the [NCRF++](https://github.com/jiesutd/NCRFpp).

## Model Architecture


## Requirements
* `python >= 3.6`
* `pytorch >= 1.1.0`
* `NCRF++`

## Data Format
* You can download the multimodal dataset from  [twitter2015](https://github.com/jefferyYu/UMT)
* We adopt the glove embeddings to initialize our model which can be downloaded [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
* We preprocess the visual contents and extract the object features with [Mask-RCNN](https://github.com/matterport/Mask_RCNN). The preprocessed data will be available upon request.


## Usage
Set the `status` attribute in demo.train.config to `train` or `decode` , and then

   ```
   python main.py --config demo.train.config
   ```



