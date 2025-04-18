**LightCNN model Using Efficeint channel Attention**
<br>
Features from base model LightCNN29 V2 has been used through Attention which extracts features from both local and global.
For Global 16x16x192 image size has been gone through GAP(Global Average Polling) and Local features are being extracted by splitting the image into four region net of size 8x8x192 each.

**Dataset**
<br>
RAD-DB dataset has been used for this model which has 7 emotion classes.

**LightCNN Model**
<BR>
LigthCNN model has been introduced by [ A Light CNN for Deep Face Representation with Noisy Labels ]([https://example.com](https://arxiv.org/abs/1511.02683))

**Efficient Channel Attention**
<br>
This has been used for dimensionality reduction using 1-d CONV. The paper (https://arxiv.org/abs/1910.0315)] has described the to be effictive and reduce model paramerters.

**GradCAM**
(Gradcam.JPG)

