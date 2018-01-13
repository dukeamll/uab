# Framework for Image Segmentation/Classification of Satellite Images
1. [Installation of Tensorflow and dependencies](https://github.com/bohaohuang/aml-docs/blob/master/install_tensorflow.ipynb)
2. [Examples](./]examples)
3. [Supported Network](./bohaoCustom)

# TODO
- [ ] uabDataReader: none queue iterator GT order
- [ ] uabDataReader: none queue iterator data aug

# Known Bugs
- [ ] Loading a model to fine tune will cause problem by calling run() with pretrained_model_dir set in bohaoCustom/uabMakeNetwork_Unet.py
- [X] Unnecessary patches are extracted when input size equals tile size at testing
- [ ] Redundant reader initialization in model.evaluate()