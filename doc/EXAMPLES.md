# Examples

Some examples of ready-to-use deep learning architectures built with the TensorFlow API from python.
All models used are provided in this directory.

**Table of Contents**
1. [Simple CNN](#part1)
2. [Fully convolutional network](#part2)
3. [M3Fusion Model](#part3)
4. [Maggiori model](#part4)
5. [Fully convolutional network with separate Pan/MS channels](#part5)

## Simple CNN <a name="part1"></a>

This simple model estimates the class of an input patch of image.
This model consists in successive convolutions/pooling/relu of the input (*x* placeholder).
At some point, the feature map is connected to a dense layer which has N neurons, N being the number of classes we want.
The training operator (*optimizer* node) performs the gradient descent of the loss function corresponding to the cross entropy of (the softmax of) the N neurons output and the reference labels (*y* placeholder).
Predicted label is the argmax of the N neurons (*prediction* tensor).
Predicted label is a single pixel, for an input patch of size 16x16 (for an input *x* of size 16x16, the *prediction* has a size of 1x1).
The learning rate of the training operator can be adjusted using the *lr* placeholder.
The following figure summarizes this architecture.

<img src ="https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/savedmodel_simple_cnn.png" />

### Generate the model

Use the python script to generate the SavedModel that will be used by OTBTF applications.

```
python create_savedmodel_simple_cnn.py --outdir $modeldir
```

Note that you can adjust the number of classes for the model with the `--nclasses` option.

### Train the model

Use **TensorflowModelTrain** to train this model.

```
otbcli_TensorflowModelTrain \
-model.dir $modeldir \
-model.saveto "$modeldir/variables/variables" \
-training.source1.il $patches_train -training.source1.patchsizex 1 -training.source1.patchsizey 1 -training.source1.placeholder "x" \
-training.source2.il $labels_train -training.source2.patchsizex 1 -training.source2.patchsizey 1 -training.source2.placeholder "y" \
-training.targetnodes "optimizer" \
-validation.mode "class" \
-validation.source1.il $patches_valid -validation.source1.name "x" \
-validation.source2.il $labels_valid -validation.source2.name "prediction"
```

Type `otbcli_TensorflowModelTrain --help` to display the help.

For instance, you can change the number of epochs to 50 with `-training.epochs 50` or you can change the batch size to 8 with `-training.batchsize 8`.
In addition, it is possible to feed some scalar values to scalar placeholder of the model (currently, bool, int and float are supported).
For instance, our model has a placeholder called *lr* that controls the learning rate of the optimizer.
We can change this value at runtime using `-training.userplaceholders "lr=0.0002"`

### Inference

This model can be used either in patch-based mode or in fully convolutional mode.

#### Patch-based mode

You can estimate the class of every pixels of your input image.
Since the model is able to estimate the class of the center value of a 16x16 patch, you can run the model over the whole image in patch-based mode.

```
otbcli_TensorflowModelServe \
-source1.il $image" \
-source1.rfieldx 16 \
-source1.rfieldy 16 \
-source1.placeholder "x" \
-model.dir $modeldir \
-output.names "prediction" \
-out $output_classif
```

However, patch-based approach is slow because each patch is processed independently, which is not computationally efficient.

#### Fully convolutional mode

In fully convolutional mode, the model is used to process larger blocks in order to estimate simultaneously multiple pixels classes.
The model has a total number of 4 strides (caused by pooling).
Hence the physical spacing of the features maps, in spatial dimensions, is divided by 4.
This is what is called *spcscale* in the **TensorflowModelServe** application.
If you want to use the model in fully convolutional mode, you have to tell **TensorflowModelServe** that the model performs a change of physical spacing of the output, 4 in our case.

```
otbcli_TensorflowModelServe \
-source1.il $image" \
-source1.rfieldx 16 \
-source1.rfieldy 16 \
-source1.placeholder "x" \
-output.names "prediction" \
-output.spcscale 4 \
-model.dir $modeldir \
-model.fullyconv on \
-out $output_classif_fcn
```

## Fully convolutional network <a name="part2"></a>

The `create_savedmodel_simple_fcn.py` script enables you to create a fully convolutional model which does not use any stride.

<img src ="https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/savedmodel_simple_fcnn.png" />

Thank to that, once trained this model can be applied on the image to produce a landcover map at the same resolution as the input image, in a fully convolutional (i.e. fast) manner.
The main difference with the model described in the previous section is the *spcscale* parameter that must be let to default (i.e. unitary).

Create the SavedModel using `python create_savedmodel_simple_fcn.py --outdir $modeldir` then train it as before.
Then you can produce the land cover map at pixel level in fully convolutional mode:
```
otbcli_TensorflowModelServe \
-source1.il $image" \
-source1.rfieldx 16 \
-source1.rfieldy 16 \
-source1.placeholder "x" \
-output.names "prediction" \
-model.dir $modeldir \
-model.fullyconv on \
-out $output_classif
```

## M3Fusion Model <a name="part3"></a>

The M3Fusion model (stands for MultiScale/Multimodal/Multitemporal satellite data fusion) is a model designed to input time series and very high resolution images.

Benedetti, P., Ienco, D., Gaetano, R., Ose, K., Pensa, R. G., & Dupuy, S. (2018). _M3Fusion: A Deep Learning Architecture for Multiscale Multimodal Multitemporal Satellite Data Fusion_. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 11(12), 4939-4949.

See the original paper [here](https://arxiv.org/pdf/1803.01945).

The M3 model is patch-based, and process two input sources simultaneously: (i) time series, and (ii) a very high resolution image.
The output class estimation is performed at pixel level.

### Generate the model

```
python create_savedmodel_ienco-m3_patchbased.py --outdir $modeldir
```

Note that you can adjust the number of classes for the model with the `--nclasses` option.
Type `python create_savedmodel_ienco-m3_patchbased.py --help` to see the other available parameters.

### Train the model

Let's train the M3 model from time series (TS) and Very High Resolution Satellite (VHRS) patches images.

<img src ="https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/model_training.png" />

First, tell OTBTF that we want two sources: one for time series + one for VHR image

```
export OTB_TF_NSOURCES=2
```

Run the **TensorflowModelTrain** application of OTBTF.

Note that for time series we could also have provided a list of images rather that a single big images stack (since "sourceX.il" is an input image list parameter).

```
otbcli_TensorflowModelTrain \
-model.dir $modeldir \
-model.saveto "$modeldir/variables/variables" \
-training.source1.il $patches_ts_train -training.source1.patchsizex 1 -training.source1.patchsizey 1 -training.source1.placeholder "x_rnn" \
-training.source2.il $patches_vhr_train -training.source2.patchsizex 25 -training.source2.patchsizey 25 -training.source2.placeholder "x_cnn" \
-training.source3.il $labels_train -training.source3.patchsizex 1 -training.source3.patchsizey 1 -training.source3.placeholder "y" \
-training.targetnodes "optimizer" \
-training.userplaceholders "is_training=true" "drop_rate=0.1" "learning_rate=0.0002" \
-validation.mode "class" -validation.step 1 \
-validation.source1.il $patches_ts_valid -validation.source1.name "x_rnn" \
-validation.source2.il $patches_vhr_valid -validation.source2.name "x_cnn" \
-validation.source3.il $labels_valid -validation.source3.name "prediction"
```

### Inference

Let's produce a land cover map using the M3 model from time series (TS) and Very High Resolution Satellite image (VHRS)

<img src ="https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/classif_map.png" />

Since we provide time series as the reference source (*source1*), the output classes are estimated at the same resolution.
This model can be run in patch-based mode only.

```
otbcli_TensorflowModelServe \
-source1.il $ts -source1.rfieldx 1 -source1.rfieldy 1 -source1.placeholder "x_rnn" \
-source2.il $vhr -source2.rfieldx 25 -source2.rfieldy 25 -source2.placeholder "x_cnn" \
-model.dir $modeldir \
-output.names "prediction" -out $output_classif
```

## Maggiori model <a name="part4"></a>

This architecture was one of the first to introduce a fully convolutional model suited for large scale remote sensing images.

Maggiori, E., Tarabalka, Y., Charpiat, G., & Alliez, P. (2016). _Convolutional neural networks for large-scale remote-sensing image classification_. IEEE Transactions on Geoscience and Remote Sensing, 55(2), 645-657.

See the original paper [here](https://hal.inria.fr/hal-01350706/document).
This fully convolutional model performs binary semantic segmentation of large scale images without any blocking artifacts.

### Generate the model

```
python create_savedmodel_maggiori17_fullyconv.py --outdir $modeldir
```

You can change the number of spectral bands of the input image that is processed with the model, using the `--n_channels` option.

### Train the model

The model perform the semantic segmentation from one single source.

```
otbcli_TensorflowModelTrain \
-model.dir $modeldir \
-model.saveto "$modeldir/variables/variables" \
-training.source1.il $patches_image_train -training.source1.patchsizex 80 -training.source1.patchsizey 80 -training.source1.placeholder "x" \
-training.source2.il $patches_labels_train -training.source2.patchsizex 16 -training.source2.patchsizey 16 -training.source2.placeholder "y" \
-training.targetnodes "optimizer" \
-training.userplaceholders "is_training=true" "learning_rate=0.0002" \
-validation.mode "class" -validation.step 1 \
-validation.source1.il $patches_image_valid -validation.source1.name "x" \
-validation.source2.il $patches_labels_valid -validation.source2.name "estimated" \
```

Note that the `userplaceholders` parameter contains the *is_training* placeholder, fed with value *true* because the default value for this placeholder is *false*, and it is used in the batch normalization layers (take a look in the `create_savedmodel_maggiori17_fullyconv.py` code).

### Inference

This model can be used in fully convolutional mode only.
This model performs convolutions with stride (i.e. downsampling), followed with transposed convolutions with strides (i.e. upsampling).
Since there is no change of physical spacing (because downsampling and upsampling have both the same number of strides), the *spcscale* parameter is let to default (i.e. unitary).
The receptive field of the model is 80x80, and the expression field is 16x16, due to the fact that the model keeps only the exact part of the output features maps.

```
otbcli_TensorflowModelServe \
-source1.il $image -source1.rfieldx 80 -source1.rfieldy 80 -source1.placeholder x \
-model.dir $modeldir \
-model.fullyconv on \
-output.names "estimated" -output.efieldx 16 -output.efieldy 16 \
-out $output_classif
```

## Fully convolutional network with separate Pan/MS channels <a name="part5"></a>

It's common that very high resolution products are composed with a panchromatic channel at high-resolution (Pan), and a multispectral image generally at lower resolution (MS).
This model inputs separately the two sources (Pan and MS) separately.

See: Gaetano, R., Ienco, D., Ose, K., & Cresson, R. (2018). A two-branch CNN architecture for land cover classification of PAN and MS imagery. Remote Sensing, 10(11), 1746.

<img src ="https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/savedmodel_simple_pxs_fcn.png" />

Use `create_savedmodel_pxs_fcn.py` to generate this model.

During training, the *x1* and *x2* placeholders must be fed respectively with patches of size 8x8 and 32x32.
You can use this model in a fully convolutional way with receptive field of size 32 (for the Pan image) and 8 (for the MS image) and an unitary expression field (i.e. equal to 1).
Don't forget to tell OTBTF that we want two sources: one for Ms image + one for Pan image

```
export OTB_TF_NSOURCES=2
```

### Inference at MS image resolution

Here we perform the land cover map at the same resolution as the MS image.
Do do this, we set the MS image as the first source in the **TensorflowModelServe** application.

```
otbcli_TensorflowModelServe \
-source1.il $ms -source1.rfieldx 8 -source1.rfieldy 8 -source1.placeholder "x1" \
-source2.il $pan -source2.rfieldx 32 -source2.rfieldy 32 -source2.placeholder "x2" \
-model.dir $modeldir \
-model.fullyconv on \
-output.names "prediction" \
-out $output_classif
```

Note that we could also have set the Pan image as the first source, and tell the application to use a *spcscale* of 4.
```
otbcli_TensorflowModelServe \
-source1.il $pan -source1.rfieldx 32 -source1.rfieldy 32 -source1.placeholder "x2" \
-source2.il $ms -source2.rfieldx 8 -source2.rfieldy 8 -source2.placeholder "x1" \
-model.dir $modeldir \
-model.fullyconv on \
-output.names "prediction" \
-output.spcscale 4 \
-out $output_classif
```

### Inference at Pan image resolution

Here we perform the land cover map at the same resolution as the Pan image.
Do do this, we set the Pan image as the first source in the **TensorflowModelServe** application.
Note that this model can not be applied in a fully convolutional fashion at the Pan image resolution.
We hence perform the processing in patch-based mode.

```
otbcli_TensorflowModelServe \
-source1.il $pan -source1.rfieldx 32 -source1.rfieldy 32 -source1.placeholder "x2" \
-source2.il $ms -source2.rfieldx 8 -source2.rfieldy 8 -source2.placeholder "x1" \
-model.dir $modeldir \
-output.names "prediction" \
-out $output_classif
```

Note that we could also have set the MS image as the first source, and tell the application to use a *spcscale* of 0.25.

```
otbcli_TensorflowModelServe \
-source1.il $ms -source1.rfieldx 8 -source1.rfieldy 8 -source1.placeholder "x1" \
-source2.il $pan -source2.rfieldx 32 -source2.rfieldy 32 -source2.placeholder "x2" \
-model.dir $modeldir \
-model.fullyconv on \
-out $output_classif
```
