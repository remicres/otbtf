# OTBTensorflow

This remote module of the [Orfeo ToolBox](https://www.orfeo-toolbox.org) (OTB) aims to provide a deep learning framework targeting remote sensing images processing.
It contains a set of new process objects that internally invoke [Tensorflow](https://www.tensorflow.org/), and a bunch of user-oriented applications to perform deep learning with real-world remote sensing images.

*Main highlights*
 - Sampling,
 - Training, supporting save/restore/import operations (a model can be trained from scratch or fine-tuned),
 - Serving models with support of OTB streaming mechanism 


# How to install
This remote module has been tested successfully on Ubuntu 16.04 and CentOs 7 with latest CUDA drivers.

## Build OTB
First, **build the latest *develop* branch of OTB from sources**. You can check the [OTB documentation](https://www.orfeo-toolbox.org/SoftwareGuide/SoftwareGuidech2.html) which details all the steps, if fact it is quite easy thank to the SuperBuild.

## Build TensorFlow
Then you have to **build Tensorflow from source** except if you want to use only the sampling applications of OTBTensorflow (in this case, skip this section).  
Follow [the instructions](https://www.tensorflow.org/install/install_sources) to build Tensorflow.

## Build this remote module
Finally, we can build this module.
Clone the repository in your the OTB sources directory for remote modules (something like `otb/Modules/Remote/`).
Re configure OTB with cmake of ccmake, and set the following variables

 - **Module_OTBTensorflow** to **ON**
 - **OTB_USE_TENSORFLOW** to **ON** (if you set to OFF, you will have only the sampling applications)
 - **TENSORFLOW_CC_LIB** to `/path/to/lib/libtensorflow_cc.so`
 - **TENSORFLOW_FRAMEWORK_LIB** to `/path/to/lib/libtensorflow_framework.so`
 - **tensorflow_include_dir** to `/path/to/include`

Re build and install OTB.
Done !

# New applications
Let's describe quickly the new applications provided.
## PatchesExtraction
This application performs the extraction of patches in images from a vector data containing points. The OTB sampling framework can be used to generate the set of selected points. After that, you can use the **PatchesExtraction** application to perform the sampling of your images.
We denote input source an input image, or a stack of input image (of the same size !). The user can set the **OTB_TF_NSOURCES** environment variable to select the number of input sources that he wants. For example, if she wants to sample a time series of Sentinel or Landsat, and in addition a very high resolution image like Spot-7 or Rapideye, she needs 2 sources (1 for the TS and 1 for the VHRS). The sampled patches will be extracted at each positions designed by the points, if they are entirely inside all input images. For each image source, patches sizes must be provided.
For each source, the application export all sampled patches as a single multiband raster, stacked in rows. For instance, if you have a number *n* of samples of size *16 x 16* in a *4* channels source image, the output image will be a raster of size *16 x 16n* with *4* channels. 
An optional output is an image of size *1 x n* containing the value of one specific field of the input vector data. Typically, the *class* field can be used to generate a dataset suitable for a model that performs pixel wise classification. 
```
This application extracts patches in multiple input images. Change the OTB_TF_NSOURCES environment variable to set the number of sources.
Parameters: 
        -source1            <group>          Parameters for source 1 
MISSING -source1.il         <string list>    Input image(s) 1  (mandatory)
MISSING -source1.out        <string> [pixel] Output patches for image 1  [pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is float) (mandatory)
MISSING -source1.patchsizex <int32>          X patch size for image 1  (mandatory)
MISSING -source1.patchsizey <int32>          Y patch size for image 1  (mandatory)
MISSING -vec                <string>         Positions of the samples (must be in the same projection as input image)  (mandatory)
        -outlabels          <string> [pixel] output labels  [pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is uint8) (optional, off by default)
        -field              <string>         field of class in the vector data  (mandatory)
        -inxml              <string>         Load otb application from xml file  (optional, off by default)
        -progress           <boolean>        Report progress 
        -help               <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.

Examples: 
otbcli_PatchesExtraction -vec points.sqlite -source1.il $s2_list -source1.patchsizex 16 -source1.patchsizey 16 -field class -source1.out outpatches_16x16.tif -outlabels outlabels.tif

```
## Build your Tensorflow model
You can build your Tensorflow model as shown in the `otb/Modules/Remote/otbtensorflow/python` directory. The high-level Python API of Tensorflow is used here to explort a *SavedModel* that applications of this remote module can read.
However, you can use any deep net available on the web, or use an existing gui application to create your own Tensorflow models.
The important thing here is to know the following parameters for your **placeholders** (the inputs of your model) and **output tensors** (the outputs of your model).
 - For each **input placeholder**:
 -- Name
 -- Perceptive field
 - For each **output tensor**:
 --Name 
 --Expression field
 --Scale factor

![alt text](shema.png "Figure 1")

Here the scale factor is related to one of the model inputs. It tells if your model perform a physical change of spacing of the output (e.g. introduced by non unitary strides in pooling or convolution operators). For each output, it must be expressed relatively to one single input called the reference input.
Additionally, you will need to remember the **target nodes** (e.g. optimizers, ...) used for training and every other placeholder that are important, especially user placeholders that are used only for training without default value (e.g. "dropout value").

## Create a new model
You can use the tool of your choice to build a tensorflow model. This is out of scope of this remote module. However, you can take a look in the `otb/Modules/Remote/otbtensorflow/python` to check how its done typically using the high level python API of Tensorflow. Python purists can even train their own models, thank to Python bindings of OTB: to get patches as 4D numpy arrays, just read the patches images with OTB (**ExtractROI** application for instance) and get the output float vector image as numpy array. Then, simply do a np.reshape to the dimensions that you want ! 
## Train your Tensorflow model
Here we assume that you have produced patches using the **PatchesExtraction** application, and that you have a model stored in a directory somewhere on your filesystem. The **TensorflowModelTrain** performs the training, validation (against test dataset, and against validation dataset) providing the usual metrics that machine learning frameworks provide (confusion matrix, recall, precision, f-score, ...).
Set you input data for training and for validation. The validation against test data is performed on the same data as for training, and the validation against the validation data, well, is performed on the dataset that you give to the application. You can set also batches sizes, and custom placeholders for single valued tensors for both training and validation. The last is useful if you have a model that behaves differently depending the given placeholder. Let's take the example of dropout: it's nice for training, but you have to disable it to use the model. Hence you will pass a placeholder with dropout=0.3 for training and dropout=0.0 for validation. 
```
This is the  (TensorflowModelTrain) application, version 6.5.0

Train a multisource deep learning net using Tensorflow. Change the OTB_TF_NSOURCES environment variable to set the number of sources.
Parameters: 
        -model                          <group>          Model parameters 
MISSING -model.dir                      <string>         Tensorflow model_save directory  (mandatory)
        -model.restorefrom              <string>         Restore model from path  (optional, off by default)
        -model.saveto                   <string>         Save model to path  (optional, off by default)
        -training                       <group>          Training parameters 
        -training.batchsize             <int32>          Batch size  (mandatory, default value is 100)
        -training.epochs                <int32>          Number of epochs  (mandatory, default value is 10)
        -training.userplaceholders      <string list>    Additional single-valued placeholders for training. Supported types: int, float, bool.  (optional, off by default)
MISSING -training.targetnodesnames      <string list>    Names of the target nodes  (mandatory)
        -training.outputtensorsnames    <string list>    Names of the output tensors to display  (optional, off by default)
        -training.source1               <group>          Parameters for source #1 (training) 
MISSING -training.source1.il            <string list>    Input image (or list to stack) for source #1 (training)  (mandatory)
MISSING -training.source1.fovx          <int32>          Field of view width for source #1  (mandatory)
MISSING -training.source1.fovy          <int32>          Field of view height for source #1  (mandatory)
MISSING -training.source1.placeholder   <string>         Name of the input placeholder for source #1 (training)  (mandatory)
        -training.source2               <group>          Parameters for source #2 (training) 
MISSING -training.source2.il            <string list>    Input image (or list to stack) for source #2 (training)  (mandatory)
MISSING -training.source2.fovx          <int32>          Field of view width for source #2  (mandatory)
MISSING -training.source2.fovy          <int32>          Field of view height for source #2  (mandatory)
MISSING -training.source2.placeholder   <string>         Name of the input placeholder for source #2 (training)  (mandatory)
        -validation                     <group>          Validation parameters 
        -validation.mode                <string>         Metrics to compute [none/class/rmse] (mandatory, default value is none)
        -validation.userplaceholders    <string list>    Additional single-valued placeholders for validation. Supported types: int, float, bool.  (optional, off by default)
        -validation.source1             <group>          Parameters for source #1 (validation) 
        -validation.source1.il          <string list>    Input image (or list to stack) for source #1 (validation)  (mandatory)
        -validation.source1.placeholder <string>         Name of the input placeholder for source #1 (validation)  (mandatory)
        -validation.source2             <group>          Parameters for source #2 (validation) 
        -validation.source2.il          <string list>    Input image (or list to stack) for source #2 (validation)  (mandatory)
        -validation.source2.placeholder <string>         Name of the input placeholder for source #2 (validation)  (mandatory)
        -inxml                          <string>         Load otb application from xml file  (optional, off by default)
        -progress                       <boolean>        Report progress 
        -help                           <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.

Examples: 
otbcli_TensorflowModelTrain -source1.il spot6pms.tif -source1.placeholder x1 -source1.fovx 16 -source1.fovy 16 -source2.il labels.tif -source2.placeholder y1 -source2.fovx 1 -source2.fovy 1 -model.dir /tmp/my_saved_model/ -training.userplaceholders is_training=true dropout=0.2 -training.targetnodenames optimizer -model.saveto /tmp/my_saved_model_vars1
```
As you can note, there is `$OTB_TF_NSOURCES` + 1 sources for practical purpose: because we need at least 1 source for input data, and 1 source for the truth.
## Serve the model
The **TensorflowModelServe** application perform model serving, it can be used to produce output raster with the desired tensors. Thanks to the streaming mechanism, very large images can be produced. The application uses the `TensorflowModelFilter` and a `StreamingFilter` to force the streaming of output. This last can be optionally disabled by the user, if he prefers using the extended filenames to deal with chunk sizes. however, it's still very useful when the application is used in other composites applications, or just without extended filename magic. Some models can consume a lot of memory. In addition, the native tiling strategy of OTB consists in strips but this might not always the best. For Constitutional Neural Networks for instance, square tiles are more interesting because the padding required to perform the computation of one single strip of pixels induces to input a lot more pixels that to process the computation of one single tile of pixels.
So, this application takes in input one or multiple images (remember that you can change the number of inputs by setting the `OTB_TF_NSOURCES` to the desired number) and produce one output of the specified tensors.
Like it was said before, the user is responsible of giving the *perceptive field* and *name* of input placeholders, as well as the *expression field*, *scale factor* and *name* of the output tensors. The user can ask for multiple tensors, that will be stack along the channel dimension of the output raster. However, if the sizes of those output tensors are not consistent (e.g. a different number of (x,y) elements), an exception will be thrown.
```
This is the  (TensorflowModelServe) application, version 6.5.0

Multisource deep learning classifier using Tensorflow. Change the OTB_TF_NSOURCES environment variable to set the number of sources.
Parameters: 
        -source1                  <group>          Parameters for source #1 
MISSING -source1.il               <string list>    Input image (or list to stack) for source #1  (mandatory)
MISSING -source1.fovx             <int32>          Field of view width for source #1  (mandatory)
MISSING -source1.fovy             <int32>          Field of view height for source #1  (mandatory)
MISSING -source1.placeholder      <string>         Name of the input placeholder for source #1  (mandatory)
        -model                    <group>          model parameters 
MISSING -model.dir                <string>         Tensorflow model_save directory  (mandatory)
        -model.userplaceholders   <string list>    Additional single-valued placeholders. Supported types: int, float, bool.  (optional, off by default)
        -model.fullyconv          <boolean>        Fully convolutional  (optional, off by default, default value is false)
        -output                   <group>          Output tensors parameters 
        -output.spcscale          <float>          The output spacing scale  (mandatory, default value is 1)
MISSING -output.names             <string list>    Names of the output tensors  (mandatory)
        -output.foex              <int32>          The output field of expression (x)  (mandatory, default value is 1)
        -output.foey              <int32>          The output field of expression (y)  (mandatory, default value is 1)
        -finetuning               <group>          Fine tuning performance or consistency parameters 
        -finetuning.disabletiling <boolean>        Disable tiling  (optional, off by default, default value is false)
        -finetuning.tilesize      <int32>          Tile width used to stream the filter output  (mandatory, default value is 16)
MISSING -out                      <string> [pixel] output image  [pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is float) (mandatory)
        -inxml                    <string>         Load otb application from xml file  (optional, off by default)
        -progress                 <boolean>        Report progress 
        -help                     <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.

Examples: 
otbcli_TensorflowModelServe -source1.il spot6pms.tif -source1.placeholder x1 -source1.fovx 16 -source1.fovy 16 -model.dir /tmp/my_saved_model/ -model.userplaceholders is_training=false dropout=0.0 -output.names out_predict1 out_proba1 -out "classif128tgt.tif?&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue=256" -finetuning.disabletiling

```
## Composite applications for classification
Who has never dreamed to use classic classifiers performing on deep learning features?
This is possible thank to two new applications that uses the existing training/classification applications of OTB:

 - TrainClassifierFromDeepFeatures: is a composite application that wire the **TensorflowModelServe** application output into the existing official **TrainImagesClassifier** application. 
```
This is the TrainClassifierFromDeepFeatures (TrainClassifierFromDeepFeatures) application, version 6.5.0

Train a classifier from deep net based features of an image and training vector data.
Parameters: 
        -source1                     <group>          Parameters for source 1 
MISSING -source1.il                  <string list>    Input image (or list to stack) for source #1  (mandatory)
MISSING -source1.fovx                <int32>          Field of view width for source #1  (mandatory)
MISSING -source1.fovy                <int32>          Field of view height for source #1  (mandatory)
MISSING -source1.placeholder         <string>         Name of the input placeholder for source #1  (mandatory)
        -model                       <group>          Deep net model parameters 
MISSING -model.dir                   <string>         Tensorflow model_save directory  (mandatory)
        -model.userplaceholders      <string list>    Additional single-valued placeholders. Supported types: int, float, bool.  (optional, off by default)
        -model.fullyconv             <boolean>        Fully convolutional  (optional, off by default, default value is false)
        -output                      <group>          Deep net outputs parameters 
        -output.spcscale             <float>          The output spacing scale  (mandatory, default value is 1)
MISSING -output.names                <string list>    Names of the output tensors  (mandatory)
        -output.foex                 <int32>          The output field of expression (x)  (mandatory, default value is 1)
        -output.foey                 <int32>          The output field of expression (y)  (mandatory, default value is 1)
        -finetuning                  <group>          Deep net fine tuning parameters 
        -finetuning.disabletiling    <boolean>        Disable tiling  (optional, off by default, default value is false)
        -finetuning.tilesize         <int32>          Tile width used to stream the filter output  (mandatory, default value is 16)
MISSING -vd                          <string list>    Input vector data list  (mandatory)
        -valid                       <string list>    Validation vector data list  (optional, off by default)
MISSING -out                         <string>         Output model  (mandatory)
        -confmatout                  <string>         Output model confusion matrix  (optional, off by default)
        -sample                      <group>          Training and validation samples parameters 
        -sample.mt                   <int32>          Maximum training sample size per class  (mandatory, default value is 1000)
        -sample.mv                   <int32>          Maximum validation sample size per class  (mandatory, default value is 1000)
        -sample.bm                   <int32>          Bound sample number by minimum  (mandatory, default value is 1)
        -sample.vtr                  <float>          Training and validation sample ratio  (mandatory, default value is 0.5)
        -sample.vfn                  <string>         Field containing the class integer label for supervision  (mandatory, default value is )
        -elev                        <group>          Elevation management 
        -elev.dem                    <string>         DEM directory  (optional, off by default)
        -elev.geoid                  <string>         Geoid File  (optional, off by default)
        -elev.default                <float>          Default elevation  (mandatory, default value is 0)
        -classifier                  <string>         Classifier [libsvm/boost/dt/gbt/ann/bayes/rf/knn/sharkrf/sharkkm] (mandatory, default value is libsvm)
        -classifier.libsvm.k         <string>         SVM Kernel Type [linear/rbf/poly/sigmoid] (mandatory, default value is linear)
        -classifier.libsvm.m         <string>         SVM Model Type [csvc/nusvc/oneclass] (mandatory, default value is csvc)
        -classifier.libsvm.c         <float>          Cost parameter C  (mandatory, default value is 1)
        -classifier.libsvm.nu        <float>          Cost parameter Nu  (mandatory, default value is 0.5)
        -classifier.libsvm.opt       <boolean>        Parameters optimization  (mandatory, default value is false)
        -classifier.libsvm.prob      <boolean>        Probability estimation  (mandatory, default value is false)
        -classifier.boost.t          <string>         Boost Type [discrete/real/logit/gentle] (mandatory, default value is real)
        -classifier.boost.w          <int32>          Weak count  (mandatory, default value is 100)
        -classifier.boost.r          <float>          Weight Trim Rate  (mandatory, default value is 0.95)
        -classifier.boost.m          <int32>          Maximum depth of the tree  (mandatory, default value is 1)
        -classifier.dt.max           <int32>          Maximum depth of the tree  (mandatory, default value is 65535)
        -classifier.dt.min           <int32>          Minimum number of samples in each node  (mandatory, default value is 10)
        -classifier.dt.ra            <float>          Termination criteria for regression tree  (mandatory, default value is 0.01)
        -classifier.dt.cat           <int32>          Cluster possible values of a categorical variable into K <= cat clusters to find a suboptimal split  (mandatory, default value is 10)
        -classifier.dt.f             <int32>          K-fold cross-validations  (mandatory, default value is 10)
        -classifier.dt.r             <boolean>        Set Use1seRule flag to false  (mandatory, default value is false)
        -classifier.dt.t             <boolean>        Set TruncatePrunedTree flag to false  (mandatory, default value is false)
        -classifier.gbt.w            <int32>          Number of boosting algorithm iterations  (mandatory, default value is 200)
        -classifier.gbt.s            <float>          Regularization parameter  (mandatory, default value is 0.01)
        -classifier.gbt.p            <float>          Portion of the whole training set used for each algorithm iteration  (mandatory, default value is 0.8)
        -classifier.gbt.max          <int32>          Maximum depth of the tree  (mandatory, default value is 3)
        -classifier.ann.t            <string>         Train Method Type [back/reg] (mandatory, default value is reg)
        -classifier.ann.sizes        <string list>    Number of neurons in each intermediate layer  (mandatory)
        -classifier.ann.f            <string>         Neuron activation function type [ident/sig/gau] (mandatory, default value is sig)
        -classifier.ann.a            <float>          Alpha parameter of the activation function  (mandatory, default value is 1)
        -classifier.ann.b            <float>          Beta parameter of the activation function  (mandatory, default value is 1)
        -classifier.ann.bpdw         <float>          Strength of the weight gradient term in the BACKPROP method  (mandatory, default value is 0.1)
        -classifier.ann.bpms         <float>          Strength of the momentum term (the difference between weights on the 2 previous iterations)  (mandatory, default value is 0.1)
        -classifier.ann.rdw          <float>          Initial value Delta_0 of update-values Delta_{ij} in RPROP method  (mandatory, default value is 0.1)
        -classifier.ann.rdwm         <float>          Update-values lower limit Delta_{min} in RPROP method  (mandatory, default value is 1e-07)
        -classifier.ann.term         <string>         Termination criteria [iter/eps/all] (mandatory, default value is all)
        -classifier.ann.eps          <float>          Epsilon value used in the Termination criteria  (mandatory, default value is 0.01)
        -classifier.ann.iter         <int32>          Maximum number of iterations used in the Termination criteria  (mandatory, default value is 1000)
        -classifier.rf.max           <int32>          Maximum depth of the tree  (mandatory, default value is 5)
        -classifier.rf.min           <int32>          Minimum number of samples in each node  (mandatory, default value is 10)
        -classifier.rf.ra            <float>          Termination Criteria for regression tree  (mandatory, default value is 0)
        -classifier.rf.cat           <int32>          Cluster possible values of a categorical variable into K <= cat clusters to find a suboptimal split  (mandatory, default value is 10)
        -classifier.rf.var           <int32>          Size of the randomly selected subset of features at each tree node  (mandatory, default value is 0)
        -classifier.rf.nbtrees       <int32>          Maximum number of trees in the forest  (mandatory, default value is 100)
        -classifier.rf.acc           <float>          Sufficient accuracy (OOB error)  (mandatory, default value is 0.01)
        -classifier.knn.k            <int32>          Number of Neighbors  (mandatory, default value is 32)
        -classifier.sharkrf.nbtrees  <int32>          Maximum number of trees in the forest  (mandatory, default value is 100)
        -classifier.sharkrf.nodesize <int32>          Min size of the node for a split  (mandatory, default value is 25)
        -classifier.sharkrf.mtry     <int32>          Number of features tested at each node  (mandatory, default value is 0)
        -classifier.sharkrf.oobr     <float>          Out of bound ratio  (mandatory, default value is 0.66)
        -classifier.sharkkm.maxiter  <int32>          Maximum number of iteration for the kmeans algorithm.  (mandatory, default value is 10)
        -classifier.sharkkm.k        <int32>          The number of class used for the kmeans algorithm.  (mandatory, default value is 2)
        -rand                        <int32>          User defined rand seed  (optional, off by default)
        -inxml                       <string>         Load otb application from xml file  (optional, off by default)
        -progress                    <boolean>        Report progress 
        -help                        <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.
```
 - ImageClassifierFromDeepFeatures: same thing using the official **ImageClassifier**.
 ```
 This is the ImageClassifierFromDeepFeatures (ImageClassifierFromDeepFeatures) application, version 6.5.0

Classify image using features from a deep net and an OTB machine learning classification model
Parameters: 
        -source1                    <group>          Parameters for source 1 
MISSING -source1.il                 <string list>    Input image (or list to stack) for source #1  (mandatory)
MISSING -source1.fovx               <int32>          Field of view width for source #1  (mandatory)
MISSING -source1.fovy               <int32>          Field of view height for source #1  (mandatory)
MISSING -source1.placeholder        <string>         Name of the input placeholder for source #1  (mandatory)
        -deepmodel                  <group>          Deep net model parameters 
MISSING -deepmodel.dir              <string>         Tensorflow model_save directory  (mandatory)
        -deepmodel.userplaceholders <string list>    Additional single-valued placeholders. Supported types: int, float, bool.  (optional, off by default)
        -deepmodel.fullyconv        <boolean>        Fully convolutional  (optional, off by default, default value is false)
        -output                     <group>          Deep net outputs parameters 
        -output.spcscale            <float>          The output spacing scale  (mandatory, default value is 1)
MISSING -output.names               <string list>    Names of the output tensors  (mandatory)
        -output.foex                <int32>          The output field of expression (x)  (mandatory, default value is 1)
        -output.foey                <int32>          The output field of expression (y)  (mandatory, default value is 1)
        -finetuning                 <group>          Deep net fine tuning parameters 
        -finetuning.disabletiling   <boolean>        Disable tiling  (optional, off by default, default value is false)
        -finetuning.tilesize        <int32>          Tile width used to stream the filter output  (mandatory, default value is 16)
MISSING -model                      <string>         Model file  (mandatory)
        -imstat                     <string>         Statistics file  (optional, off by default)
        -nodatalabel                <int32>          Label mask value  (optional, off by default, default value is 0)
MISSING -out                        <string> [pixel] Output image  [pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is uint8) (mandatory)
        -confmap                    <string> [pixel] Confidence map image  [pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is double) (optional, off by default)
        -ram                        <int32>          Ram  (optional, off by default, default value is 128)
        -inxml                      <string>         Load otb application from xml file  (optional, off by default)
        -progress                   <boolean>        Report progress 
        -help                       <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.
```
Note that you can still set the `OTB_TF_NSOURCES` environment variable.
Some examples and tutorial are coming soon for this part :)
# Practice
Here we will try to provide a simple example of doing a classification using a deep net that performs on one single VHRS image.
Our data set consists in one Spot-7 image, *spot7.tif*, and a training vector data, *terrain_truth.shp* that qualifies two classes that are forest / non-forest.
First, we **compute statistics** of the vector data : how many points can we sample inside objects, and how many objects in each class.
```
otbcli_PolygonClassStatistics -vec terrain_truth.shp -in spot7.tif -out vec_stats.xml
```
Then, we will select some samples with the **SampleSelection** application of the existing machine learning framework of OTB.
```
otbcli_SampleSelection -in spot7.tif -vec terrain_truth.shp -instats stats.xml -field class -out points.shp
```
Ok. Now, let's use our **PatchesExtraction** application. Out model has a perceptive field of 16x16 pixels. 
We want to produce one image of patches, and one image for the corresponding labels.
```
otbcli_PatchesExtraction -in spot7.tif -patchsizex 16 -patchsizey 16 -vec samplespos.shp -field class -outlabels samp_labels.tif -outpatches samp_patches.tif
```
That's it. Now we have two images for patches and labels. If we wanna, we can split them to distinguish test/validation groups (with the **ExtractROI** application for instance). But here, we will just perform some fine tuning of our model, located in the `outmodel` directory. Our model is quite basic. It has two input placeholders, **x1** and **y1** respectively for input patches (with size 16x16) and input reference labels (with size 1x1). We named **prediction** the tensor that predict the labels and the optimizer that perform the stochastic gradient descent is an operator named **optimizer**. We perform the fine tuning and we export the new model variables in the `newvars` folder.
Let's use our **TensorflowModelTrain** application to perform the training of this existing model.
```
otbcli_TensorflowModelTrain -model.dir /path/to/oursavedmodel -training.targetnodesnames optimizer -training.source1.il samp_patches.tif -training.source1.fovx 16 -training.source1.fovy 16 -training.source1.placeholder x1 -training.source2.il samp_labels.tif -training.source2.fovx 1 -training.source2.fovy 1 -training.source2.placeholder -y1 -model.saveto newvars
```
Note that we could also have performed validation in this step. In this case, the `validation.source2.placeholder` would be different than the `training.source2.placeholder`, and would be **prediction**. This way, the program know what is the target tensor to evaluate. 

After this step, we decide to produce an entire map of forest over the whole Spot-7 image. First, we duplicate the model, and we replace its variable with the new ones that have been computed in the previous step.
Then, we use the **TensorflowModelServe** application to produce the **prediction** tensor output for the entire image.
```
otbcli_TensorflowModelServe -source1.il spot7.tif -source1.placeholder x1 -source1.fovx 16 -source1.fovy 16 -model.dir /tmp/my_new_model -output.names prediction -out map.tif uint8
```

# Contact
You can contact Rémi Cresson if you have any issues with this remote module at remi [dot] cresson [at] irstea [dot] fr


