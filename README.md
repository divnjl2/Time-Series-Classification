# Analysis and Innovation in Time Series Classification Algorithms and Methods:
## Towards a Superior Convolutional Algorithm with Feature Selection


The present repository is dedicated to Time Series Classification. It contains the implementations based on two scopes:
### 1. Systematic evaluation on various datasets from the UCR archive.
   The datasets are split based on:
   * Dimensions (univariate or multivarite)
   * Length (<300, >=300, >700)
   * Classes (<10, >=10, >=30)
   We conducted our experiments on various datasets for this part, but we include here 11, since based on the former categorization UCR contains this number of datasets.
   We use numerous datasets to evaluate 21 classifiers which we separate in 6 categories:
   1. Deep Learning:
      * Multi Layer Perceptron (MLP)
      * Convolutional Neural Network (CNN)
      * Fully Convolutional Network (FCN)
      * Multi-Channels Deep Convolutional Neural Networks (MCD-CNN)
   2. Dictionary - based:
      * Bag of SFA Symbols (BOSS)
      * Contractable Bag of SFA Symbols (cBOSS)
      * Individual Bag of SFA Symbols (iBOSS)
      * Temporal Dictionary Ensemble (TDE)
      * Individual Temporal Dictionary Ensemble (iTDE)
      * Word Extraction for Time Series Classification (WEASEL)
      * MUltivariate Symbolic Extension (MUSE)
   3. Distance - based:
      * Shape Dynamic Time Warping (shapeDTW)
      * K-Nearest Neighbors (using DTW) (KNN)
   4. Feature - based:
      * Canonical Time-series Characteristics (catch22)
      * Fresh Pipeline with RotatIoN forest (FreshPRINCE)
   5. Interval - based:
      * Supervised Time Series Forest (STSF)
      * Time Series Forest (TSF)
      * Canonical Interval Forest (CIF)
      * Diverse Representation Canonical Interval Forest (DrCIF)
   6. Interval - based:
      * ROCKET
      * Arsenal

  We are using aeon library while except for MCD-CNN which is only available on sktime.
  For the evaluation we create metrics such as F1, Accuracy, Precision, AUC scores (macro and micro) while we also calculate execution times and memory consumptions.
  We also create plots such as macro average ROC AUC curve per classifier, confusion matrices and ROC AUC curves for each class.
  We also introduce results with and without cross-validation. For cross-validation we use both k-fold and TimeSeriesSplit. We use k-fold only for comparison reasons with other papers,
  since we are opposite to its use for time series data, for temporal structure reasons. Therefore, we always suggest TimeSeriesSplit if you want to use cross-validation.

### 2. Creation of a new convolutional algorithm that uses feature selection **(ConvFS)**
  Based on the results we produce, on the second part of the present thesis we introduce ConvFS (Convolutional Feature Selection), which
  achieves an incredible balance between accuracy and time. Our approach exhibits the highest performance in terms of time complexity, due to the use of random kernels in
  combination with feature selection - a method which we first introduce in combination with convolutional classifiers - especially when taking into consideration that time series
  classifiers, especially deep-leaning models, possess increased time complexities.

  We introduce ConvFS's flow below:
![Blank board](https://github.com/SophiaVei/TimeSeriesClassification/assets/92432705/34ea154c-4b89-4aa1-a02e-bcd3a5d0ce11)

  For this part we conduct our experiments on all 128 UCR datasets, plus 40 synthetic datasets.

  For this part, we also create plots such as relative accuracy with 1NN-DTW which is a baseline algorithm for TSC, train size vs time logarithmic plots and accuracy improvement plots
  comparing ConvFS with the other classifiers that we previously use for TSC.
  
  * We furthermore create example plots for time series such as ED and DTW distance measures to view their differences for time series data while also examples of time series and TSC
  * so that the viewer maintains a better idea.
  * Furthermore, for the first part, we also include Jupyter Notebooks so the viewer can easier find the outputs for the respective datasets.

