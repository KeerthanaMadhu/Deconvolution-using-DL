 # Sparse Blind Deconvolution using Deep Learning
 
1. The dataset is taken from kaggle.
   https://www.kaggle.com/shravankumar9892/image-colorization



2. The best model is in the location models/model_cnn_BN_uniform_sigma1_addnoise/
 
 
3. To run this code, the notebook Train_Inference.ipynb is sufficient.
   Run each cell step by step to load data, train the model and run inference on test data.



4. While training the models, a new name has to be given to the log folder and the model folder so that models can be saved after each      epoch and losses could be stored



![alt text](https://raw.githubusercontent.com/KeerthanaMadhu/Deep-Learning-for-SBD/master/image_readme.PNG)



#### The results of the best model are given below:

<p float="left">
  <img src="https://raw.githubusercontent.com/KeerthanaMadhu/Deep-Learning-for-SBD/master/orig_best.png" height = "250" width="250" />
  <img src="https://raw.githubusercontent.com/KeerthanaMadhu/Deep-Learning-for-SBD/master/inpu_best.png" height = "250" width="250"  /> 
  <img src="https://raw.githubusercontent.com/KeerthanaMadhu/Deep-Learning-for-SBD/master/rec_best.png" height = "250" width="250" />
</p>
