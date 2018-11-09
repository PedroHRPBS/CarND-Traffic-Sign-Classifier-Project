# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Dataset%20samples.png "Dataset samples"
[image2]: ./examples/Dataset%20distribution.png "Dataset distribution"
[image3]: ./examples/Original%20to%20Grayscale.png "Original to Grayscale"
[image4]: ./examples/Grayscale%20to%20Normalized.png  "Grayscale to Normalized"
[image5]: ./examples/Normalized%20to%20Rotated.png  "Normalized to Rotated"
[image6]: ./examples/Rotated%20to%20Translated.png  "Rotated to Translated"
[image7]: ./examples/Translated%20to%20Random%20brightness.png  "Translated to Random brightness"
[image8]: ./examples/Original%20to%20After%20Pipeline.png "Original to After Pipeline"
[image9]: ./examples/Training%20data%20distribution%20after%20augmentation.png "Training data distribution after augmentation"
[image10]: ./examples/Web%20samples.png "Web samples"
[image11]: ./examples/Softmax%20Probabilities.png "Softmax Probabilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/PedroHRPBS/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, we have 5 random samples of images from the training set, just for the reader to understand what kind of data we are working with.

![alt text][image1]

Then we have 3 histograms, that show how many images we have from each class.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because doing so, we can reduce the number of data to work with and increase the processing speed. Also, some papers have stated that using grayscale images have made their overall accuracy increase.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a second step, I normalized the image data because as stated in some papers, having a dataset with 0 mean and with standard deviation of 1 is always a good practice. Specially, when we are dealing with multiple features that have different value ranges. Doing so we prevent the features that have higher range to have a higher influence on the results.

Here is an example of the previous image after normalization.

![alt text][image4]

As a third step, I rotated the image following the paper from Sermanet and LeCun. Considering that in the real world the sign would be visible in different orientations, having the possibility to utilize this kind of data in training is a good technique to generalize the results.

Here is an example of the previous image after rotation.

![alt text][image5]

As a fourth step, I translated the image also following Sermanet and LeCun. Considering that the position of the sign inside the image doesn't interfere with the existence of the sign. Generating data with the same sign in different positions, improve the capability of generalization.

Here is an example of the previous image after translation.

![alt text][image6]

As a fifth step, I randomized the brightness of the image. Considering that the real world provide different intensities of illumination, it's important to have data that considers these changes.

Here is an example of the previous image after brightness randomization.

![alt text][image7]

After discretizing all the 5 steps that were developed, a pipeline that does all of them sequentially was created. 

Here is an example of a random image before and after the pipeline.

![alt text][image8]

---

I decided to generate additional data because, having different quantities of data for different classes makes the model trend in the direction of the class with more samples. And the classes that have fewer examples trend to be harder to predict.

Considering so, a technique to equalize the number of samples per class was implemented.

To add more data to the the data set, I used the following technique:
* First the training data set was entirely changed to grayscale and normalized
* After that, we found the number of samples that the class with more samples had.
* Then 2 loops were implemented. The first was a loop to loop through all the classes. The second was a loop that measured the number of samples of the respective class.
* With both loops implemented, we just had to generate a new image from that class (using the pipeline) until the class had the same number of images than the class with most images.

After that we achieved the following distribution:

![alt text][image9]

The difference between the original data set and the augmented data set is 34799 to 86430 images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (Following Sermanet and LeCun):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x1176			|
| RELU					|												|
| Flatten		| Conv1 (inputs 14x14x6 - outputs 1176)							|
| Flatten		| Conv3 (inputs 1x1x1176 - outputs 1176)						|
| Concatenation		| Conv1 and Conv3 - outputs 2352						|
| Fully Connected	   | outputs 43			|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following pipeline:

* EPOCHS = 30; BATCH_SIZE = 128; LEARNING RATE = 0.001
* Logits from Model(x) (mean = 0, standard deviation = 0.1
* Cross-Entropy from Softmax
* Reduce Mean for Loss operation
* Adam Optimizer
* Minimize(Loss operation)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.2% 
* test set accuracy of 92.7%

To achieve this final model, I did a mix between iterative approach and well known architecture.
First, I tried to use the LeNet architecture, but I got results that couldn't reach the minimum of 0.93.
I tried some different models that had a worse result than LeNet. So I started considering what I could do differently with my data. 
With that in mind, I figured out the approach of augmenting my data (I wasn't doing that before). After augmenting the data, LeNet was doing a better job than before, but it was still not sufficient.
I tried increasing the number of EPOCHS from 10 to 30, but also had no significant improvement.
After reading the Sermanet and LeCun paper, I tried to modify my LeNet implementation to look more similar to the model discussed on the paper.
That's when I finally arrived to a result that was higher than the minimum required.
Although I saw many implementations on the internet that achieved more than 98% of accuracy, I prefered to stick with my own, and on the future I will improve from that.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (after grayscale and normalization):

![alt text][image10] 

The last image might be difficult to classify because it is not on a good orientation and also because this sign has a complex drawing to comprehend on a low resolution image (even for humans).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Turn right ahead     			| Turn right ahead 										|
| No entry					| No entry											|
| 70 km/h	      		| 70 km/h					 				|
| Road work			| No passing for vehicles over 3.5 metric tons      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.7%. Considering what was mentioned before, in fact, the Road Work sign would be difficult to predict considering its characteristics.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were as described:

![alt text][image11] 
 
 We can see that the model was completely sure about the first 4 signs. But for the last one, the correct guess was the third guess of the model. 
