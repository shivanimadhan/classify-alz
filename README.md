# classify-alz
AlzNN.py uses an Artificial Neural Network (ANN) to classify Alzheimer's patients' MRI scans into four categories based on severity of disease: Non-demented, very mild demented, mild demented, and moderate demented.

## <a href= "https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images">Alzheimer's Dataset</a>

The dataset contains images of MRI segmentation and was uploaded to Kaggle by data engineer Sarvesh Dubey. The 6400 image files are divided into two sets — training and testing. Within each set, there are four categories based on the severity of disease in the MRI scans: MildDemented, VeryMildDemented, NonDemented, and ModerateDemented. More information about the dataset can be found on the Kaggle page from where it was retrieved.

## To Use

1. Download AlzNN.py and the Dataset folder.
2. <b>Lines 35-38</b>: Uncomment this section of the code — this will save the numpy arrays to your device. <i> Comment this section of the code again after you run the program for the first time.</i>
3. <b>Line 48</b>: Copy the paths of the <u>train</u> directory saved on your device and replace the current path. New code should be <code>with open('[train path].npy', 'rb') as f2:</code>
4.  <b>Line 50</b>: Copy the paths of the <u>test</u> directory saved on your device and replace the current path. New code should be <code>with open('[test path].npy', 'rb') as f3:</code>  
5.  Run the program using Terminal: <code>python AlzNN.py</code>.

## Other Notes

To increase the training and validation accuracy of the program, increase the number of epochs <b>(Line 121)</b> or increase the number of convolutional layers <b>(copy and paste Lines 68-69)</b>.


