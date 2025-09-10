Setup
Dependencies:
Pytorch, numpy, scipy, random, argparse.

Data preprocessing:
Resize the input images of all the modalities to 16 × 16, and rescale them to have pixel values between 0 and 255. This is for keeping the hyperparameter selections valid.

Save the data in a .mat file that includes verctorized features in a 1024xN matrix with the name features and labels in a vector with the name Label.

A sample preprocessed dataset is available in: data/umd.mat

Note:
To keep the regularization parameters valid, please make sure that the preprocessing stage is done correctly. Also, for large datasets since the batch size will be larger, the learning rate (or the maximum number of iterations) may need to be adapted accordingly.

Demo:
Use the following script to run the included demo for the S2WCD dataset.

python dsrc_main.py --mat umd 
Running the code
Run dsrc_main.py and use

--mat  DATA to specify your dataset where DATA.mat is stored in the "data" folder.

--epoch  x to specify the maximum number of iterations.

--pretrain_step  x to specify the maximum number of pretraining iterations. (Since the demo uses a larger batch-size than the paper, it is set to have a defult maximum pretraining iteration of 1000 steps.)

--rate  x to specify the ratio of number of training samples to total number of samples.

--display_step  x to specify the frequency of reports.
