Setup
Dependencies:
Pytorch, numpy, scipy, random, argparse.

Step 1: Data Preparation
Goal:
Prepare my wildfire dataset so that each sample consists of a pre-disaster patch and a post-disaster patch, with a label indicating change/no-change (binary).

Data preprocessing:
Resize the input images of all the modalities to 16 × 16, and rescale them to have pixel values between 0 and 255. This is for keeping the hyperparameter selections valid.

Save the data in a .mat file that includes verctorized features in a 1024xN matrix with the name features and labels in a vector with the name Label.

A sample preprocessed dataset is available in: data/umd.mat

Step 2: Data Loader
Write a function to load pre- and post- patches and their labels.
Format:
Img_train: pre-disaster patches (training)
Img_test: pre-disaster patches (testing)
post_train: post-disaster patches (training)
post_test: post-disaster patches (testing)
train_labels, test_labels: binary labels
function signature:
def get_wildfire_data(pre_img_path, post_img_path, label_path, patch_size=16, train_rate=0.8):    # Load images, split into patches, assign labels, split train/test    return pre_train, pre_test, post_train, post_test, train_labels, test_labels

Step 3: Model Adaptation
Input: Stack pre- and post- patches as channels or treat as separate modalities.
Encoder: Feed pre-disaster patches.
Decoder: Reconstruct post-disaster patches.
Self-expression: Use the latent space of pre-disaster patches to reconstruct post-disaster patches for test samples via the self-expression matrix.

Step 4: Training
Pretrain: Autoencoder on pre-disaster patches.
Finetune: Use self-expression loss and reconstruction loss as in Deep_SRC.

Step 5: Inference & Binary Map Generation
Reconstruction Error: For each test patch, compute the error between reconstructed and actual post-disaster patch.
Thresholding: Use Otsu or quantile thresholding to convert error map to binary change map.

Step 6: Evaluation
Compare the predicted binary map to the ground-truth binary map using metrics like Precision, Recall, F1, IoU.

Note:
To keep the regularization parameters valid, make sure that the preprocessing stage is done correctly. Also, for large datasets since the batch size will be larger, the learning rate (or the maximum number of iterations) may need to be adapted accordingly.

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
