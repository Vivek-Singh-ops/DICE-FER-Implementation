DICE-FER: Decoupling Identity Confounders for Enhanced Facial Expression Recognition

This repository contains an unofficial PyTorch implementation of the architecture proposed in the paper "Decoupling Identity Confounders for Enhanced Facial Expression Recognition: An Information-Theoretic Approach" (EE-656 Course Project).

 Project Overview

Facial Expression Recognition (FER) models often inadvertently learn a subject's facial structure (identity) rather than the actual emotion, causing them to fail when tested on new individuals. This project implements a latent-space disentanglement framework that uses Mutual Information (MI) and Adversarial Training to mathematically decouple "Identity" from "Expression."

 Architecture

The model consists of four core components:

Dual Encoders (ResNet-18): Modified to accept 112x112 grayscale images. They extract separate 64-dimensional vectors for Expression ($E$) and Identity ($I$).

Mutual Information Neural Estimator (MINE): Uses Donsker-Varadhan representation across both Global and Local (spatial) Statistics Networks to maximize shared information between the input image and the Expression vector.

Adversarial Discriminator: A 3-layer MLP that attempts to classify whether an ($E, I$) pair comes from the same image or a shuffled batch, forcing the networks to minimize mutual information between identity and expression.

FER Classifier: A 2-layer MLP that predicts one of 7 emotion classes from the purified Expression vector.

Results & Disentanglement Proof

The model was trained on the CK+ (Extended Cohn-Kanade) dataset.

Final Classification Accuracy: ~78.70% (Peak ~87.40%)

Adversarial Convergence: The discriminator loss stabilized near 0.65, indicating successful adversarial confusion and successful feature disentanglement.

Qualitative Evaluation (Image Retrieval)

To prove latent space disentanglement, Cosine Similarity was used to retrieve the nearest neighbors for a query image.

Top Row (Expression Retrieval): The model successfully retrieves different people exhibiting the exact same emotion.

Bottom Row (Identity Retrieval): The model successfully retrieves the exact same person exhibiting completely different emotions.

(Upload your generated plot image to the repository and replace disentanglement_proof.png below with your filename)


 Limitations & Deviations from Original Paper

Due to compute constraints and to facilitate rapid prototyping, the following modifications were made compared to the original research paper:

Pre-training: The ResNet-18 encoders were initialized from scratch rather than using CASIA-WebFace pre-trained weights.

Data Augmentation: Only basic resizing and tensor conversions were applied, bypassing the 14 advanced rotations and flips used by the authors.

Evaluation Protocol: The model was evaluated on a standard train/eval split rather than 10-fold cross-validation.

Retrieval K-Value: The retrieval top_k was adjusted to 4 (from 6) to account for the low number of samples per identity in the CK+ dataset, ensuring the identity retrieval did not have to fall back on nearest-neighbor "strangers."

 How to Run

Download the CK+ dataset and place it in the CK_Dataset/CK+48 directory.

Run the files in the following order:

dataloader.py (Prepares the paired identity/expression batches)

networks.py (Initializes the PyTorch modules)

train.py (Executes the adversarial training loop)

evaluate.py (Calculates final accuracy and plots the cosine similarity grid)

 Tech Stack

Python 3.10+

PyTorch & Torchvision

OpenCV / PIL

Matplotlib