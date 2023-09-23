#!/usr/bin/python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.cluster import KMeans


def display_images(images: list[str], columns: int=5, width: int=20, height: int=8):
    """Display images in a subplot.
    Based on https://keestalkstech.com/2020/05/plotting-a-grid-of-pil-images-in-jupyter/

    Args:
        images: List of paths to images.
        columns: Number of columns in the display grid.
        width: width in inches of the grid.
        height: height in inches of the grid.

    """
    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.title(image.filename, fontsize=9)
        plt.axis('off')
        plt.imshow(image)
    plt.show()


class Identity(torch.nn.Module):
    """Layer that does nothing (input = output).
    Based on https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2"""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class DBAL:
    """Active learning query strategy based on the paper 'Diverse mini-batch Active Learning' from Fedor Zhdanov.
    Paper: https://doi.org/10.48550/arXiv.1901.05954).
    
    Args:
        N: maximum number of queried samples (int) or None which uses all images.
        k: number of queried samples.
        beta: pre-filter factor. k samples will be selected out of beta * k images.
        image_size: image will be resized to image_size as input to the Neural Network.
        use_weighted_kmeans: if True, use weighted kmeans; if False, use normal kmeans.
        
    """
    def __init__(self, N=None, k: int=20, beta: int=10, image_size: int=224, use_weighted_kmeans: bool=True):
        self.k = int(k)
        self.beta = beta 
        self.N = N 
        self.image_size = image_size # ImageNet is 224 x 224
        self.use_weighted_kmeans = use_weighted_kmeans
        assert self.beta*self.k<=self.N, "k * beta has to be smaller or equal to N"
        assert self.beta>=1, "beta must be greater or equal than 1"
        assert self.k>=1, "k must be greater or equal than 1"


        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) #  load a pre-trained deep learning model
        self.resnet.eval() # set to evaluation mode

        self.resnet_features = resnet18(weights=ResNet18_Weights.DEFAULT) #  load a pre-trained model as feature extractor
        self.resnet_features.fc = Identity() # remove last fully connected layer
        self.resnet_features.eval()

        # perform image preprocessing according to ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),                
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])])
    
    def query(self, image_path: str) -> list[str]:
        """Run DBAL active learning strategy.

        Args:
            image_path: Path to a folder with images.

        Returns:
            List of selected images in image_path according to the query strategy.

        """
        files = os.listdir(image_path)
        if (self.N == None):
            N = len(files)
        else:
            N = self.N
        files = files[0:N]
        margins = []
        X = np.zeros((N, 512), dtype=np.float32)
        for i, file in enumerate(files):
            # iterate through the unlabeled pool of data
            img = Image.open(os.path.join(image_path, file))
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)

            predictions = self.resnet(img) # model output
            features = self.resnet_features(img) # feature vector
            X[i,:] = features.detach().numpy() # feature matrix
            probabilities = torch.nn.functional.softmax(predictions, dim=1)[0] # convert predictions to class probabilities
            P = sorted(probabilities.tolist()) # a sorted list of class probabilities
            margins.append(1 - (P[-1] - P[-2]))


        # Prefilter to top beta*k informative examples
        indices = np.argsort(margins)
        keep_idx = []
        for i in range(-1, -self.k*self.beta-1, -1):
            keep_idx.append(indices[i])
        X = X[keep_idx, :]
        margins = [margins[i] for i in keep_idx]
        files = [files[i] for i in keep_idx]

        if (self.beta == 1):
            # perform only margin based sampling
            images = []
            for i in range(self.k):
                im = Image.open(os.path.join(image_path, files[i]))
                im.filename = str(files[i])
                images.append(im)
            return images
        else:
            # cluster beta*k examples to k clusters with weighted K-means
            kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=1, 
                            tol=1e-4, verbose=1).fit(X, sample_weight=margins if self.use_weighted_kmeans==True else None)
            distances = kmeans.transform(X)

            # select k different examples closest to the cluster centers
            images = []
            for column in range(self.k):
                idx = np.argmin(distances[:, column]) # find closest sample to the current cluster
                im = Image.open(os.path.join(image_path, files[idx]))
                im.filename = str(files[idx])
                images.append(im)
            return images


if __name__ == '__main__':
    images = DBAL(N=200, k=10, beta=10,
                           use_weighted_kmeans=True).query(image_path="./Pet-Dataset/")
    display_images(images)