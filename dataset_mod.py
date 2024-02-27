#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:05:05 2021

@author: laurent
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision


def get_datasets(selected_digits=[3, 8]):
    """
    Return PyTorch datasets for MNIST with only specified digits.
    """

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transform)

    # Filter train dataset
    train_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in selected_digits]
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    # Filter test dataset
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in selected_digits]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    return train_dataset, test_dataset

def getSets_mod(filteredClass = None, removeFiltered = True) :
	"""
	Return a torch dataset
	
	selectedDigits: 
	
	"""
	train_dataset, test_dataset = get_datasets(selected_digits=[3, 8])
	
	if filteredClass is not None :
		# Do if filteredClass is specified	
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
	
		#train_loader = torch.utils.data.DataLoader(train, batch_size=len(train))
		train_labels = next(iter(train_loader))[1].squeeze()
		
		#test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
		test_labels = next(iter(test_loader))[1].squeeze()
		
		if removeFiltered : 
			# Creates the unseen class
			trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels != filteredClass).squeeze()
		else :
			trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels == filteredClass).squeeze()
		
		train = torch.utils.data.Subset(train_dataset, trainIndices)
		test = torch.utils.data.Subset(test_dataset, testIndices)
	
	return train, test

if __name__ == "__main__" :
	
	#test getSets function
	train, test = getSets_mod(filteredClass = 3, removeFiltered = False)
	
	test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
	
	images, labels = next(iter(test_loader))
	
	print(images.shape)
	print(torch.unique(labels.squeeze()))