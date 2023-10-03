# Python code for creating a sample dataset for image classification

# Import necessary libraries
import torch
import torchvision
import os


def get_detection_dataset(transforms):
    # banchmark dataset
    # q: I got an error when I run this code on my local machine error code : KeyError: 'id'
    # a: I think you need to download the dataset first. You can do this by running the following commands:
    # wget http://images.cocodataset.org/zips/train2017.zip
    # wget http://images.cocodataset.org/zips/val2017.zip
    # wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    # unzip train2017.zip
    # unzip val2017.zip
    # unzip annotations_trainval2017.zip
    # You can also find the dataset here: http://cocodataset.org/#download
    dataset = torchvision.datasets.CocoDetection(
        root="./data/val2017/",
        annFile="./data/annotations/instances_val2017.json",
        transform=transforms,
        target_transform=torchvision.transforms.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.transforms.ToTensor(),
            ]
        ),
    )
    return dataset


def get_classification_dataset(transforms):
    # banchmark dataset
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transforms,
    )
    return dataset


def get_segmentation_dataset(transforms):
    # banchmark dataset

    dataset = torchvision.datasets.VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        transform=transforms,
        # target size is 256x256
        target_transform=torchvision.transforms.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )

    return dataset


if __name__ == "__main__":
    # test classifcation dataset
    # dataset = get_classification_dataset()
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0][0].shape)
    # print(dataset[0][1])

    # test detection dataset
    dataset = get_detection_dataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])

    # test segmentation dataset
    # dataset = get_segmentation_dataset()
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0][0].shape)
