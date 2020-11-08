import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

#Function for training a model. Arguments:
#model: A neural network of choise (alexnet and resnet18 in this example)
#data_loaders: a formatted dataset
#criterion: a loss function (CrossEntropyLoss)
#optimizer: an optimizer (Adam)
def train_model(model, data_loaders, criterion, optimizer, epochs=1):
    for epoch in tqdm(range(epochs)):
        #print('Epoch %d / %d' % (epoch, epochs-1))
        #First we train, then we evaluate
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct = 0

            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, prediction = torch.max(outputs, 1)
                    #The lines above will decide what the picture showes
                    #if in train mode it will evaluate and train the model: (.step())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(prediction == labels.data)
            
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_accuracy = correct.double() / len(data_loaders[phase].dataset)
            #print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))
    print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

if __name__ == '__main__':
    root_dir = 'hymenoptera_data/'

    image_transforms = {
        'train': torchvision.transforms.Compose([torchvision.transforms.RandomRotation((-270, 270)),
                 torchvision.transforms.Resize((224, 224)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        'val': torchvision.transforms.Compose([torchvision.transforms.RandomRotation((-270, 270)),
                 torchvision.transforms.Resize((224, 224)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    
    }

    #Generating the dataset to be loaded into the model
    dataset = {k: torchvision.datasets.ImageFolder(os.path.join(root_dir, k),
                    image_transforms[k]) for k in ['train', 'val']}

    data_loader = {k: torch.utils.data.DataLoader(dataset[k], batch_size=2,
                    shuffle=True, num_workers=4) for k in ['train', 'val']}

    #Checks if you can compute it fast or slooooow:
    #enables cuda if it can find a supported GPU on your system.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Loading Alexnet from torchvision
    #Pretrained = True so their trained weigths are enabled
    model = torchvision.models.alexnet(pretrained=True)

    print(model.eval())

    #Then we freeze the parameteres for feature extraction, because we wnat to use them
    for param in model.parameters():
        param.requires_grad = False
    #and unfreeze the last fully-conected layers (the classifier) as we want to train them our selves.
    for param in model.classifier.parameters():
        param.requires_grad = True 

    #Placing the model onto the chosen device
    model.to(device)

    #Choosing the loss function and our optimizer
    criterion = nn.CrossEntropyLoss()
    opitmizer = optim.Adam(model.parameters(), lr=0.001)

    #Printing parameter that are trained; for AlexNet this should only print the classifier.x.yyyy
    #**just for lols**
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print('\t', name)

    #Unleashing the beast
    train_model(model, data_loader, criterion, opitmizer)

    #The dataset consist of two classes, ants and bees, with ca. 120 pictures of each for trainning 
    #and 75 each for evaluating our model.
    #Running 25 EPOCHS on my GTX 1070 takes 2 minutes and results in an accuracy of  87.6%

    #Code for printing activations
    image = Image.open("hymenoptera_data/train/ants/258217966_d9d90d18d3.jpg")

    image_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    feature = model.features[0:1]

    image =  image_transform(image)[None]
    image = image.to(device)
    activation = feature(image).cpu()

    _, axs = plt.subplots(8, 8)

    axs = axs.ravel()

    for i in range(len(activation[0])):
        im_f = torch_image_to_numpy(activation[0][i])*255
        img = Image.fromarray(im_f.astype(np.uint8))
        axs[i].imshow(img)
    #plt.figure()
    #im_f = torch_image_to_numpy(activation[0][3])*255
    #img = Image.fromarray(im_f.astype(np.uint8))
    #plt.imshow(img)
    plt.savefig("test-conv2d.png")
    #Compare "test-conv2d.png" to "test.jpg" 