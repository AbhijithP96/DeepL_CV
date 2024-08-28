import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1,2,0)
  image = image * (np.array([0.5,0.5,0.5])) + np.array([0.5,0.5,0.5])
  image = image.clip(0,1)
  return image
  
def view_data(loader):
    dataiter = iter(loader)
    images , labels = next(dataiter)
    fig = plt.figure(figsize = (25,4))

    for i in range(20):
        ax = fig.add_subplot(2,10,i+1)
        plt.imshow(im_convert(images[i]))
        ax.axis('off')
        ax.set_title([classes[labels[i].item()]])

    plt.show()

def view_predictions(loader):
    dataiter = iter(val_loader)
    images , labels = next(dataiter)
    images_ = images.to(device)
    labels = labels.to(device)
    outputs = model(images_)
    _,preds = torch.max(outputs,1)

    fig = plt.figure(figsize = (25,4))

    for i in range(20):
        ax = fig.add_subplot(2,10,i+1)
        plt.imshow(im_convert(images[i]))
        ax.axis('off')
        ax.set_title('{} ({})'.format(str(classes[preds[i].item()]),str(classes[labels[i].item()])) , color = ('green' if preds[i]==labels[i] else 'red'))

    plt.show()

def loss_plot():
    plt.plot(losses,label='training loss')
    plt.plot(val_losses,label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()

def accuracy_plot():
    plt.plot(acc,label='training accuracy')
    plt.plot(val_acc,label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy Graph')
    plt.show()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform_train = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=1,contrast=1,saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))])

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
training_dataset = datasets.ImageFolder('./ants_and_bees/train', transform=transform_train)

training_loader = torch.utils.data.DataLoader(dataset = training_dataset , batch_size = 20 , shuffle = True)

val_dataset = datasets.ImageFolder('./ants_and_bees/val', transform=transform)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset , batch_size = 20 , shuffle = False)

classes = ['ants', 'bees']

view_data(training_loader)

model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT) #alexnet

#freezing features layer of the alexnet
for param in model.features.parameters():
   param.requires_grad = False\
   
#changing the last layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features,len(classes))

model.to(device)

criterion = nn.CrossEntropyLoss() #includes softmax
optimizer = torch.optim.Adam(model.parameters() , lr = 0.0001)

epochs = 5
losses = []
acc = []

val_losses = []
val_acc = []

for e in range(epochs):
  run_e_loss = 0.0
  run_e_acc = 0.0

  run_v_loss = 0.0
  run_v_acc = 0.0
  for images,labels in training_loader:
    inputs = images.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _,preds = torch.max(outputs,1)
    run_e_loss += loss.item()
    run_e_acc += torch.sum(preds == labels.data)
  else:

    print(f'epoch : {e+1}')
    with torch.no_grad():
      for val_images,val_labels in val_loader:
        val_inputs = val_images.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs,val_labels)

        _,val_preds = torch.max(val_outputs,1)
        run_v_loss += val_loss.item()
        run_v_acc += torch.sum(val_preds == val_labels.data)

    epoch_loss = run_e_loss/len(training_loader.dataset)
    epoch_acc = run_e_acc.float()/len(training_loader.dataset)
    losses.append(epoch_loss)
    acc.append(epoch_acc.item()*100)
    print('training loss / accuracy : {:.4f} / {:.4f}'.format(epoch_loss,epoch_acc.item()*100))


    val_epoch_loss = run_v_loss/len(val_loader.dataset)
    val_epoch_acc = run_v_acc.float()/len(val_loader.dataset)
    val_losses.append(val_epoch_loss)
    val_acc.append(val_epoch_acc.item()*100)
    print('validation loss / accuracy : {:.4f} / {:.4f}'.format(val_epoch_loss,val_epoch_acc.item()*100))


loss_plot()
accuracy_plot()

view_predictions(val_loader)