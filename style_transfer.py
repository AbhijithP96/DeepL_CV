import torch
import torch.optim as optim
from torchvision import transforms,models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])
    
    image = transform(image).unsqueeze(0)

    return image

def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.squeeze()
  image = image.transpose(1,2,0)
  image = image * (np.array([0.5,0.5,0.5])) + np.array([0.5,0.5,0.5])
  image = image.clip(0,1)
  return image

def plot_images(images):
    fig,ax = plt.subplots(1,len(images),figsize=(10,5))
    ax[0].imshow(im_convert(images[0]))
    ax[0].axis('off')
    ax[1].imshow(im_convert(images[1]))
    ax[1].axis('off')
    if len(images) == 3:
        ax[2].imshow(im_convert(images[2]))
        ax[2].axis('off')  
    plt.show()

def get_features(image,model):
    layers = {'0' : 'conv1_1',
              '5' : 'conv2_1',
              '10' : 'conv3_1',
              '19' : 'conv4_1',
              '21' : 'conv4_2', # content extraction
              '28' : 'conv5_1'}
    
    features = {}

    for name,layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image

    return features

def gram_matrix(tensor):
    _,d,h,w = tensor.size()
    tensor = tensor.view(d,h*w)
    gram = torch.mm(tensor,tensor.t())

    return gram

def get_video(size,images,len):
    writer = cv2.VideoWriter('./videos/transition.avi',cv2.VideoWriter.fourcc(*'XVID'),30,(size[1],size[0]))
    
    for i in range(len):
        img = np.array(images[i]*255,dtype=np.uint8)
        writer.write(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    writer.release()


# model setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

vgg = models.vgg19(weights = models.VGG19_Weights.DEFAULT).features

for param in vgg.parameters():
    param.requires_grad_(False)

vgg.to(device)

#import image
content = load_image('./Images/City.jpg').to(device)
style = load_image('./Images/StarryNight.jpg',shape=content.shape[-2:]).to(device)

# plot image
all_images = [content,style]
plot_images(all_images)

# extract fearutes
content_features = get_features(content,vgg)
style_features = get_features(style,vgg)

# apply gram matrix
style_grams = {layer : gram_matrix(style_features[layer]) for layer in style_features}

# weights initialization
style_weights ={'conv1_1' : 1,
                'conv2_1' : 0.75,
                'conv3_1' : 0.2,
                'conv4_1' : 0.2,
                'conv5_1' : 0.2}

content_weight = 1
style_weight = 1e6

target = content.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target],lr=0.003)

steps = 10000
length = 500

h,w,d = im_convert(target).shape
image_array = np.empty((length,h,w,d))
capture_frame = steps/length
counter = 0
losses = []

#optimization

for i in tqdm(range(1,steps+1)):
    target_features = get_features(target,vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _,d,h,w = target_feature.shape
        style_loss += layer_style_loss/(d*w*h)

    total_loss = content_weight*content_loss + style_weight*style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses.append(total_loss.item())

    if i%capture_frame == 0:
        image_array[counter] = im_convert(target)
        counter += 1

# convert to video
get_video(im_convert(target).shape[:2],image_array,length)

#plotting images and losses
all_images.append(target)
plot_images(all_images)

plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Total Loss')
plt.show()
