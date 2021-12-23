import torch
from vit_pytorch import ViT
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset.dataset import ImageDataset,ImgAugDataset, Dataset, KerasGenerator, DatasetIterator, ImgAugMTDataset
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 81,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

from vit_pytorch.cait import CaiT

net = CaiT(
    image_size = 224,
    patch_size = 32,
    num_classes = 81,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
)



net.to(device)

#img = torch.randn(1, 3, 224, 224)

#preds = net(img) # (1, 1000)
#print(preds)
train_dataset = DatasetIterator(csv_file="data/train.csv", num_classes=82, to_categorical=False)
trainloader = ImgAugMTDataset(train_dataset, base_path="../training_caip_contest/", img_size=(224,224))
test_dataset = DatasetIterator(csv_file="data/val.csv", num_classes=82, to_categorical=False)
testloader = ImageDataset(test_dataset, base_path="../training_caip_contest/", img_size=(224,224))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

epochs=2

def preprocess(inputs):
    return inputs/255.

print_every = 10000
for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    train_losses, test_losses = [], []
    train_accuracy=0
    running_loss = 0.0
    step=0
    for data in tqdm(trainloader):
        step+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #inputs = preprocess(inputs)
        inputs = inputs.reshape((-1, 3, 224,224))
        inputs = (torch.from_numpy(inputs).float()/255.).to(device)
        labels = (torch.from_numpy(labels).long()).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logps = net.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        
        
        if step % print_every == 0:
            
            test_loss = 0
            accuracy = 0
            net.eval()
            with torch.no_grad():
                for inputs, labels in testloader:

                    inputs = inputs.reshape((-1, 3, 224, 224))
                    inputs = (torch.from_numpy(inputs).float()/255.).to(device)
                    labels = (torch.from_numpy(labels).long()).to(device)

                    logps = net.forward(inputs)
                    #labels = labels.squeeze(0)#.float()
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/print_every)
            test_losses.append(test_loss/len(test_dataset))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Train accuracy: {train_accuracy/print_every:.3f}.. "
                f"Test loss: {test_loss/len(test_dataset):.3f}.. "
                f"Test accuracy: {accuracy/len(test_dataset):.3f}")
            """
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Train accuracy: {train_accuracy/print_every:.3f}.. "
                )
            """
            running_loss = 0
            train_accuracy=0
            net.train()
        

print('Finished Training')