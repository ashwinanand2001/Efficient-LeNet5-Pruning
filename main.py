# Ashwin Anand Introduction to Deep Learning Project

# same imports from hw5 main.py file provided in canvas resources
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sparse import PruningWeight

# imports needed for time to taken train and saving results into files for plotting
import time

# same code from hw5 main.py file provided in canvas resources
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# LeNet code imported from HW4 code submission
class LeNet(nn.Module):
    
    # defining _init(self) function
    def __init__(self):
        
        # super method 
        super(LeNet, self).__init__()
        
        # layer containing 1655 outputs and 120 inputs 
        self.fconn1 = nn.Linear(16 * 5 * 5, 120)
        
        # layer containing 120 outputs and 84 inputs   
        self.fconn2 = nn.Linear(120, 84)
        
        # layer containing 84 outputs and 10 inputs 
        self.fconn3 = nn.Linear(84, 10)
        
        # layer created with 1 inputs, 6 outputs, 3x3 kernel
        self.cnv1 = nn.Conv2d(1, 6, 3)
        
        # layer created with 6 inputs, 16 outputs, 3x3 kernel
        self.cnv2 = nn.Conv2d(6, 16, 3)

    # defining forward function
    def forward(self, x):
        
        # Applying convolutional layer cnv1 and max pooling function to x
        x = F.max_pool2d(F.relu(self.cnv1(x)), (2, 2))
        
        # # Applying convolutional layer cnv2 and max pooling function to x
        x = F.max_pool2d(F.relu(self.cnv2(x)), 2)
        
        # Reshaping x too be in 2D shape
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        
        # Applying fconn1 to x 
        x = F.relu(self.fconn1(x))
        
        # Applying fconn2 to x 
        x = F.relu(self.fconn2(x))
        
        # Applying fconn3 to x
        x = self.fconn3(x)
        
        # returning softmax to output of x 
        return  F.log_softmax(x, dim=1)

# From hw5 main.py file provided in canvas resources
def train(args, model, device, train_loader, optimizer, epoch, prune):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()        
        optimizer.step()
        prune.RecoverSparse(model)
       # if batch_idx % args.log_interval == 0:
       #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
       #         epoch, batch_idx * len(data), len(train_loader.dataset),
       #         100. * batch_idx / len(train_loader), loss.item()))

# Code mostly same as HW5 with a couple exceptions line importing results in test function       
def test(args, model, device, test_loader,results):
    
    # From hw5 main.py file provided in canvas resources
    model.eval()
    test_loss = 0
    correct = 0
    
    # array too hold reults value
    results = []
    
    # From hw5 main.py file provided in canvas resources
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # adding test loss data to results array
    results.append(test_loss)
    
# Code in this function is used to test various parameters
def main():
    # Training settings
    
    # result array
    result=[]
    
    for i in [9]:
        
        # spare_ratio calcution
        sparse_ratio = float(i/10)
        
        # same code from HW5
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=True,
                            help='For Saving the current Model')
        parser.add_argument('--ratio', default=sparse_ratio, help='how much weight will be removed')
        args = parser.parse_args()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        
        # start time
        start_time=time.time()
        
        # From hw5 main.py file provided in canvas resources
        model = LeNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        #pruning weight before Training
        prune = PruningWeight(ratio=float(args.ratio))
        prune.Init(model)    
        
        # res_arr to hold all values
        
        # From hw5 main.py file provided in canvas resources
        res_arr=[]
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, prune)
            test(args, model, device, test_loader,res_arr)
        
        # end time 
        end_time=time.time()
        
        # testing module through TestSparse function
        prune.TestSparse(model)
        
        # Epoch time printing
        print("Epoch Running Time:",(start_time-end_time)/10)
        result.append(res_arr)
        
    # saving model to data .pt files for future
    #if (args.save_model):
    #    torch.save(model.state_dict(),"mnist_cnn_1.pt")

# running main file when called
if __name__ == '__main__':
    main()
