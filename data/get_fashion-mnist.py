import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Download the training set
    train = torchvision.datasets.FashionMNIST(
        root = '.',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()                                 
        ])
        )
    
    # Download the testing set
    test = torchvision.datasets.FashionMNIST(
        root = '.',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()                                 
        ])
        )