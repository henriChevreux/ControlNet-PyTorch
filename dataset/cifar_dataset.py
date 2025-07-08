import glob
import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class CifarDataset(Dataset):
    r"""
    CIFAR dataset class that mirrors the structure of MnistDataset.
    Downloads CIFAR-10 data automatically and provides the same interface
    as the MNIST dataset class.
    """
    
    def __init__(self, split, im_path, im_ext='png', im_size=32, return_hints=False, download=False):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder where images will be stored
        :param im_ext: image extension. assumes all images would be this type.
        :param im_size: image size (CIFAR-10 is 32x32)
        :param return_hints: whether to return canny edge hints
        :param download: whether to download CIFAR-10 data automatically
        """
        self.split = split
        self.im_ext = im_ext
        self.im_size = im_size
        self.return_hints = return_hints
        self.im_path = im_path
        
        # Download and convert CIFAR-10 data if needed
        if download:
            self._download_and_convert_cifar()
        
        self.images = self.load_images(im_path)
    
    def _download_and_convert_cifar(self):
        r"""
        Downloads CIFAR-10 dataset and converts it to image files
        organized in the same structure as MNIST dataset
        """
        # Create directories for train and test
        train_dir = os.path.join(self.im_path, 'train')
        test_dir = os.path.join(self.im_path, 'test')
        
        # Check if data already exists
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            if len(os.listdir(train_dir)) > 0 and len(os.listdir(test_dir)) > 0:
                print(f"CIFAR-10 data already exists in {self.im_path}")
                return
        
        # Download CIFAR-10 dataset
        print("Downloading CIFAR-10 dataset...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./temp_cifar', train=True, download=True, transform=None
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./temp_cifar', train=False, download=True, transform=None
        )
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Convert and save train images
        self._convert_and_save_images(train_dataset, train_dir, class_names, 'train')
        
        # Convert and save test images
        self._convert_and_save_images(test_dataset, test_dir, class_names, 'test')
        
        # Clean up temporary directory
        import shutil
        if os.path.exists('./temp_cifar'):
            shutil.rmtree('./temp_cifar')
        
        print(f"CIFAR-10 dataset converted and saved to {self.im_path}")
    
    def _convert_and_save_images(self, dataset, save_dir, class_names, split_name):
        r"""
        Converts PIL images from torchvision dataset to files
        :param dataset: torchvision CIFAR dataset
        :param save_dir: directory to save images
        :param class_names: list of class names
        :param split_name: 'train' or 'test'
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create class directories
        for class_name in class_names:
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        print(f"Converting {split_name} images...")
        for idx, (image, label) in enumerate(tqdm(dataset)):
            class_name = class_names[label]
            filename = f"{idx:05d}.{self.im_ext}"
            filepath = os.path.join(save_dir, class_name, filename)
            
            # Save image
            image.save(filepath)
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path: root path containing train/test folders
        :return: list of image paths
        """
        split_path = os.path.join(im_path, self.split)
        assert os.path.exists(split_path), "images path {} does not exist".format(split_path)
        
        ims = []
        for d_name in tqdm(os.listdir(split_path)):
            class_path = os.path.join(split_path, d_name)
            if os.path.isdir(class_path):
                for fname in glob.glob(os.path.join(class_path, '*.{}'.format(self.im_ext))):
                    ims.append(fname)
        
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        
        # Ensure image is RGB (CIFAR-10 images should be RGB)
        if im.mode != 'RGB':
            im = im.convert('RGB')
            
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        
        if self.return_hints:
            canny_image = Image.open(self.images[index])
            if canny_image.mode != 'RGB':
                canny_image = canny_image.convert('RGB')
            
            canny_image = np.array(canny_image)
            
            # Convert to grayscale for Canny edge detection
            canny_gray = cv2.cvtColor(canny_image, cv2.COLOR_RGB2GRAY)
            canny_edges = cv2.Canny(canny_gray, 100, 200)
            
            # Convert back to 3 channels
            canny_edges = canny_edges[:, :, None]
            canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
            
            canny_image_tensor = torchvision.transforms.ToTensor()(canny_edges)
            
            return im_tensor, canny_image_tensor
        else:
            return im_tensor


# Example usage:
if __name__ == "__main__":
    # Create dataset - this will automatically download CIFAR-10
    train_dataset = CifarDataset(
        split='train',
        im_path='./data/cifar10_data',
        return_hints=False,
        download=True
    )
    
    test_dataset = CifarDataset(
        split='test', 
        im_path='./data/cifar10_data',
        return_hints=False,
        download=False  # Already downloaded
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample tensor shape: {sample.shape}")
    print(f"Sample tensor range: {sample.min()} to {sample.max()}")