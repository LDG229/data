import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
])


def keep_image_size_open(path, size=(256, 256)):
    """
    打开图像并保持比例缩放到指定大小。
    """
    img = Image.open(path).convert('RGB')

    temp = max(img.size)

    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)

    return mask


class MyDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'mask'))
        print(f"mask 文件夹下文件的数量为: {len(self.name)}")

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]

        segment_path = os.path.join(self.path, 'mask', segment_name)
        image_path = os.path.join(self.path, 'image', segment_name)

        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)

        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset(r'E:\桌面\GPU_U-net\image_use')
    print(data[0][0].shape)
    print(data[0][1].shape)
