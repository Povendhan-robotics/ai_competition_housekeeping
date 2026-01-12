"""Transform presets and augmentation hooks (placeholder)."""
import torchvision.transforms as T

def get_basic_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
