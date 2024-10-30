import torch
from PIL import Image
from RealESRGAN import RealESRGAN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gan_model = RealESRGAN(device, scale=4)
# gan_model.load_weights('weights/models--sberbank-ai--Real-ESRGAN/snapshots/8110204ebf8d25c031b66c26c2d1098aa831157e/RealESRGAN_x4.pth', download=True)
gan_model = RealESRGAN(device, scale=2)
gan_model.load_weights('weights/models--sberbank-ai--Real-ESRGAN/snapshots/8110204ebf8d25c031b66c26c2d1098aa831157e/RealESRGAN_x2.pth', download=True)

img_path = 'data/camera1.jpg'
image = Image.open(img_path).convert('RGB')

sr_image = gan_model.predict(image)
sr_image.save('data/sr_camera1.jpg')