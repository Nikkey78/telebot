import torch
import torchvision.models as models

        # сохранить часть фитчей предобученной модели VGG19 в отдельный файл
        # для возможности работы на Heroku  (ограничены в ресурсах)  
cnn = models.vgg19(pretrained=True).features.to('cpu').eval()
for param in cnn.parameters():
    param.requires_grad_(False)

cnn = cnn[:6]
torch.save(cnn.state_dict(), 'trim_vgg.pth')
