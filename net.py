import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import time
import os
from tqdm import tqdm
from loguru import logger


        # декоратор для анализа длительности работы функции
def time_counter(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        logger.debug(f'elapsed time: {(end-start)//60:.0f}m {(end-start)%60:.0f}s')
        return result
    return wrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.debug(device.type + ': ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')

        # если используем ресурсы с GPU, можно использовать 8 первых слоев VGG19
# vgg = models.vgg19(pretrained=True)
# cnn = vgg.features.to(device).eval()
# for param in cnn.parameters():
#     param.requires_grad_(False)
# cnn = cnn[:8]


        # загружаем подготовленный файл фитчей VGG19, если используем Heroku (ограничены в ресурсах) 
cnn = models.vgg19().features.to(device).eval()
cnn = cnn[:6]
cnn.load_state_dict(torch.load('trim_vgg.pth'))


        # возвращаем тензор torch из изображения PIL
def get_tensor_from_image(img, imsize):
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])  
    image = Image.open(img)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


        # преобразуем тензор torch в изображение PIL
def get_image_from_tensor(tensor, imsize, scale, ratio):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)    
    image = unloader(image)
    image = image.resize((int(imsize*scale*ratio), int(imsize*scale)))
    return image


        # модуль потерь контента
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


        #  матрица Грама
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)  
    G = torch.mm(features, features.t()) 
    return G.div(a * b * c * d)


        # модуль потерь стиля
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


        # функция создания модели для style transfer
def get_style_model_and_losses(cnn, style_img, content_img):

            # если используем ресурсы с GPU, можно использовать более "жирную" модель для более качественного результата
    # content_layers = ['conv_5']
    # style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

            # добавляем данные слои, если используем расчеты на Heroku (ограничены в ресурсах)  
    content_layers = ['conv_3']
    style_layers = ['conv_1', 'conv_2', 'conv_3']

    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    model = nn.Sequential(normalization)  

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses


        # функция реализации модели 
def run_style_transfer(cnn, content_img, style_img, input_img,
                       num_steps=10,                            # num steps ~100 и более, если используем GPU
                       style_weight=1e6, content_weight=1):     # коэффициенты для вклада контента или стиля в результат можно варьировать

    logger.debug('Building the style transfer model..')

    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_(True)], lr=1)

    for i in tqdm(range(num_steps)):
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            return loss

        optimizer.step(closure)

        logger.debug('optimizer step: ' + str(i))

    input_img.data.clamp_(0, 1)
    return input_img


@time_counter
def draw(content, style):
    image = Image.open(content)    
    
    height = image.size[0] 
    width = image.size[1] 
    
    # задаем размер для тензора, чем больше, тем больше нужно памяти и времени вычисления,
    # но конечный результат качественнее

    # imsize = 380          # такой размер работает на GTX 950 2Gb
    
    imsize = 220            # размерность тензора для Heroku

    ratio = height/width    # соотношение сторон исходной картинки
    scale = height/imsize   # масшабный коэффициент   

    content_img = get_tensor_from_image(content, imsize)
    style_img   = get_tensor_from_image(style, imsize)
    input_img   = content_img.clone()

    logger.debug('start transfer...')
    output = run_style_transfer(cnn, content_img, style_img, input_img)
    logger.debug('end transfer...')

    image = get_image_from_tensor(output, imsize, scale, ratio)
    return image

