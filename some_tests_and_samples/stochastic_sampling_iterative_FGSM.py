import torch
from torch import nn


def generate_gradient(model, image, true_label, criterion):
    # criterion = nn.CrossEntropyLoss()
    # set require_grad attribute of tensor
    # important for Attack
    image.requires_grad = True
    # Forward pass the image through the model
    prediction_label = model(image)
    # Calculate the loss
    loss = criterion(prediction_label, true_label)
    # Zero all existing gradients
    model.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward()
    # Collect image’s gradient
    image_grad = image.grad.data
    return image_grad


def fgsm_attack(model, image, true_label, epsilon, criterion):
    image_grad = generate_gradient(model, image, true_label, criterion)
    # Call FGSM Attack
    # Collect the element-wise sign of the image gradient
    sign_image_grad = image_grad.sign()
    # print(source_image.shape) batchsize * channel * row * column
    # Create the perturbed source_image by adjusting each pixel of the input source_image
    perturbed_image = image + epsilon * sign_image_grad
    # Adding clipping to maintain [0,1] range
    # torch.clamp zoom out the input tensor to the range [min,max] and return a new tensor。
    # [0-255]--->[0-1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed source_image
    return perturbed_image
