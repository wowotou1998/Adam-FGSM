import numpy as np
import torch
from torch import nn


def generate_gradient(model, image, true_label, criterion):
    # set require_grad attribute of tensor
    # important for Attack
    image.requires_grad = True
    # Forward pass the image through the model
    image.retain_grad()
    prediction_label = model(image)
    # Calculate the loss
    loss = criterion(prediction_label, true_label)
    # Zero all existing gradients
    model.zero_grad()
    # Calculate gradients of model in backward pass
    loss.backward(retain_graph=True)
    # Collect image’s gradient
    # print('image.grad', image.grad)
    image_grad = image.grad.data

    # Call FGSM Attack
    return image_grad


def norm_zoom(norm_value: float, norm_max: float, p: int) -> float:
    if p == 1:
        return norm_max / norm_value
    if p == 2:
        return (norm_max ** 2 / norm_value ** 2) ** (0.5)
    else:
        return norm_max / norm_value


def norm_clip(perturbation, norm_max, p):
    assert 0 < p < float('inf')
    B, C, H, W = perturbation.shape
    perturbation_changed = perturbation.clone().detach()
    for B_i in range(B):
        norm_value = torch.norm(torch.flatten(perturbation[B_i]), p=p).item()
        # print(norm_value)
        if norm_value > norm_max:
            # print('%.2f>%.2f' % (norm_value, norm_max))
            factor = norm_zoom(norm_value, norm_max, p)
            perturbation_changed[B_i] = perturbation[B_i] * factor
    # show_many_imgs(perturbation, 'perturbation')
    return perturbation_changed
    # norm_value = torch.maximum(norm_value, 1e-12)  # avoid divsion by zero
    # factor = torch.minimum(1, norm_max / norm_value)  # clipping -> decreasing but not increasing


# --------------------------------------

def fgsm_attack(model, clean_image, true_label, criterion, epsilon, norm_p):
    # Collect the element-wise sign of the image gradient
    image_grad = generate_gradient(model, clean_image, true_label, criterion)
    # loss = criterion(model(clean_image), true_label)
    # clean_image.requires_grad_(True)
    # loss.requires_grad_(True)
    # print('image_grad shape ',
    #       len(torch.autograd.grad(loss, clean_image, retain_graph=False, create_graph=False, allow_unused=True)))
    # image_grad = torch.autograd.grad(loss, clean_image, retain_graph=False, create_graph=False)[0]
    image_grad_sign = image_grad.sign()

    # print(clean_image.shape) batch size * channel * row * column
    # Create the perturbed clean_image by adjusting each pixel of the input clean_image
    perturbed_image = clean_image + epsilon * image_grad_sign
    # Adding clipping to maintain [0,1] range
    # torch.clamp zoom out the input tensor to the range [min,max] and return a new tensor。
    # [0-255]--->[0-1]
    perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, p=norm_p)
    # Return the perturbed clean_image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# --------------------------------------
def i_fgsm_attack(model, clean_image, true_label, criterion, epsilon, iterations, norm_p):
    perturbed_image = clean_image.clone().detach()
    epsilon_t = epsilon / iterations * 1.0
    # print('epsilon,epsilon_t_i', epsilon, epsilon_t_i)
    for t in range(iterations):
        # 这里的梯度 G 可能有 batch size个
        G = (generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)).sign()
        # epsilon_t_i 的大小要和 G 一致
        perturbed_image = perturbed_image + epsilon_t * G
        # 这里每一个迭代每一个小步骤都要进行一次clip,并且不同的范数有不同的clip准则
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, p=norm_p)
        # Return the perturbed clean_image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def mi_fgsm_attack(model, clean_image, true_label, criterion, momentum, epsilon, iterations, norm_p):
    perturbed_image = clean_image.clone().detach()
    epsilon_t = epsilon / iterations * 1.0
    momentum = 1.0
    G_total = torch.zeros(clean_image.shape, dtype=torch.float, device='cuda:0')
    # print('epsilon,epsilon_t_i', epsilon, epsilon_t_i)
    for t in range(iterations):
        # 这里的梯度 G 可能有 batch size个
        G = generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)
        # epsilon_t_i 的大小要和 G 一致
        batch, C, H, W = G.shape
        # 增加维度
        # 扩充维度
        # 变形，再恢复
        # 这一切都是为了做除法运算时能够维度相同，手工进行广播操作

        G_norm_1 = (torch.linalg.norm(G.view(batch, -1), ord=1, dim=1)).unsqueeze(1)
        G_norm_1 = G_norm_1.expand(batch, C * H * W)
        # print(G_norm_1[0])
        # print('G.shape, G_norm_1.shape ', G.shape, G_norm_1.shape)
        G_normalization = (G.view(batch, -1) / G_norm_1).view(batch, C, H, W)
        G_total = momentum * G_total + G_normalization
        perturbed_image = perturbed_image + epsilon_t * G_total.sign()
        # 这里每一个迭代每一个小步骤都要进行一次clip,并且不同的范数有不同的clip准则
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, p=norm_p)
        # Return the perturbed clean_image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def adam_fgsm_attack(model, clean_image, true_label,
                     criterion,
                     epsilon, iterations, norm_p):
    # 训练集，每个样本有三个分量
    # 初始化
    perturbed_image = clean_image.clone().detach()
    shape = clean_image.shape
    beta_1 = 0.9  # 算法作者建议的默认值
    beta_2 = 0.999  # 算法作者建议的默认值
    e = 10e-8  # 极小量,防止除0错误
    m_t = torch.zeros(shape, dtype=torch.float, device='cuda:0')
    v_t = torch.zeros(shape, dtype=torch.float, device='cuda:0')
    epsilon_t = epsilon / iterations
    for i in range(iterations):
        gradient = generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)
        m_t = beta_1 * m_t + (1 - beta_1) * gradient
        v_t = beta_2 * v_t + (1 - beta_2) * gradient.pow(2)
        m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
        v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
        v_t_hat_sqrt = v_t_hat.sqrt()  # 因为只能对标量进行开方
        perturbed_image = perturbed_image + epsilon_t * m_t_hat / (v_t_hat_sqrt + e)
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, norm_p)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def adam_fgsm_attack2(model, clean_image, true_label, criterion, epsilon, iterations, norm_p):
    perturbed_image = clean_image.clone().detach()
    epsilon_t = epsilon / iterations * 1.0
    mu = 1.0
    G_total = torch.zeros(clean_image.shape, dtype=torch.float, device='cuda:0')
    # print('epsilon,epsilon_t_i', epsilon, epsilon_t_i)
    for t in range(iterations):
        # 这里的梯度 G 可能有 batch size个
        G = generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)
        # epsilon_t_i 的大小要和 G 一致
        G_normalization = G.clone().detach()
        for B_i in range(G.shape[0]):
            G_normalization[B_i] = G[B_i] / torch.norm(torch.flatten(G[B_i]), p=1).item()
        G_total = mu * G_total + G_normalization
        perturbed_image = perturbed_image + epsilon_t * G_total.sign()
        # 这里每一个迭代每一个小步骤都要进行一次clip,并且不同的范数有不同的clip准则
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, p=norm_p)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def rmsprop_fgsm_attack2(model, clean_image, true_label,
                         criterion, beta,
                         epsilon, iterations, norm_p):
    perturbed_image = clean_image.clone().detach()
    shape = clean_image.shape
    e = 1e-9  # 极小量,防止除0错误
    G = torch.zeros(shape, dtype=torch.float, device='cuda:0')
    epsilon_t = epsilon / iterations
    for i in range(iterations):
        gradient = generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)
        G = beta * G + (1 - beta) * torch.mul(gradient, gradient)
        perturbed_image = perturbed_image + torch.mul(epsilon_t / (G.sqrt() + e), gradient)
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, norm_p)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def rmsprop_fgsm_attack(model, clean_image, true_label,
                        criterion, beta,
                        epsilon, iterations, norm_p):
    perturbed_image = clean_image.clone().detach()
    shape = clean_image.shape
    e = 1e-9  # 极小量,防止除0错误
    G = torch.zeros(shape, dtype=torch.float, device='cuda:0')
    epsilon_t = epsilon / iterations
    for i in range(iterations):
        gradient = generate_gradient(model, perturbed_image.clone().detach(), true_label, criterion)
        G = beta * G + (1 - beta) * torch.mul(gradient, gradient)
        perturbed_image = perturbed_image + torch.mul(epsilon_t / (G.sqrt() + e), gradient)
        perturbed_image = clean_image + norm_clip(perturbed_image - clean_image, epsilon, norm_p)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
