import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchattacks.attack import Attack


# from torchattacks import MIFGSM, BIM


# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。

class I_FGSM(Attack):
    def __init__(self, model, eps=4 / 255, alpha=1 / 255, steps=0):
        super().__init__("I_FGSM", model)
        # extra variables
        self.L = []

        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
        self._supported_mode = ['default', 'targeted']

    def get_iterations(self):
        temp_L = self.L.copy()
        self.L.clear()
        return temp_L

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        # extra variables
        self.L.append(images.clone().detach())

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images \
                + (adv_images < a).float() * a
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) \
                + (b <= ori_images + self.eps).float() * b
            images = torch.clamp(c, max=1).detach()

            self.L.append(images.clone().detach())

        return images


class MI_FGSM(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0):
        super().__init__("MI_FGSM", model)
        self.L = []

        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']

    def get_iterations(self):
        temp_L = self.L.copy()
        self.L.clear()
        return temp_L

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        # extra data
        self.L.append(adv_images.clone().detach())

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            self.L.append(adv_images.clone().detach())

        return adv_images


class Adam_FGSM(Attack):
    # Adam_FGSM_V3 ,torchattacks ==3.2.6
    def __init__(self, model, eps=4 / 255, steps=0, decay=0.9):
        super().__init__("Adam_FGSM", model)
        self.L = []
        self.eps = eps
        self.steps = steps
        self.alpha = self.eps / self.steps
        self.decay = decay
        self._supported_mode = ['default', 'targeted']

    def get_iterations(self):
        temp_L = self.L.copy()
        self.L.clear()
        return temp_L

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        self.L.append(images.clone().detach())

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        # --------------
        shape = images.shape
        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)

        for _ in range(self.steps):
            i = _

            images.requires_grad = True
            outputs = self.model(images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            # Update adversarial images
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)
            # d = \frac{d}{norm_1(d)}
            direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = images + self.alpha * direction
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images \
                + (adv_images < a).float() * a
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) \
                + (b <= ori_images + self.eps).float() * b
            images = torch.clamp(c, max=1).detach()

            self.L.append(images.clone().detach())

        return images
