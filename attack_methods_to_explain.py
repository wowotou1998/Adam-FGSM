import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchattacks.attack import Attack

# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。
# 使用自己的类的变量来保存攻击过程中的余弦相似度和L1和L_inf值
"""
torchattacks==3.2.6没问题， 还是可以记录在迭代过程中的数据。
只有当 self._return_type == 'int' 时才会有麻烦， 因为此时会将所有数据变成int型，
"""

class I_FGSM(Attack):
    def __init__(self, model, eps=4 / 255, alpha=1 / 255, steps=0):
        super().__init__("I_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.alpha = self.eps / self.steps
        self._supported_mode = ['default', 'targeted']

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
        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

        for i in range(self.steps):
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

            # calculate norm_1
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(images - previous_images).view(B, -1)
            perturbation = torch.abs(images - ori_images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                # 缩放向量的大小并不会改变相似性， 但是可以有效的避免数值溢出的情况
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [images, norm_1_norm_inf_similarity]


class MI_FGSM(Attack):

    def __init__(self, model, eps=8 / 255, steps=5, decay=1.0):
        super().__init__("MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps
        self._supported_mode = ['default', 'targeted']

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

        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

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

            # calculate norm_1
            i = _
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(adv_images - previous_images).view(B, -1)
            perturbation = torch.abs(adv_images - images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [adv_images, norm_1_norm_inf_similarity]


class Adam_FGSM(Attack):
    def __init__(self, model, eps=8 / 255, steps=5, decay=0.9):
        super().__init__("Adam_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.alpha = self.eps / self.steps
        self.decay = decay
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        # --------------
        shape = images.shape
        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)

        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

        for i in range(self.steps):
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
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)

            direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = adv_images.detach() + self.alpha * direction

            # 扰动控制
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # -------------calculate norm_1
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(adv_images - previous_images).view(B, -1)
            perturbation = torch.abs(adv_images - images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [adv_images, norm_1_norm_inf_similarity]


'''
# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。
# 使用自己的类的变量来保存攻击过程中的余弦相似度和L1和L_inf值

class I_FGSM(Attack):
    def __init__(self, model, eps=4 / 255, alpha=1 / 255, steps=0):
        super().__init__("I_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.alpha = self.eps / self.steps
        self._supported_mode = ['default', 'targeted']

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
        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

        for i in range(self.steps):
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

            # calculate norm_1
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(images - previous_images).view(B, -1)
            perturbation = torch.abs(images - ori_images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                # 缩放向量的大小并不会改变相似性， 但是可以有效的避免数值溢出的情况
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [images, norm_1_norm_inf_similarity]


class MI_FGSM(Attack):

    def __init__(self, model, eps=8 / 255, steps=5, decay=1.0):
        super().__init__("MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps
        self._supported_mode = ['default', 'targeted']

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

        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

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

            # calculate norm_1
            i = _
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(adv_images - previous_images).view(B, -1)
            perturbation = torch.abs(adv_images - images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [adv_images, norm_1_norm_inf_similarity]


class Adam_FGSM(Attack):
    def __init__(self, model, eps=8 / 255, steps=5, decay=0.9):
        super().__init__("Adam_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.alpha = self.eps / self.steps
        self.decay = decay
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        # --------------
        shape = images.shape
        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)

        # prepare the data
        B, C, H, W = images.shape
        norm_1_norm_inf_similarity = torch.zeros((3, self.steps), dtype=torch.float).to(self.device)
        previous_grad = torch.zeros_like(images).detach().to(self.device)
        previous_images = images.clone().detach()

        for i in range(self.steps):
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
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)

            direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = adv_images.detach() + self.alpha * direction

            # 扰动控制
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # -------------calculate norm_1
            if i == 0:
                previous_grad = grad.detach().clone()
            # perturbation = torch.abs(adv_images - previous_images).view(B, -1)
            perturbation = torch.abs(adv_images - images).view(B, -1)
            norm_1_norm_inf_similarity[0][i] = torch.mean(perturbation).detach().item()
            # calculate norm_inf
            value, indices = torch.max(perturbation, dim=1)
            norm_1_norm_inf_similarity[1][i] = torch.mean(value).detach().item()
            # calculate similarity
            # 注意，这里的余弦相似度极有可能会数值溢出，因此一定要做好数值截断处理，将数据控制在合理的范围之内

            norm_1_norm_inf_similarity[2][i] = 1.0
            if i != 0:
                cosine_similarity = torch.cosine_similarity(x1=previous_grad.view(B, -1) * 100,
                                                            x2=grad.view(B, -1) * 100,
                                                            dim=1,
                                                            eps=1e-11)
                cosine_similarity = torch.clamp(cosine_similarity, min=-1., max=1.)
                norm_1_norm_inf_similarity[2][i] = torch.mean(cosine_similarity).detach().item()
            previous_grad = grad.clone().detach()
            previous_images = images.clone().detach()

        return [adv_images, norm_1_norm_inf_similarity]

'''
