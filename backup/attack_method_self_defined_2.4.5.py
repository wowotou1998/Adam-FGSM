import torch
import torch.nn as nn
from torchattacks.attack import Attack


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

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        # --------------
        shape = images.shape
        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)
            # d = \frac{d}{norm_1(d)}
            direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = adv_images.detach() - self.alpha * direction

            # 扰动控制
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class Adam_FGSM_incomplete(Attack):
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

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        # --------------
        shape = images.shape
        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)
            # d = \frac{d}{norm_1(d)}
            # direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = adv_images.detach() - self.alpha * direction

            # 扰动控制
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class Adam_FGSM2(Attack):
    r"""
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 8/255)
        decay (float): momentum factor. (DEFAULT: 1.0)
        steps (int): number of iterations. (DEFAULT: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model, eps=8 / 255, steps=5, decay=0.9):
        super(Adam_FGSM2, self).__init__("Adam_FGSM2", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        shape = images.shape
        beta_1 = 1 - (1 / self.steps)  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        e = 10e-8  # 极小量,防止除0错误
        m_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        v_t = torch.zeros(shape, dtype=torch.float).to(self.device)
        batch, C, H, W = images.shape
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * grad.pow(2)
            m_t_hat = m_t / (1 - (beta_1 ** (i + 1)))
            v_t_hat = v_t / (1 - (beta_2 ** (i + 1)))
            v_t_hat_sqrt = v_t_hat.sqrt()
            direction = m_t_hat / (v_t_hat_sqrt + e)

            direction = direction / torch.mean(torch.abs(direction), dim=(1, 2, 3), keepdim=True)

            adv_images = adv_images.detach() - self.alpha * direction

            # 扰动控制
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # 总体图像值控制
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
