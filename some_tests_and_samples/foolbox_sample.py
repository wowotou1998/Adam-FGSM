#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import torch


def main():
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print("clean accuracy:%.2f " % (clean_acc))

    # apply the attack
    attack = LinfPGD()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]

    labels = labels.astype(torch.long)
    raw_advs, clipped_advs, is_adv = attack(fmodel, images, labels, epsilons=epsilons)

    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - is_adv.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print("L_inf norm <= eps:%.4f, acc:%.2f " % (eps, acc.item() * 100.))
    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    print("we can also manually check this:")
    print("robust accuracy for perturbations with")
    for eps, adv_i in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, adv_i, labels)
        print("L_inf norm <= eps:%.2f, acc:%.2f " % (eps, acc2 * 100.))
        print("perturbation sizes:")
        perturbation_sizes = (adv_i - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()
    print('end')
