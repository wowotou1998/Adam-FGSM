# coding = UTF-8
import numpy
import torch
import torchattacks
import torchvision.models
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from attack_method_self_defined import Adam_FGSM
from utils import save_model_results
from MNIST_models import lenet5, FC_256_128
import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model
import os
import torch.nn.functional as F
import os
from attack_models_on_datasets import generate_attack_method, load_model_args, load_dataset


def generate_transfer_rate(original_model,
                           target_model,
                           test_loader,
                           test_loader_size,
                           attack_method,
                           Epsilon,
                           Iterations,
                           Momentum):
    test_count = 0.
    epoch_num = 0
    acc_before_attack = 0.
    sample_attacked = 0
    attack_success_num = 0
    attack_success_confidence = 0.
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")

    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=test_loader_size)
    original_model.to(device)
    original_model.eval()
    target_model.to(device)
    target_model.eval()

    # Norm_p = 1
    # Epsilon = 10
    epsilon = Epsilon / 255.
    # Iterations = 10
    # Momentum = 0.9
    # print('len(test_loader)', len(test_loader))

    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad_(True)
        epoch_size = labels.shape[0]
        epoch_num += 1
        test_count += epoch_size
        pbar.update(epoch_size)
        if test_count > 10000:
            break

        _, predict = torch.max(original_model(images), 1)
        # 选择预测正确的images和labels，剔除预测不正确的images和labels
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ------------------------------
        _, predict = torch.max(target_model(images), 1)
        # 选择预测正确的images和labels，剔除预测不正确的images和labels
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ---------------------------------

        if min(images.shape) == 0:
            print('empty tensor')
            continue
        if min(labels.shape) == 0:
            print('empty tensor')
            continue

        acc_before_attack += predict_answer.sum().item()

        atk = generate_attack_method(attack_method, original_model, epsilon, Iterations, Momentum)
        images_under_attack = atk(images, labels)

        confidence, predict = torch.max(F.softmax(target_model(images_under_attack), dim=1), dim=1)
        # 记录每一个攻击方法在每一批次的成功个数
        attack_success_num += (labels != predict).sum().item()

        # 记录误分类置信度
        # 选出攻击成功的对抗样本的置信度
        # 选择攻击成功的images的confidences
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_incorrect_index = torch.flatten(torch.nonzero(labels != predict))
        # print('predict_correct_index', predict_correct_index)
        confidence_value = torch.index_select(confidence, 0, predict_incorrect_index)
        attack_success_confidence += confidence_value.sum().item()

        if epoch_num == 1:
            print('predict_correct_element_num: ', predict_correct_index.nelement())
            # show_one_image(images, 'image_after_' + attack_name)

        sample_attacked += labels.shape[0]

        # break

    attack_success_rate = (attack_success_num / sample_attacked) * 100.
    attack_success_confidence_ave = (attack_success_confidence / attack_success_num)
    print(attack_success_confidence_ave)

    print('epsilon %d/255 \nIterations %d \nMomentum %.2f' % (Epsilon, Iterations, Momentum))
    print('%s_succ_rate = %.2f%%' % (attack_method, attack_success_rate))

    pbar.close()
    print('model acc %.2f' % (acc_before_attack / test_count))
    return attack_success_rate, attack_success_confidence_ave


def generate_transfer_rate_matrix(dataset, batch_size,
                                  original_model_set, target_model_set,
                                  attack_method,
                                  Epsilon, Iterations, Momentum,
                                  work_name):
    test_loader, test_dataset_size = load_dataset(dataset, batch_size)

    matrix_shape = (len(original_model_set), len(target_model_set))
    transfer_rate_matrix = numpy.zeros(shape=matrix_shape)
    transfer_confidence_matrix = numpy.zeros(shape=matrix_shape)

    for i, model_i_name in enumerate(original_model_set):
        model_i, model_i_acc = load_model_args(model_i_name)
        for j, model_j_name in enumerate(target_model_set):
            if model_i_name != model_j_name:
                continue
            # 模型名称相同, 则不用重复加载模型
            if model_i_name == model_j_name:
                model_j, model_j_acc = model_i, model_i_acc
            else:
                model_j, model_j_acc = load_model_args(model_j_name)

            attack_success_rate, confidence = generate_transfer_rate(original_model=model_i,
                                                                     target_model=model_j,
                                                                     test_loader=test_loader,
                                                                     test_loader_size=test_dataset_size,
                                                                     attack_method=attack_method,
                                                                     Epsilon=Epsilon,
                                                                     Iterations=Iterations,
                                                                     Momentum=Momentum)
            transfer_rate_matrix[i][j] = attack_success_rate
            transfer_confidence_matrix[i][j] = confidence

    lab_result_head = ['model', 'model acc', 'attack_method', 'Epsilon', 'Iterations', 'Momentum']
    lab_result_content = [[' ', ' ', attack_method, '%d/255' % Epsilon, Iterations, Momentum], original_model_set]
    for i in range(matrix_shape[0]):
        lab_result_content.append(transfer_rate_matrix[i])
    lab_result_content.append([' '])
    for i in range(matrix_shape[0]):
        lab_result_content.append(transfer_confidence_matrix[i])
    save_model_results(work_name, lab_result_head, lab_result_content)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    batch_size = 1

    # 'FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM',
    cifar10_models = ['ResNet50', 'ResNet101']  # 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121'
    imagenet_models = ['DenseNet121_ImageNet',
                       'ResNet101_ImageNet',
                       'ResNet50_ImageNet',
                       'VGG19_ImageNet',
                       'VGG16_ImageNet',
                       ]

    generate_transfer_rate_matrix('ImageNet',
                                  batch_size,
                                  original_model_set=imagenet_models,
                                  target_model_set=imagenet_models,
                                  attack_method='Adam_FGSM',
                                  Epsilon=6,
                                  Iterations=10,
                                  Momentum=1.0,
                                  work_name='generate_transfer_rate_matrix_Adam_FGSM', )

    generate_transfer_rate_matrix('ImageNet',
                                  batch_size,
                                  original_model_set=imagenet_models,
                                  target_model_set=imagenet_models,
                                  attack_method='I_FGSM',
                                  Epsilon=6,
                                  Iterations=10,
                                  Momentum=1.0,
                                  work_name='generate_transfer_rate_matrix_I_FGSM', )

    generate_transfer_rate_matrix('ImageNet',
                                  batch_size,
                                  original_model_set=imagenet_models,
                                  target_model_set=imagenet_models,
                                  attack_method='MI_FGSM',
                                  Epsilon=6,
                                  Iterations=10,
                                  Momentum=1.0,
                                  work_name='generate_transfer_rate_matrix_MI_FGSM', )

    print("ALL WORK HAVE BEEN DONE!!!")
