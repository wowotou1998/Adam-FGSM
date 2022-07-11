import torch
import torchattacks
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from methods_to_attack import Adam_FGSM
from utils import save_model_results
from MNIST_models import lenet5, FC_256_128


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images, cmap='gray')
    plt.title(title)
    plt.show()


def load_model_args(model_name):
    import os
    assert os.path.isdir('../../Checkpoint'), 'Error: no checkpoint directory found!'
    model = models.vgg16(num_classes=10)
    if model_name == 'LeNet5':
        model = lenet5()
    if model_name == 'FC_256_128':
        model = FC_256_128()
    check_point = torch.load('../Checkpoint/%s.pth' % (model_name), map_location={'cuda:0': 'cpu'})
    model.load_state_dict(check_point['model'])
    print(model_name, 'has been load！', check_point['test_acc'])
    return model, check_point['test_acc']


def attack_one_model(model, test_loader, attack_method_set, Epsilon, Iterations, Momentum, ):
    test_count = 0
    acc_before_attack = 0
    sample_attacked = 0
    attack_success_num = torch.zeros(len(attack_method_set), dtype=torch.float)

    pbar = tqdm(total=10000)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    epsilon = Epsilon / 255.

    for data in test_loader:
        test_count += 1
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        images.requires_grad_(True)
        # labels.requires_grad_(True)
        labels = labels.to(device)
        _, predict = torch.max(model(images), 1)
        # 选择预测正确的imgs和labels，剔除预测不正确的imgs和labels
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ------------------------------
        acc_before_attack += predict_answer.sum().item()

        for idx, attack_name in enumerate(attack_method_set):
            atk = None
            acc_after_attack = 0
            if attack_name == 'FGSM':
                atk = torchattacks.FGSM(model, eps=epsilon)
            elif attack_name == 'I_FGSM':
                atk = torchattacks.BIM(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations, )
            elif attack_name == 'PGD':
                atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations,
                                       random_start=True)
            elif attack_name == 'MI_FGSM':
                atk = torchattacks.MIFGSM(model, eps=epsilon, steps=Iterations, decay=Momentum)
            elif attack_name == 'Adam_FGSM':
                atk = Adam_FGSM(model, eps=epsilon, steps=Iterations, decay=Momentum)
            else:
                pass

            imgs_under_attack = atk(images, labels)
            _, predict = torch.max(model(imgs_under_attack), 1)
            # 记录每一个攻击方法在每一批次的成功个数
            attack_success_num[idx] += (labels != predict).sum().item()
            # if test_count == 1:
            #     print('predict_correct_element_num: ', predict_correct_index.nelement())
            #     show_one_image(images, 'original')
            #     show_one_image(imgs_under_attack, 'image_after_' + attack_name)

        sample_attacked += labels.shape[0]
        pbar.update(labels.shape[0])
        # to quickly test
        # break

    attack_succ_rate = (attack_success_num / sample_attacked) * 100.0

    print('epsilon %d/255 \nIterations %d \nMomentum %.2f' % (Epsilon, Iterations, Momentum))
    for i in range(len(attack_method_set)):
        print('%s_succ_rate = %.2f%%' % (attack_method_set[i], attack_succ_rate[i]))

    pbar.close()
    return attack_succ_rate


def attack_many_model(model_name_set, attack_method_set, batch_size, work_name, Epsilon_set, Iterations_set, Momentum):
    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    lab_result_head = ['model', 'model acc', 'Epsilon', 'Iterations', 'Momentum'] + attack_method_set + ['MNIST']
    lab_result_content = []

    for mode_name in model_name_set:
        model, model_acc = load_model_args(mode_name)
        for Epsilon in Epsilon_set:
            for Iterations in Iterations_set:
                # FGSM, I_FGSM, PGD, MI_FGSM, Adam_FGSM_acc, Adam_FGSM2_acc
                attack_method_succ_list = attack_one_model(model=model,
                                                           test_loader=test_loader,
                                                           attack_method_set=attack_method_set,
                                                           Epsilon=Epsilon,
                                                           Iterations=Iterations,
                                                           Momentum=Momentum)
                tmp_list = [mode_name, model_acc, '%d/255' % Epsilon, Iterations,
                            Momentum] + attack_method_succ_list.numpy().tolist()
                lab_result_content.append(tmp_list)
    save_model_results(work_name, lab_result_head, lab_result_content)


if __name__ == '__main__':
    batch_size = 200
    #
    attack_method_set = ['FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM']
    model_name_set = ['FC_256_128', 'LeNet5']
    attack_many_model(model_name_set,
                      attack_method_set,
                      batch_size,
                      work_name='MNIST_performance',
                      Epsilon_set=[40, 42, 50],
                      Iterations_set=[6, 8, 16],
                      Momentum=1.0)

    # attack_many_model(model_name_set, attack_method_set,
    #                   batch_size,
    #                   work_name='iterations_compare',
    #                   Epsilon_set=[5],
    #                   Iterations_set=[1],
    #                   Momentum=0.9)
    print("ALL WORK HAVE BEEN DONE!!!")
