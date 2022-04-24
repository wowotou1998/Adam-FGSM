import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import save_model_results
from attack_methods_to_explain import I_FGSM, MI_FGSM, Adam_FGSM
from attack_models_on_datasets import load_model_args, load_dataset


# from torchattacks import BIM

# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。


def attack_one_model(model, test_loader, attack_name, Epsilon, Iterations, Momentum):
    test_count = 0
    sample_attacked_num = 0
    attack_success_num = 0
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=10000)
    model.to(device)
    model.eval()
    norm_1_norm_inf_similarity_total = torch.zeros((3, Iterations), dtype=torch.float).to(device)
    print('len(test_loader)', len(test_loader))
    for data in test_loader:
        test_count += 1
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        _, predict = torch.max(model(images), 1)
        # 选择预测正确的images和labels，剔除预测不正确的images和labels
        predict_answer = (labels == predict)
        '''
        torch.nonzero()函数得到的返回值是非零元素的下标，所以他的input有n维，那他的输出会多一维，即输出有（n+1)维
        即使获取不到内容，也会返回一个无内容的tensor
        '''
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        '''
        torch.index_select(input, dim, index, out=None) 函数的三个关键参数，函数参数有：
            input(Tensor) - 需要进行索引操作的输入张量；
            dim(int) - 需要对输入张量进行索引的维度；
            index(LongTensor) - 包含索引号的 1D 张量；
            out(Tensor, optional) - 指定输出的张量。
        '''
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ------------------------------
        atk = None
        if attack_name == 'I_FGSM':
            atk = I_FGSM(model, eps=Epsilon, alpha=Epsilon / Iterations,
                         steps=Iterations, )
        elif attack_name == 'MI_FGSM':
            atk = MI_FGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)
        elif attack_name == 'Adam_FGSM':
            atk = Adam_FGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)
        else:
            pass
        # --------------
        images_under_attack_extra_data = atk(images, labels)
        images_under_attack = images_under_attack_extra_data[0]
        # norm_1 norm_inf,similarity
        extra_data = images_under_attack_extra_data[1]
        norm_1_norm_inf_similarity_total += extra_data
        _, predict = torch.max(model(images_under_attack), 1)
        # 记录每一个攻击方法在每一批次的成功个数
        attack_success_num += (labels != predict).sum().item()
        if test_count == 1:
            print('predict_correct_element_num: ', predict_correct_index.nelement())

        sample_attacked_num += labels.shape[0]
        pbar.update(labels.shape[0])

        # to quickly test
        # break

    attack_succ_rate = (attack_success_num / sample_attacked_num) * 100.0
    print('attack_success_num,sample_attacked_num', attack_success_num, sample_attacked_num)
    norm_1_norm_inf_similarity_total = norm_1_norm_inf_similarity_total / len(test_loader)
    pbar.close()
    return attack_succ_rate, norm_1_norm_inf_similarity_total


def attack_many_model(model_name_set, attack_method_set, dataset_name, batch_size, work_name, Epsilon_set,
                      Iterations_set, Momentum):
    test_loader, _ = load_dataset(dataset_name, batch_size)

    lab_result_head = ['model', 'model acc', 'Epsilon', 'Momentum', 'attack_name', 'data_type']
    lab_result_content = []

    for mode_name in model_name_set:
        model, model_acc = load_model_args(mode_name)
        for Epsilon in Epsilon_set:
            for Iterations in Iterations_set:
                # FGSM, I_FGSM, PGD, MI_FGSM, Adam_FGSM_acc
                for idx, attack_name in enumerate(attack_method_set):
                    attack_succ, extra_data = attack_one_model(model=model,
                                                               test_loader=test_loader,
                                                               attack_name=attack_name,
                                                               Epsilon=Epsilon / 255.,
                                                               Iterations=Iterations,
                                                               Momentum=Momentum)

                    print("%s, epsilon %.4f,iteration %d attack success rate %.2f" % (
                        attack_name,
                        Epsilon / 255.,
                        Iterations,
                        attack_succ))
                    # extra_data size = attack_methods * data_type * iteration
                    for i, data_type in enumerate(['L1_norm', 'Linf_norm', 'similarity']):
                        tmp_list = [mode_name, model_acc, '%d/255' % Epsilon, Momentum, ] + \
                                   [attack_name] + [data_type] + \
                                   extra_data[i].detach().cpu().numpy().tolist()
                        lab_result_content.append(tmp_list)

    save_model_results(work_name, lab_result_head, lab_result_content)


if __name__ == '__main__':
    # explain why Adam-FGSM outperform other attack methods
    batch_size = 700
    attack_method_set = ['I_FGSM', 'MI_FGSM', 'Adam_FGSM']  # 'I_FGSM',  'MI_FGSM' ,'Adam_FGSM'
    # model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']
    model_name_set = ['LeNet5']

    attack_many_model(model_name_set, attack_method_set,
                      'MNIST',
                      batch_size,
                      work_name='explain',
                      Epsilon_set=[5],
                      Iterations_set=[15],
                      Momentum=1.0)
    # attack_many_model(model_name_set, attack_method_set,
    #                   'ImageNet',
    #                   batch_size,
    #                   work_name='explain',
    #                   Epsilon_set=[5],
    #                   Iterations_set=[15],
    #                   Momentum=1.0)
    print()
    print("----ALL WORK HAVE BEEN DONE!!!----")
