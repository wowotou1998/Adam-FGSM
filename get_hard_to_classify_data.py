import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import save_model_results
from torchattacks import BIM, MIFGSM
from attack_method_self_defined import Adam_FGSM
from attack_models_on_datasets import load_model_args, load_dataset


# from torchattacks import BIM

# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。


def get_hard_to_classify(model, test_loader, Epsilon, Iterations, Momentum):
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    small_dataset = []
    test_count = 0
    Adam_FGSM_success_sum = 0
    bar = tqdm(total=10000)
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
        # predict_answer为一维向量，大小为batch_size
        predict_answer = (labels == predict)
        # torch.nonzero会返回一个二维矩阵，大小为（nozero的个数）*（1）
        no_zero_predict_answer = torch.nonzero(predict_answer)
        # 我们要确保 predict_correct_index 是一个一维向量,因此使用flatten,其中的元素内容为下标
        predict_correct_index = torch.flatten(no_zero_predict_answer)
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ------------------------------

        atk1 = BIM(model, eps=Epsilon, alpha=Epsilon / Iterations,
                   steps=Iterations, )

        atk2 = MIFGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)

        atk3 = Adam_FGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)

        # 选择I-FGSM攻击不成功的样本集合A_1
        images_under_attack_extra_data = atk1(images, labels)
        images_under_attack = images_under_attack_extra_data[0]
        _, predict_attacked_images = torch.max(model(images_under_attack), 1)

        # attack_success_num += (labels == predict_attacked_images).sum().item()
        attack_no_success = (labels == predict_attacked_images)
        I_FGSM_no_success = attack_no_success.sum().item()
        # 如果攻击之后原图片的标签还没改变，那就证明攻击失败了
        attack_no_success_index = torch.flatten(torch.nonzero(attack_no_success))
        images_a_1 = torch.index_select(images, 0, attack_no_success_index)
        labels_a_1 = torch.index_select(labels, 0, attack_no_success_index)

        # 选择MI-FGSM对于集合A攻击未成功的样本集合A_2
        images_under_attack_extra_data = atk2(images_a_1, labels_a_1)
        images_under_attack = images_under_attack_extra_data[0]

        _, predict_attacked_images = torch.max(model(images_under_attack), 1)
        attack_no_success = (labels_a_1 == predict_attacked_images)
        MI_FGSM_no_success = attack_no_success.sum().item()
        attack_no_success_index = torch.flatten(torch.nonzero(attack_no_success))
        images_a_2 = torch.index_select(images_a_1, 0, attack_no_success_index)
        labels_a_2 = torch.index_select(labels_a_1, 0, attack_no_success_index)
        # 选择Adam-FGSM对于集合A_2攻击成功的样本集合B
        images_under_attack_extra_data = atk3(images_a_2, labels_a_2)
        images_under_attack = images_under_attack_extra_data[0]
        _, predict_attacked_images = torch.max(model(images_under_attack), 1)
        attack_success = (labels_a_2 != predict_attacked_images)
        Adam_FGSM_success = attack_no_success.sum().item()
        Adam_FGSM_success_sum += Adam_FGSM_success
        attack_success_index = torch.flatten(torch.nonzero(attack_success))
        images_b = torch.index_select(images_a_2, 0, attack_success_index)
        labels_b = torch.index_select(labels_a_2, 0, attack_success_index)
        #
        if images_b.shape[0] != 0:
            small_dataset.append((images_b, labels_b))

        bar.update(labels.shape[0])
        if test_count == 1:
            print('predict_correct_element_num: ', predict_correct_index.nelement())

        # to quickly test
        # break
        print('I-FGSM_no_success %d,'
              'MI-FGSM_no_success %d,'
              'Adam-FGSM_success %d',
              'Adam-FGSM_success sum %d',
              I_FGSM_no_success,
              MI_FGSM_no_success,
              Adam_FGSM_success,
              Adam_FGSM_success_sum
              )

    bar.close()
    return small_dataset


def attack_one_model(model, test_loader, attack_name, Epsilon, Iterations, Momentum):
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_count = 0
    sample_attacked_num = 0
    attack_success_num = 0
    print('get_hard_to_classify has been done!')
    bar = tqdm(total=10000)
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
        # predict_answer为一维向量，大小为batch_size
        predict_answer = (labels == predict)
        # torch.nonzero会返回一个二维矩阵，大小为（nozero的个数）*（1）
        nozero_predict_answer = torch.nonzero(predict_answer)
        # 我们要确保 predict_correct_index 是一个一维向量,因此使用flatten,其中的元素内容为下标
        predict_correct_index = torch.flatten(nozero_predict_answer)
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)
        # ------------------------------
        atk = None
        if attack_name == 'I_FGSM':
            atk = BIM(model, eps=Epsilon, alpha=Epsilon / Iterations,
                      steps=Iterations, )
        elif attack_name == 'MI_FGSM':
            atk = MIFGSM(model, eps=Epsilon, steps=Iterations, decay=Momentum)
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
        bar.update(labels.shape[0])

        # to quickly test
        # break

    attack_succ_rate = (attack_success_num / sample_attacked_num) * 100.0
    print('attack_success_num,sample_attacked_num', attack_success_num, sample_attacked_num)
    norm_1_norm_inf_similarity_total = norm_1_norm_inf_similarity_total / len(test_loader)
    bar.close()
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
                    filter_test_loader = get_hard_to_classify(model, test_loader,
                                                              Epsilon=Epsilon / 255.,
                                                              Iterations=Iterations,
                                                              Momentum=Momentum)
                    attack_succ, extra_data = attack_one_model(model=model,
                                                               test_loader=filter_test_loader,
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
    batch_size = 300
    attack_method_set = ['I_FGSM', 'MI_FGSM', 'Adam_FGSM']  # 'I_FGSM',  'MI_FGSM' ,'Adam_FGSM'
    # model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']
    model_name_set = ['VGG19', ]  # 'ResNet101', 'DenseNet121'
    attack_many_model(model_name_set,
                      attack_method_set,
                      'ImageNet',
                      batch_size,
                      work_name='explain',
                      Epsilon_set=[5],
                      Iterations_set=[15],
                      Momentum=1.0)

    print()
    print("----ALL WORK HAVE BEEN DONE!!!----")
