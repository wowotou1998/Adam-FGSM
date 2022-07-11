[//]: # (这个项目的论文 Jiebao Zhang, Wenhua Qian*, Rencan Nie, Jinde Cao, Dan Xu.  最终代码版本 [pdf]&#40;https://doi.org/10.1007/s10489-022-03437-z&#41; )

[//]: # (根据 torchattacks 3.2.6 的更新， 我修改了自定义攻击方法， 也修改了迭代过程的对抗样本存储方法，三个分析指标可以不用改， 直接在 torchattacks3.2.6 版本上也可以直接运行)

# Adam-FGSM

Official implementation for

- Generate Adversarial Examples by Adaptive Moment Iterative Fast Gradient Sign Method, Applied Intelligence 2022. ([Paper](https://doi.org/10.1007/s10489-022-03437-z))

For any questions, contact (zhangjiebao2014@mail.ynu.edu.cn).

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Torattacks >= 3.2.6](https://github.com/Harry24k/adversarial-attacks-pytorch)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [Pytorchcv](https://github.com/osmr/imgclsmob)
## Preparations
- some file paths will be created manually
- the datasets will be downloaded automatically at the first, but for ImageNet, you should sign up in the [website](https://image-net.org/) and then log in before you start to download it.


## Attack different models on different datasets

```
python attack_models_on_datasets.py 
```



[//]: # (## References)



## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{zhang_generate_2022,
	title = {Generate adversarial examples by adaptive moment iterative fast gradient sign method},
	issn = {1573-7497},
	url = {https://doi.org/10.1007/s10489-022-03437-z},
	doi = {10.1007/s10489-022-03437-z},
	journaltitle = {Applied Intelligence},
	author = {Zhang, Jiebao and Qian, Wenhua and Nie, Rencan and Cao, Jinde and Xu, Dan},
	date = {2022-04-25},
}
```
