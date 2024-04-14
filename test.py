import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import data_loading as dl
import cv2
from arch import LYT, Denoiser
import argparse
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import lpips
import torch
from find_gamma import *
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Filter deprecated warnings from torchvision and lpips
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")#过滤了torchvision和lpips模块的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="lpips")

tf.random.set_seed(1)

def get_time():#获取当前时间
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return current_time

def start_test(dataset, weights, gtmean):
    print('LYT-Net 2024 (c) Brateanu, A., Balmez, R., Avram A., Orhei, C.C.')
    print(f"({get_time()}) Testing on dataset: {dataset}")

    raw_test_path = ''
    corrected_test_path = ''
    weights_path = weights

    # Load dataset 根据指定的数据集设置原始图像和校正后图像的路径
    raw_test_path = './data/LOLv1/Test/input/*.png'
    corrected_test_path = './data/LOLv1/Test/target/*.png'

    # Load dataset 加载数据集，并打印加载数据集的时间信息
    print(f'({get_time()}) Loading dataset {dataset}.')
    test_dataset = dl.get_datasets_metrics(raw_test_path, corrected_test_path, 0)
    print(f'({get_time()}) Successfully loaded dataset.')

    # Build model
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))

    # 加载了预训练的模型权重
    model.load_weights(f'{weights_path}')

    # Results directory
    os.makedirs(f'./results/{dataset}', exist_ok=True)
    file_names = os.listdir(raw_test_path[:len(raw_test_path)-5]) # 创建用于保存结果的文件夹。
    original_stdout = sys.stdout # 临时关闭标准输出和标准错误，并创建一个LPIPS模型对象。
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    lpips_model = lpips.LPIPS(net='alex')
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    file_names.sort()

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_samples = 0
    image = []
    for raw_image, corrected_image in tqdm(test_dataset):
        if is_dark(raw_image):
            generated_image = model(raw_image) # 使用模型对原始图像进行处理，生成修正后的图像
            raw_image = (raw_image + 1.0) / 2.0 # 对图像进行归一化
            corrected_image = (corrected_image + 1.0) / 2.0
            generated_image = (generated_image + 1.0) / 2.0

            if gtmean:# 对生成的图像进行gamma校正
                gamma_values = np.linspace(0.4, 2.5, 100)
                optimal_gamma = find_optimal_gamma(generated_image, corrected_image, gamma_values, 1.0)
                generated_image = adjust_gamma(generated_image, optimal_gamma)
            # 计算PSNR、SSIM和LPIPS
            # total_psnr += tf.image.psnr(corrected_image, generated_image, max_val=1.0)
            # total_ssim += tf.image.ssim(corrected_image, generated_image, max_val=1.0)
            # total_lpips += compute_lpips(generated_image, corrected_image, lpips_model)

            # # save to results folder
            save_path = f'./results/{dataset}/{file_names[num_samples]}'
            generated_image_np = generated_image.numpy()
            # plt.imsave(save_path, generated_image_np[0], format='png')

            num_samples += 1

            image.append(generated_image_np)
        else:
            image.append(raw_image)
        
    # avg_psnr = total_psnr / num_samples
    # avg_ssim = total_ssim / num_samples
    # avg_lpips = total_lpips / num_samples
    #
    # avg_psnr = tf.reduce_mean(avg_psnr)
    # avg_ssim = tf.reduce_mean(avg_ssim)
    # avg_lpips = tf.reduce_mean(avg_lpips)
    #
    # print(f"PSNR: {avg_psnr:.6f}")
    # print(f"SSIM: {avg_ssim:.6f}")
    # print(f"LPIPS: {avg_lpips:.6f}")
    return image

def is_dark(img, threshold=10):
    img = torch.tensor(img)
    img = torch.squeeze(img, dim=0).permute(1, 2, 0)
    gray = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)  # 转灰度图 输入应该是 (通道数, 高度, 宽度)
    averge = cv2.mean(gray)[0]
    return averge < threshold  # 1是低照度，0不是

    
def compute_lpips(predicted_image, ground_truth_image, lpips_model):
    # lpips_model = lpips.LPIPS(net='alex')

    predicted_image_pt = torch.from_numpy(predicted_image.numpy()).permute(0, 3, 1, 2).float()  
    ground_truth_image_pt = torch.from_numpy(ground_truth_image.numpy()).permute(0, 3, 1, 2).float()  

    predicted_image_pt = predicted_image_pt / 255.0 if predicted_image_pt.max() > 1.0 else predicted_image_pt
    ground_truth_image_pt = ground_truth_image_pt / 255.0 if ground_truth_image_pt.max() > 1.0 else ground_truth_image_pt

    with torch.no_grad():
        lpips_distance = lpips_model(predicted_image_pt, ground_truth_image_pt)

    return lpips_distance.item()


if __name__ == '__main__':
    image = start_test(dataset='LOLv1', weights='./model_weights/LOLv1.h5', gtmean=True)