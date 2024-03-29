import argparse
import logging
import yaml
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffattack import DiffAttack
import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data
from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion
from runners.diffattack_ddpm_cifar import Diffusion_cifar
import torchvision.transforms as T
from tqdm import tqdm

#创建一个模型类
class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # 获取一个图像分类器并将其移动到指定设备上
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ddpm_cifar':
            self.runner = Diffusion_cifar(args)
        else:
            raise NotImplementedError('unknown diffusion type')
        #注册一个名‘counter’的缓冲区，其大小是为1的零张量并将其移动到指定设备上
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    #重置函数
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, return_mid=False, return_std_z=False, steps = 0):

        if steps!=0:
            tmp_steps = self.args.t
            self.args.t = steps
        #.item()用于获取张量中的单个值，并将其转换为python标量
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        #采用F.interpolate()函数，对输入图像进行双插值操作。
        #mode='bilinear'表示采用双插值
        #align_corners=False表示不强制对图像的角点元素进行对齐
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        #time.time()获取当前时间
        start_time = time.time()

        if return_std_z == False:
            x_re,mid_x,ori_x = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        else:
            x_re, mid_x, ori_x, a_std, z__ = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, return_std_z=True)
        #time.time() - start_time表示被除数，程序运行的总时间，60表示除数
        #计算的整除部分赋值给minutes，余数部分赋值给seconds
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            #{:0>2}表示将 minutes 变量格式化为长度为2的字符串，如果不足两位则在前面用 0 填充
            #{:05.2f}表示将seconds变量格式化长度为5的字符串，并保留两位小数
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        x_re = (x_re + 1) * 0.5
        out = self.classifier(x_re)

        self.counter += 1

        if steps!=0:
            self.args.t = tmp_steps

        if return_mid == False:
            return out

        if return_std_z:
            return out, mid_x, ori_x, a_std, z__
        return out, mid_x, ori_x
#测试bpda攻击方法的性能
def test_bpda_adv(args, model, x_adv, y_val, config):
    x_adv = x_adv.to(config.device)
    with torch.no_grad():
        bs = 10 #设置batch size大小
        n_batches = int(np.ceil(x_adv.shape[0] / bs)) #计算处理x_adv数据集总共需要多少个批次，np.ceil()表示向上取整
        robust_flags = torch.zeros(x_adv.shape[0], dtype=torch.bool, device=x_adv.device)
        y_adv = torch.empty_like(y_val)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs #计算当前批次的起始索引
            end_idx = min((batch_idx + 1) * bs, x_adv.shape[0])#计算结束索引且保证不超过x_adv的长度
            #从输入数据中切片出当前批次的样本，并将其克隆到与x_adv相同的设备上，避免了在原始数据上进行操作
            x = x_adv[start_idx:end_idx, :].clone().to(x_adv.device)
            y = y_val[start_idx:end_idx].clone().to(x_adv.device)
            #取预测结果中概率最大的类别作为最终输出。
            output = model(x, steps=args.original_step_t).max(dim=1)[1]
            y_adv[start_idx: end_idx] = output #将当前批次的模型预测结果存储到 y_adv 中对应的位置。
            #用来比较预测标签和真实标签是否相等，返回值为bOOl
            correct_batch = y.eq(output)
            #将当前批次的比较值过赋给robust_flags
            robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
        #计算总的鲁棒性准确率
        robust_accuracy = torch.sum(robust_flags).item() / x_adv.shape[0]
    print(f'--------------')
    print(f'Robust accuracy of BPDA attack: {robust_accuracy}')
#测试diffattack方法的攻击性能
def eval_diffattack(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    attack_version = args.attack_version  # ['standard', 'rand', 'custom']
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']

    # ---------------- apply the attack to sde_adv ----------------
    print(f'apply the attack to sde_adv [{args.lp_norm}]...')
    model_.reset_counter()
    adversary_sde = DiffAttack(model, norm=args.lp_norm, eps=args.adv_eps,
                               version=attack_version, attacks_to_run=attack_list,
                               log_path=f'{log_dir}/log_sde_adv.txt', device=config.device, args=args)
    if attack_version == 'custom':
        adversary_sde.apgd.n_restarts = 1
        adversary_sde.fab.n_restarts = 1
        adversary_sde.apgd_targeted.n_restarts = 1
        adversary_sde.fab.n_target_classes = 9
        adversary_sde.apgd_targeted.n_target_classes = 9
        adversary_sde.square.n_queries = 5000
    if attack_version == 'rand':
        adversary_sde.apgd.eot_iter = args.eot_iter
        print(f'[adv_sde] rand version with eot_iter: {adversary_sde.apgd.eot_iter}')
    print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

    x_adv_sde = adversary_sde.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
    print(f'x_adv_sde shape: {x_adv_sde.shape}')
    torch.save([x_adv_sde, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')

    if args.bpda==1:
        test_bpda_adv(args, model, x_adv_sde, y_val, config)

def robustness_eval(args, config):
    #若args.attack_version在列表['stadv', 'standard', 'rand']中，则用下划线连接args.diffusion_type, args.attack_version
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    #用os.path.join()方法构建日志文件夹路径，其中包括图片文件夹路径，分类器名称等信息
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    #使用os.makedirs()函数构建路径，exist_ok=True表示若路劲已存在则不做任何操作
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    #file_mode="w+"指定文件模式为可读写，should_flush=True设置立即刷新，确保日志信息能够及时的写入文件
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        #对模型进行封装，以便可以在多个GPU上运行
        model = torch.nn.DataParallel(model)
    #将模型设置为评估模式，无需进行梯度更新并将其移动到指定的设备上
    model = model.eval().to(config.device)

    # load data
    x_val, y_val = load_data(args, adv_batch_size)

    # eval classifier and sde_adv against attacks
    if args.attack_version in ['standard', 'rand', 'custom']:
        eval_diffattack(args, config, model, x_val, y_val, adv_batch_size, log_dir)
    else:
        raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_interval', type=int, default=10, help='Sampling interval')
    parser.add_argument('--bpda', type=int, default=0, help='whether it is BPDA attack')
    parser.add_argument('--original_step_t', type=int, default=0, help='original diffusion steps for BPDA attack')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    robustness_eval(args, config)
