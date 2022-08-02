import os
import numpy as np
import random 
from PIL import Image
import cv2
import pandas as pd
import itertools
import math


def get_jigsaw(im:Image, resize, grid) -> Image:
    s = int(resize[0] / grid)
    tile = [im.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int)) for n in range(grid**2)]
    random.shuffle(tile)
    dst = Image.new('RGB', (int(s * grid), int(s * grid)))
    for i, t in enumerate(tile):
        dst.paste(t, (i % grid * s, int(i / grid) * s))
    im = dst

    return im


def fft_spectrums(img: np.array) -> np.array:
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    img_abs = np.fft.fftshift(img_abs)
    img_pha = np.fft.fftshift(img_pha)
    return img_abs, img_pha


def spectrums_ifft(img_abs: np.array, img_pha: np.array):
    img_abs = np.fft.ifftshift(img_abs)
    img_pha = np.fft.ifftshift(img_pha)
    img_ifft = img_abs * (np.e ** (1j * img_pha))
    img_ifft = np.fft.ifft2(img_ifft).real
    # fmax, fmin = img_ifft.max(), img_ifft.min()
    # img_ifft = np.uint8(np.array([[(f-fmin)/(fmax-fmin) for f in img] for img in img_ifft])*255)  # minmax
    img_ifft = np.uint8(np.clip(img_ifft, 0, 255))  # clip
    return img_ifft




### 位相・振幅に一定値を入れてクラス情報を壊すことを試みる
def input_const_values(im:Image, resize, const_abs=True, const_pha=False, n_random=0, const_value=0) -> Image:
    """
        input: 
            im: データ拡張対象の画像 (C, W, H)
            const_abs, const_pha: abs/pha に一定値を入れるか否か
                    const_abs=True, const_pha=False  ->  推奨
                    const_abs=False, const_pha=True  ->  画像真っ黒になる
                    const_abs=True, const_pha=True   ->  ERROR
            n_random: 一定値を入れるpixel数
        return:
            フーリエ変換してデータ拡張処理し, 逆フーリエ変換して戻した画像.(W, H, C)
    """
    fourier_img = []
    # フーリエ変換は各チャンネルごとに独立して行う
    im = np.array(im).transpose(2, 0, 1)  # to (C, W, H)
    for img in im:
        img_abs, img_pha = fft_spectrums(img)

        ### random amp,pha following normal distribution
        if const_abs:
            noize_pixels = [(random.randint(0, resize[0]-1), random.randint(0, resize[0]-1)) for i in range(n_random)]  # 一定値を入れるピクセル(x, y)
            for noize_pixel in noize_pixels:
                img_abs[noize_pixel] = const_value
        if const_pha:
            noize_pixels = [(random.randint(0, resize[0]-1), random.randint(0, resize[0]-1)) for i in range(n_random)]  # 一定値を入れるピクセル(x, y)
            for noize_pixel in noize_pixels:
                img_pha[noize_pixel] = const_value

        img_ifft = spectrums_ifft(img_abs, img_pha)
        fourier_img.append(img_ifft)

    fourier_img = np.array(fourier_img).transpose(1, 2, 0) # to (W, H, C)
    im = Image.fromarray(fourier_img)

    return im




### 位相・振幅にランダムな値を入れる．
# 位相・振幅に，フーリエ変換直後と同じ平均・分散を持つ正規分布に慕う乱数を代入してクラス情報を破壊する事を試みる.
def input_random_values(im:Image, resize, randomize_abs=True, randomize_pha=False, n_random=0) -> Image:
    """
        input: 
            im: データ拡張対象の画像 (C, W, H)
            randomize_abs, randomize_pha: abs/pha にランダムな値を入れるか否か
                    randomize_abs=True, randomize_pha=False  ->  推奨
                    randomize_abs=False, randomize_pha=True  ->  画像真っ黒になる
                    randomize_abs=True, randomize_pha=True   ->  ERROR
            n_random: ランダムな値を入れるpixel数
        return:
            フーリエ変換してデータ拡張処理し, 逆フーリエ変換して戻した画像.(W, H, C)
    """
    fourier_img = []
    # フーリエ変換は各チャンネルごとに独立して行う
    im = np.array(im).transpose(2, 0, 1)  # to (C, W, H)
    for img in im:
        img_abs, img_pha = fft_spectrums(img)

        ### random amp,pha following normal distribution
        if randomize_abs:
            mean, std = img_abs.mean(), img_abs.std()
            normals = np.random.normal(mean, std, size=n_random)  # 同じ平均・標準偏差の正規分布に従う乱数をn_random個作成
            noize_pixels = [(random.randint(0, resize[0]-1), random.randint(0, resize[0]-1)) for i in range(n_random)]  # ランダムな値を入れるピクセル(x, y)
            for (noize_pixel, normal) in zip(noize_pixels, normals):
                img_abs[noize_pixel] = normal
        if randomize_pha:
            mean, std = img_pha.mean(), img_pha.std()
            # normals = np.random.normal(mean, std, size=n_random)  # 同じ平均・標準偏差の正規分布に従う乱数をn_random個作成
            random_values = [random.random() * math.pi for _ in range(n_random)]
            noize_pixels = [(random.randint(0, resize[0]-1), random.randint(0, resize[0]-1)) for i in range(n_random)]  # ランダムな値を入れるピクセル(x, y)
            for (noize_pixel, random_value) in zip(noize_pixels, random_values):
                img_pha[noize_pixel] = random_value

        img_ifft = spectrums_ifft(img_abs, img_pha)
        fourier_img.append(img_ifft)

    fourier_img = np.array(fourier_img).transpose(1, 2, 0) # to (W, H, C)
    im = Image.fromarray(fourier_img)

    return im



def get_all_filenames(config):
    root = os.path.join("/nas/data/syamagami/GDA/data/", config.dataset.parent)
    target_text_trains = [os.path.join(root, f"{each_dset_name}.txt") for each_dset_name in config.target_dsets_name.split('_')]  # mix用.mixするにはdslr_webcamだったら2つのdsetのfilenameを一遍に取得しなくてはならない

    df_target = [pd.read_csv(att, sep=" ", names=("filename", "label")) for att in target_text_trains]  # mix用.mixするにはdslr_webcamだったら2つのdsetのfilenameを一遍に取得しなくてはならない
    df_target = pd.concat(df_target)
    df_target['dset'] = df_target['filename'].apply(lambda f: f.split('/')[0])
    all_filenames = df_target.filename.values

    return all_filenames


def get_mix_filenames(config, edls):
    target_all_filenames = get_all_filenames(config)

    mix_filenames = []
    for dnum in range(len(np.unique(edls))):
        mix_filenames.append([filename for filename, edl in zip(target_all_filenames, edls) if edl == dnum])  # mix_filenamesのインデックス番号は推定ドメインラベルが一致

    return mix_filenames
    

### 振幅/位相を他画像とMix
def mix_amp_phase_and_mixup(im: Image, root, resize, mix_filenames, mix_amp=True, mix_pha=False, mixup=True, LAMB = 0.7) -> Image:
    """ ランダムな他2つの画像と位相/振幅をMixし, その後ピクセル空間でMixUp
        input: 
            im: データ拡張対象の画像 (C, W, H)
            root, filenames: mixする他画像のPATH
            LAMB: 位相/振幅をmixする時の他画像の重み
        return:
            フーリエ変換してデータ拡張処理し, 逆フーリエ変換して戻した画像.(W, H, C)
    """
    MXIUP_RATE = 0.5  # image mixup rate

    if not isinstance(mix_filenames, list):
        mix_filenames = list(mix_filenames)
    # samples = random.sample(filenames, 2)  # filenamesの第2次元目が空配列なので
    samples = random.sample(mix_filenames, 2)  # 本番環境のfilenamesはカンマ区切りでなくリストでないのでtolist()をつける．
    samples = [Image.open(os.path.join(root, s)).convert("RGB").resize(resize) for s in samples]
    sample0 = np.array(samples[0]).transpose(2, 0, 1)
    sample1 = np.array(samples[1]).transpose(2, 0, 1)

    fourier_img = []
    # フーリエ変換は各チャンネルごとに独立して行う
    im = np.array(im).transpose(2, 0, 1)  # to (C, W, H)
    for img, sam0, sam1 in zip(im, sample0, sample1):
        img_abs, img_pha = fft_spectrums(img)
        ##### samples
        sam0_abs, sam0_pha = fft_spectrums(sam0)
        sam1_abs, sam1_pha = fft_spectrums(sam1)
        ### mix amp
        if mix_amp:
            mix0_abs = LAMB*img_abs + (1-LAMB)*sam0_abs
            mix1_abs = LAMB*img_abs + (1-LAMB)*sam1_abs
        else:
            mix0_abs = img_abs
            mix1_abs = img_abs
        ### mix phase
        if  mix_pha:
            mix0_pha = LAMB*img_pha + (1-LAMB)*sam0_pha
            mix1_pha = LAMB*img_pha + (1-LAMB)*sam1_pha
        else:
            mix0_pha = img_pha
            mix1_pha = img_pha
            
        mix0_ifft = spectrums_ifft(mix0_abs, mix0_pha)
        mix1_ifft = spectrums_ifft(mix1_abs, mix1_pha)

        if mixup:
            mix_img = MXIUP_RATE*mix0_ifft + (1-MXIUP_RATE)*mix1_ifft  # mixup
        else:
            mix_img = mix0_ifft

        # fmax, fmin = mix_img.max(), mix_img.min()
        # mix_img = np.uint8(np.array([[(f-fmin)/(fmax-fmin) for f in img] for img in mix_img])*255)  # minmax
        mix_img = np.uint8(np.clip(np.array(mix_img), 0, 255))  # clip

        fourier_img.append(mix_img)

    fourier_img = np.array(fourier_img).transpose(1, 2, 0) # to (W, H, C)
    im = Image.fromarray(fourier_img)

    return im



def mask_randomly(im:Image, resize, square_edge=20, rate=0.5) -> Image:
    """ ランダムにマスクする """
    num_per_edge = resize[0]//square_edge + 1
    
    all_position_pairs = [pair for pair in itertools.product(list(range(num_per_edge)), list(range(num_per_edge)))]
    mask_positions = random.sample(all_position_pairs, int(num_per_edge*num_per_edge*rate))
    mask = Image.new("L", (square_edge, square_edge), 128)
    for x, y in mask_positions:
        im.paste(mask, (x*square_edge, y*square_edge))

    return im



def cutmix_self(im:Image, resize, grid=3, n_cutmix=2) -> Image:
    """ 1つの画像内でcropして貼り付けする """
    s = int(resize[0] / grid)
    tile = [im.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int)) for n in range(grid**2)]
    random.shuffle(tile)

    for n in range(n_cutmix):
        pos = random.sample(range(resize[0]-s), 2)
        im.paste(tile[n], (pos[0], pos[1]))

    return im



def cutmix_other(im:Image, resize, root, mix_filenames, mix_edge_div=5, crop_part='center') -> Image:
    """ 他の画像と中心部分をcutmixする. """
    mix_im = Image.open(os.path.join(root, random.choice(mix_filenames))).convert("RGB").resize(resize)
    mix_edge = resize[0] // mix_edge_div
    pos_lu = resize[0]//2 - mix_edge
    pos_rb = resize[0]//2 + mix_edge
    
    if crop_part == 'center':
        crop_im = mix_im.crop((pos_lu, pos_lu, pos_rb, pos_rb))
    elif crop_part == 'left_upper':
        crop_im = mix_im.crop((0, 0, 2*mix_edge, 2*mix_edge))  # 他の画像の左上の部分を元画像の中心に貼り付け.

    im.paste(crop_im, (pos_lu, pos_lu))

    return im




def cutmix_spectrums(im:Image, resize, root, mix_filenames, does_mix_amp=False, does_mix_pha=True, mix_edge_div=2) -> Image:
    """ 他の画像と振幅/位相のスペクトルをcutmixする. """
    square = resize[0] // mix_edge_div
    lt, rb = resize[0] - square, resize[0] + square

    mix_im = Image.open(os.path.join(root, random.choice(mix_filenames))).convert("RGB").resize(resize)
    im = np.array(im).transpose(2, 0, 1)
    mix_im = np.array(mix_im).transpose(2, 0, 1)

    fourier_img = []
    for img, mix_img in zip(im, mix_im):
        amp, pha = fft_spectrums(img)
        mixed_amp, mixed_pha = fft_spectrums(mix_img)

        if does_mix_amp:
            amp[lt:rb, lt:rb] = mixed_amp[lt:rb, lt:rb]
        if does_mix_pha:
            pha[lt:rb, lt:rb] = mixed_pha[lt:rb, lt:rb]

        # Image.fromarray(np.uint8(np.clip(20*np.log(amp), 0, 255)))
        imifft = spectrums_ifft(amp, pha)
        fimg = np.uint8(np.clip(imifft, 0, 255))
        fourier_img.append(fimg)

    fourier_img = np.array(fourier_img).transpose(1, 2, 0) # to (W, H, C)
    im = Image.fromarray(fourier_img)
    return im



def leave_amp_pha_big_small(im:Image, amp_pha='amp', big_small='big') -> Image:
    """ 位相/振幅の大きな/小さな所だけ残す
        amp_pha: 振幅残すか位相残すか
        big_small: 大きな部分を残すか小さな部分を残すか
    """
    im = np.array(im).transpose(2, 0, 1)  # to (C, W, H)
    
    fourier_img = []
    for img in im:
        # フーリエ変換
        img_fft = np.fft.fft2(img)
        img_abs = np.abs(img_fft)
        img_pha = np.angle(img_fft)

        if amp_pha=='amp' and big_small=='big':
            # 大きな振幅だけ残す  全体にノイズが走っただけ？
            img_abs = np.where(img_abs < np.percentile(img_abs, 98), 0, img_abs)
        if amp_pha=='amp' and big_small=='small':
            # 小さな振幅だけ残す  背景の色が変わっただけ？
            img_abs = np.where(img_abs > np.percentile(img_abs, 99.98), 0, img_abs)
        if amp_pha=='pha' and big_small=='big':
            # 大きな位相だけ残す
            img_pha = np.where(img_pha < np.percentile(img_pha, 2), 0, img_pha)
        if amp_pha=='pha' and big_small=='small':
            # 小さな位相だけ残す  画面が暗くなっただけのような
            img_pha = np.where(img_pha > np.percentile(img_pha, 98), 0, img_pha)

        # 逆フーリエ変換
        img_ifft = img_abs * (np.e ** (1j * img_pha))
        img_ifft = np.fft.ifft2(img_ifft).real
        fmax, fmin = img_ifft.max(), img_ifft.min()
        img_ifft = np.uint8(np.array([[(f-fmin)/(fmax-fmin) for f in img] for img in img_ifft])*255)

        fourier_img.append(img_ifft)

    fourier_img = np.array(fourier_img).transpose(1, 2, 0) # to (W, H, C)
    fourier_img = Image.fromarray(fourier_img)

    return fourier_img