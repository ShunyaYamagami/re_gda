import os
import numpy as np
import random 
from PIL import Image
import cv2
import pandas as pd


def get_jigsaw(im:Image, resize, grid) -> Image:
    s = int(resize[0] / grid)
    tile = [im.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int)) for n in range(grid**2)]
    random.shuffle(tile)
    dst = Image.new('RGB', (int(s * grid), int(s * grid)))
    for i, t in enumerate(tile):
        dst.paste(t, (i % grid * s, int(i / grid) * s))
    im = dst

    return im


def fft_spectrums(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    img_abs = np.fft.fftshift(img_abs)
    img_pha = np.fft.fftshift(img_pha)
    return img_abs, img_pha

def spectrums_ifft(img_abs, img_pha):
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



def get_mix_filenames(config, edls):
    root = os.path.join("/nas/data/syamagami/GDA/data/", config.dataset.parent)
    target_text_trains = [os.path.join(root, f"{each_dset_name}.txt") for each_dset_name in config.dataset.target_dsets]  # mix用.mixするにはdslr_webcamだったら2つのdsetのfilenameを一遍に取得しなくてはならない

    df_target = [pd.read_csv(att, sep=" ", names=("filename", "label")) for att in target_text_trains]  # mix用.mixするにはdslr_webcamだったら2つのdsetのfilenameを一遍に取得しなくてはならない
    df_target = pd.concat(df_target)
    df_target['dset'] = df_target['filename'].apply(lambda f: f.split('/')[0])
    target_all_filenames = df_target.filename.values

    mix_filenames = []
    for dnum in range(len(np.unique(edls))):
        mix_filenames.append([filename for filename, edl in zip(target_all_filenames, edls) if edl == dnum])  # mix_filenamesのインデックス番号は推定ドメインラベルが一致

    return mix_filenames
    

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
