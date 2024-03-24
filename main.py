import sys
import numpy as np
import cv2
import glob
from aligment import alignment, shift_image
import matplotlib.pyplot as plt


def CRC(sample: np.ndarray, exposure: np.ndarray, weight: np.ndarray, lambd: int = 20):
    pnt_cnt = sample.shape[0]
    img_cnt = sample.shape[1]
    A = np.zeros([pnt_cnt*img_cnt+1+254, 256+pnt_cnt])
    b = np.zeros(A.shape[0])

    idx = 0
    for p in range(sample.shape[0]):
        for i in range(sample.shape[1]):
            zij = int(sample[p, i])
            A[idx, zij] = weight[zij]
            A[idx, 256+p] = -weight[zij]
            b[idx] = weight[zij]*exposure[i]
            idx += 1

    A[idx, 128] = 1
    idx += 1

    for p in range(1, 254):
        A[idx, p-1] = lambd*weight[p]
        A[idx, p] = -2*lambd*weight[p]
        A[idx, p+1] = lambd*weight[p]
        idx += 1

    x = np.linalg.lstsq(A, b, rcond=None)
    g = x[0][:256]

    # visualize recovered crc
    # plt.plot(g, np.linspace(0, 255, g.shape[0]))
    # plt.show()
    # plt.close()
    return g


def HDR(images: list[np.ndarray], crc: np.ndarray, weight:np.ndarray, exposure: np.ndarray):
    img_h = images[0].shape[0]
    img_w = images[0].shape[1]
    flatten_image = np.reshape(images[0][:, :, 0], (img_w * img_h))
    rad_map = np.zeros([flatten_image.shape[0], 3])
    weight_sum = np.zeros([flatten_image.shape[0], 3])
    for idx, image in enumerate(images):
        for c in range(3):
            flatten_image = np.reshape(image[:, :, c], (img_w * img_h))
            rad_map[:, c] += weight[flatten_image] * (crc[:, c][flatten_image] - exposure[idx])
            weight_sum[:, c] += weight[flatten_image]
    rad_map = np.exp(rad_map / weight_sum)
    # rad_map = (rad_map / np.amax(rad_map) * 255)

    hdr = np.zeros([img_h, img_w, 3]).astype(np.float32)
    for c in range(3):
        hdr[:, :, c] = np.reshape(rad_map[:, c], (img_h, img_w)).astype(np.float32)
    plt.figure(constrained_layout=False, figsize=(10, 10))
    plt.title("fused HDR radiance map", fontsize=20)
    plt.imshow(hdr)
    plt.show()
    return hdr


def tone_mapping_global(hdr: np.ndarray, a: float, lwhite: float):
    delta = 0.001
    lw = 0.299 * hdr[:, :, 0] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 2] + 0.0001
    lw_bar = np.exp(np.average(np.log(delta + lw)))
    lm = a * lw / lw_bar
    ld = lm * (1 + lm / (lwhite ** 2)) / (1 + lm)
    ldr = np.zeros(hdr.shape)
    for c in range(3):
        ldr[:, :, c] = hdr[:, :, c]*255 / lw * ld
    ldr[ldr > 255] = 255

    # plt.figure()  # constrained_layout=False, figsize=(10, 10))
    # plt.title("global tone map", fontsize=10)
    # plt.imshow(ldr, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    return ldr, lm


def tone_mapping_local(hdr: np.ndarray, lm: np.ndarray, eps: float, level: int):
    l_blur = np.zeros([lm.shape[0], lm.shape[1], level+1])
    l_blur[:, :, 0] = lm
    for i in range(1, level+1):
        sigma = 1.02 ** i
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        l_blur[:, :, i] = cv2.GaussianBlur(lm, (kernel_size, kernel_size), sigmaX=sigma)

    v_blur = np.zeros([lm.shape[0], lm.shape[1], level])
    for i in range(level):
        v_blur[:, :, i] = l_blur[:, :, i+1] - l_blur[:, :, i]

    l_blur[:, :, 1:][v_blur < eps] = 0
    l_blur_max = np.max(l_blur, axis=2)

    ld = lm / (1 + l_blur_max)
    lw = 0.299 * hdr[:, :, 0] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 2] + 0.0001
    ldr = np.zeros(hdr.shape)
    for c in range(3):
        ldr[:, :, c] = hdr[:, :, c]*255 / lw * ld
    ldr[ldr > 255] = 255

    # plt.figure()  # constrained_layout=False, figsize=(10, 10))
    # plt.title("local tone map", fontsize=10)
    # plt.imshow(ldr, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    return ldr


def main() -> int:
    img_list = sorted(list(glob.glob("./test_image/*")))
    img_cnt = len(img_list)
    img_w = 1500
    img_h = 1000
    weight = np.linspace(1, 128, 128)
    weight = np.concatenate([weight, weight[::-1]])
    exposure_time = np.array([np.log(1 / 160), np.log(1 / 250), np.log(1 / 100), np.log(1 / 80), np.log(1 / 125), np.log(1 / 50), np.log(1 / 40), np.log(1 / 60),
                     np.log(1 / 25), np.log(1 / 20), np.log(1 / 30), np.log(1 / 13), np.log(1 / 10), np.log(1 / 15), np.log(1 / 6), np.log(1 / 5), np.log(1 / 8),
                     np.log(1 / 3), np.log(0.4), np.log(1 / 4), np.log(0.625), np.log(0.7692), np.log(0.5), np.log(1.3), np.log(1.6), np.log(1), np.log(2.5)])

    images = []
    for img_name in img_list:
        image = cv2.resize(cv2.imread(img_name), (img_w, img_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # Alignment
    align_MTB = cv2.createAlignMTB()
    align_MTB.process(images, images)
    align_images = [images[0]]
    for image in images[1:]:
        # shift = alignment(images[0], image)
        align_img = image #  shift_image(image, shift)  #
        align_images.append(align_img)
        # cv2.imshow("align_img", align_img)
        # cv2.waitKey(0)

    # Sample pixels
    sample_pixels = np.zeros([500, img_cnt, 3])
    for i in range(20):
        for j in range(25):
            for idx, image in enumerate(align_images):
                sample_pixels[i*25+j, idx] = image[45*(i+1), 50*(j+1)]

    # Get Camera Response Curve
    crc = np.zeros([256, 3])
    for i in range(3):
        crc[:, i] = CRC(sample_pixels[:, :, i], exposure_time, weight)

    # Gen radiance map
    rad_map = HDR(align_images, crc, weight, exposure_time)

    # Tone mapping global
    ldr_global, lm = tone_mapping_global(rad_map, 1.7, 2500)
    cv2.imwrite("./tone_global.png", cv2.cvtColor(ldr_global.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Tone mapping local
    ldr_local = tone_mapping_local(rad_map, lm, 0.05, 8)
    cv2.imwrite("./tone_local.png", cv2.cvtColor(ldr_local.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return 0


if __name__ == '__main__':
    sys.exit(main())
