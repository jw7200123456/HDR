import numpy as np
import cv2


def shift_image(image: np.ndarray, shift: np.ndarray) -> np.ndarray:
    shift = shift.astype(np.int64)
    ret = np.zeros(image.shape, dtype=image.dtype)
    if shift[0] >= 0 and shift[1] >= 0:
        if shift[0] == 0 and shift[1] == 0:
            ret = image
        elif shift[0] == 0:
            ret[:, shift[1]:] = image[:, :-shift[1]]
        elif shift[1] == 0:
            ret[shift[0]:, :] = image[:-shift[0], :]
        else:
            ret[shift[0]:, shift[1]:] = image[:-shift[0], :-shift[1]]
    elif shift[0] >= 0 and shift[1] < 0:
        if shift[0] == 0:
            ret[:, :shift[1]] = image[:, -shift[1]:]
        else:
            ret[shift[0]:, :shift[1]] = image[:-shift[0], -shift[1]:]
    elif shift[0] < 0 and shift[1] >= 0:
        if shift[1] == 0:
            ret[:shift[0], :] = image[-shift[0]:, :]
        else:
            ret[:shift[0], shift[1]:] = image[-shift[0]:, :-shift[1]]
    else:
        ret[:shift[0], :shift[1]] = image[-shift[0]:, -shift[1]:]
    return ret


def alignment(ref_image: np.ndarray, target_image: np.ndarray,
              level: int = 3, max_level: int = 3, deadzone: int = 5) -> np.ndarray:
    """

    :param ref_image:
    :param target_image:
    :param level:
    :param max_level:
    :param deadzone:
    :return : shift value
    """
    shift = np.zeros(2)
    ratio = 0.5 ** (max_level - level)
    r_img = cv2.resize(ref_image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    t_img = cv2.resize(target_image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    # recursive call
    if level > 1:
        shift = alignment(ref_image, target_image, level-1, max_level, deadzone)
    shift *= 2

    # convert to binary image
    r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
    t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
    r_threshold = np.median(r_img)
    t_threshold = np.median(t_img)
    r_binary_img = (r_img > r_threshold).astype(np.float64)
    t_binary_img = (t_img > t_threshold).astype(np.float64)
    r_mask = np.logical_not(np.logical_and(r_img < (r_threshold + deadzone), r_img > (r_threshold - deadzone)))
    t_mask = np.logical_not(np.logical_and(t_img < (t_threshold + deadzone), t_img > (t_threshold - deadzone)))

    # search offset
    result = np.zeros([3, 3])
    t_binary_img = shift_image(t_binary_img, shift)
    t_mask = shift_image(t_mask, shift)
    for a in range(0, 3):
        for b in range(0, 3):
            temp_shift = np.array([a-1, b-1])
            temp_t_binary_img = shift_image(t_binary_img, temp_shift)
            temp_t_mask = shift_image(t_mask, temp_shift)

            diff = np.logical_xor(r_binary_img.astype(np.float64), temp_t_binary_img)
            diff = np.logical_and(diff, r_mask)
            diff = np.logical_and(diff, temp_t_mask)

            result[a, b] = np.sum(diff)

    if np.all(result == result[0]):
        return shift + np.array([0, 0])
    x, y = np.unravel_index(np.argmin(result), result.shape)
    return shift + np.array([x-1, y-1])


if __name__ == '__main__':
    a = np.zeros([8, 8, 3], dtype=np.uint8)
    b = np.zeros([8, 8, 3], dtype=np.uint8)
    a[0:2, :] = 50
    a[:, 0:2] = 50
    b[0:2, :] = 50
    b[:, 0:2] = 50
    a[5:8, 5:8] = 100
    b[4:7, 4:7] = 100
    s = alignment(a, b ,2, 2)
    print(s)
    c = shift_image(b, s)
    assert (np.all(a[3:, 3:] == c[3:, 3:]))
