import numpy as np
from scipy import ndimage

_categories = (-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15, 15, 15, 15, 15, 15,
               16, 16, 16, 16, 16, 16, 16)

RPC_SUPPORT_CATEGORIES = (1, 17, 200)

_coco_categories = (
    -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7,
    7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11)
COCO_SUPPORT_CATEGORIES = (1, 12, 80)


def contiguous_coco_category_to_super_category(category_id, num_classes):
    cat_id = -1
    assert num_classes in COCO_SUPPORT_CATEGORIES, 'Not support {} density categories'.format(num_classes)
    if num_classes == 12:
        cat_id = _coco_categories[category_id]
    elif num_classes == 1:
        cat_id = 0
    elif num_classes == 80:
        cat_id = category_id - 1
    assert 79 >= cat_id >= 0
    return cat_id


def rpc_category_to_super_category(category_id, num_classes):
    """Map category to super-category id
    Args:
        category_id: list of category ids, 1-based
        num_classes: 1, 17, 200
    Returns:
        super-category id, 0-based
    """
    cat_id = -1
    assert num_classes in RPC_SUPPORT_CATEGORIES, 'Not support {} density categories'.format(num_classes)
    if num_classes == 17:
        cat_id = _categories[category_id]
    elif num_classes == 1:
        cat_id = 0
    elif num_classes == 200:
        cat_id = category_id - 1
    assert 199 >= cat_id >= 0
    return cat_id


def generate_density_map(labels, boxes, scale=50.0 / 800, size=50, num_classes=200, min_sigma=1):
    density_map = np.zeros((num_classes, size, size), dtype=np.float32)
    for category, box in zip(labels, boxes):
        x1, y1, x2, y2 = [x * scale for x in box]
        w, h = x2 - x1, y2 - y1
        box_radius = min(w, h) / 2
        sigma = max(min_sigma, box_radius * 5 / (4 * 3))  # 3/5 of gaussian kernel is in box
        cx, cy = round((x1 + x2) / 2), round((y1 + y2) / 2)
        density = np.zeros((size, size), dtype=np.float32)
        density[cy, cx] = 100
        density = ndimage.filters.gaussian_filter(density, sigma, mode='constant')
        density_map[category, :, :] += density

    return density_map


if __name__=='__main__':
    from scipy.ndimage import gaussian_filter
    a = np.arange(50, step=2).reshape((5,5))
    gaussian_filter(a, sigma=1)
    from scipy import misc
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ascent = misc.ascent()
    result = gaussian_filter(ascent, sigma=5)
    ax1.imshow(ascent)
    ax2.imshow(result)
    plt.show()
    plt.savefig("densitymap.png")