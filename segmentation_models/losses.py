import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects

from .metrics import jaccard_score, f_score

SMOOTH = 1.

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
]


# ============================== Jaccard Losses ==============================

def jaccard_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Jaccard loss function for imbalanced datasets:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Jaccard loss in range [0, 1]

    """
    return 1 - jaccard_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)


def bce_jaccard_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    r"""Sum of binary crossentropy and jaccard losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch (only for jaccard loss)

    Returns:
        loss
    
    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + jaccard_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss


def cce_jaccard_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Sum of categorical crossentropy and jaccard losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        loss
    
    """
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + jaccard_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)


# Update custom objects
get_custom_objects().update({
    'jaccard_loss': jaccard_loss,
    'bce_jaccard_loss': bce_jaccard_loss,
    'cce_jaccard_loss': cce_jaccard_loss,
})


# ============================== Dice Losses ================================

def dice_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, beta=1.):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=beta)


def bce_dice_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True, beta=1.):
    r"""Sum of binary crossentropy and dice losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta)
    return loss


def cce_dice_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True, beta=1.):
    r"""Sum of categorical crossentropy and dice losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + dice_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image, beta=beta)


def jaccard_distance_l(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Update custom objects
get_custom_objects().update({
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cce_dice_loss': cce_dice_loss,
    'jaccard_distance_l': jaccard_distance_l,
})
