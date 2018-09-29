import numpy as np
import SimpleITK as sitk
def window_convert( pix, center, width):
    pix_out = np.zeros(shape=pix.shape, dtype=np.uint8)
    low = center - width / 2 # 0
    hig = center + width / 2 # 60
    w1 = np.where(pix > low) and np.where(pix < hig)
    pix_out[w1] = ((pix[w1] - center + 0.5) / (width - 1) + 0.5) * 255
    pix_out[np.where(pix <= low)] = 0
    pix_out[np.where(pix >= hig)] = 255
    return pix_out

def window_convert_1( pix, center, width):
    low = center - width / 2
    hig = center + width / 2
    return sitk.GetArrayFromImage(sitk.IntensityWindowing(sitk.GetImageFromArray(pix), low, hig, 0, 255))
