import numpy as np
import math
import richdem as rd


def readTIFF(path, tiff, masked=False, padding=False):
    dem = rd.LoadGDAL(path+tiff)
    no_data = dem.no_data

    if padding:
        pad_width = 64 - dem.shape[1]
        dem = np.pad(dem, pad_width=((0, 0), (0, pad_width)), mode='edge')

    if not masked:
        return np.asarray(dem)
    elif no_data > 0:
        dem_np = np.asarray(dem)
        return np.ma.masked_where(dem_np > no_data/10, dem_np)
    else:
        dem_np = np.asarray(dem)
        return np.ma.masked_where(dem_np < no_data/10, dem_np)

def readDem(dem_path, masked=False):
    dem = rd.LoadGDAL(dem_path, 3.4028230607370965e+38)
    no_data = dem.no_data
    if not masked:
        return np.asarray(dem)
    elif no_data > 0:
        dem_np = np.asarray(dem)
        return np.ma.masked_where(dem_np > no_data/10, dem_np)
    else:
        dem_np = np.asarray(dem)
        return np.ma.masked_where(dem_np < no_data/10, dem_np)

def normalizeRela(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    return (arr-arr_mean)/arr_std

def normalize01(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def slicePic(pic, cut_width, cut_length):
    sliced_pic = []
    width = pic.shape[0]
    length = pic.shape[1]
    num_width = int(width/cut_width)
    num_length = int(length/cut_length)
    for i in range(0, num_width):
        for j in range(0, num_length):
            # dtm_ = dtm[i*cut_width:(i+1)*cut_width, j*cut_length:(j+1)*cut_length]
            # dsm_ = dsm[i*cut_width:(i+1)*cut_width, j*cut_length:(j+1)*cut_length]
            pic_ = pic[i*cut_width:(i+1)*cut_width, j *
                       cut_length:(j+1)*cut_length]
            # mean_pic_ = np.mean(pic_[:, :, 0])
            # std_pic_ = np.std(pic_[:, :, 0])
            # pic_[:, :, 0] = (pic_[:, :, 0]-mean_pic_)/std_pic_
            sliced_pic.append(pic_)
    return sliced_pic


def sliceRisPic(pic, cut_width, cut_length, dt=False):
    dt_sliced_pic = {}
    ls_sliced_pic = []
    width = pic.data.shape[0]
    length = pic.data.shape[1]
    num_width = math.ceil(width/cut_width)
    num_length = math.ceil(length/cut_length)
    count = 0
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic_ = np.zeros((cut_width, cut_length))
            key_ = ''
            if i <= num_width-2 and j <= num_length-2:
                pic_ = pic[i*cut_width:(i+1)*cut_width,
                           j * cut_length:(j+1)*cut_length]
                key_ = str(i*cut_width)+':'+str((i+1)*cut_width) + \
                    ','+str(j*cut_length)+':'+str((j+1)*cut_length)
                # sliced_pic[key_]=pic_
            elif i <= num_width-2 and j == num_length-1:
                pic_ = pic[i*cut_width:(i+1)*cut_width,
                           length-cut_length:length]
                key_ = str(i*cut_width)+':'+str((i+1)*cut_width) + \
                    ','+str(length-cut_length)+':'+str(length)
                # sliced_pic[key_]=pic_
            elif i == num_width-1 and j <= num_length-2:
                pic_ = pic[width-cut_width:width,
                           j * cut_length:(j+1)*cut_length]
                key_ = str(width-cut_width)+':'+str(width)+',' + \
                    str(j*cut_length)+':'+str((j+1)*cut_length)
            else:
                pic_ = pic[width-cut_width:width, length-cut_length:length]
                key_ = str(width-cut_width)+':'+str(width)+',' + \
                    str(length-cut_length)+':'+str(length)
                # sliced_pic[key_]=pic_
            # sliced_pic[key_]=pic_
            ls_sliced_pic.append(pic_)
            if dt:
                dt_sliced_pic[count]=key_
                count+=1
    if not dt:
        return ls_sliced_pic
    else:
        return ls_sliced_pic, dt_sliced_pic

def sliceRisPicOL(pic, cut_height, cut_width, olp=0.15, dt=False):
    dt_sliced_pic = {}
    ls_sliced_pic = []

    height = pic.shape[0]
    width = pic.shape[1]

    steph = cut_height - math.floor(cut_height*olp)
    stepw = cut_width - math.floor(cut_width*olp)
    
    lasth = height - cut_height
    lastw = width - cut_width

    hOffsets = list(range(0, lasth + 1, steph))
    wOffsets = list(range(0, lastw + 1, stepw))

    if len(hOffsets) == 0 or hOffsets[-1] != lasth:
        hOffsets.append(lasth)
    if len(wOffsets) == 0 or wOffsets[-1] != lastw:
        wOffsets.append(lastw)

    # img_patches = []
    # indices = []

    count = 0

    for hOffset in hOffsets:
        for wOffset in wOffsets:
            ls_sliced_pic.append(
                pic[(slice(hOffset, hOffset + cut_height), slice(wOffset, wOffset + cut_width))]
            )
            key_ = str(hOffset)+':'+str(hOffset + cut_height)+',' + \
                    str(wOffset)+':'+str(wOffset + cut_width)
            # indices.append((yOffset, yOffset + windowSizeY, xOffset, xOffset + windowSizeX))
            dt_sliced_pic[count]=key_
            count+=1

    if dt:
        return ls_sliced_pic, dt_sliced_pic
    else:
        return ls_sliced_pic




def augmentateData(pics):
    pics_augmentated = []
    for pic in pics:
        pic_rot90 = np.rot90(pic)
        pic_rot180 = np.rot90(pic, 2)
        pic_rot270 = np.rot90(pic, 3)
        pic_flip = np.fliplr(pic)
        pic_flip_rot90 = np.rot90(pic_flip)
        pic_flip_rot180 = np.rot90(pic_flip, 2)
        pic_flip_rot270 = np.rot90(pic_flip, 3)
        pics_augmentated += [pic_rot90, pic_rot180, pic_rot270,
                             pic_flip, pic_flip_rot90, pic_flip_rot180, pic_flip_rot270]
    return pics_augmentated

def generateGroundTruth(dsm, dtm, thr=2.0):
    length = dsm.shape[0]
    width = dsm.shape[1]
    mask_ = np.zeros((2, length, width), dtype=bool)
    for i in range(length):
        for j in range(width):
            mask_[0][i][j] = True
            # if dsm.mask[i][j] == False:
            #     if dtm.mask[i][j] == True:
            #         mask_[1][i][j] = True
            #         mask_[0][i][j] = False
            #     elif dsm.data[i][j] - dtm.data[i][j] > thr:
            #         mask_[1][i][j] = True
            #         mask_[0][i][j] = False
    mask_[0][(dsm.mask == False) & (dtm.mask == True)] = False
    mask_[1][(dsm.mask == False) & (dtm.mask == True)] = True
    mask_[0][(dsm.mask == False) & (dtm.mask == False) & (dsm.data - dtm.data > thr)] = False
    mask_[1][(dsm.mask == False) & (dtm.mask == False) & (dsm.data - dtm.data > thr)] = True
    return mask_



    # mask_[0][(dsm.mask == False) & (dtm.mask == True)] = False
    # mask_[1][(dsm.mask == False) & (dtm.mask == True)] = True
    # mask_[0][(dsm.mask == True) & (dsm.data - dtm.data > thr)] = False
    # mask_[1][(dsm.mask == True) & (dsm.data - dtm.data > thr)] = True
def getNormalParams(lt):
    ltar = np.asarray(lt, dtype=np.float64)
    mean_ar = np.mean(ltar)
    std_ar = np.std(ltar)
    return mean_ar, std_ar

def stackPics(dsm_fill, slope, layer, normalization=False):
    pics = []
    for n in range(len(dsm_fill)):
        if normalization:
            pics.append(np.stack((normalize01(dsm_fill[n]), normalize01(
                slope[n]), normalize01(layer[n])), axis=2))
        else:
            pics.append(np.stack((dsm_fill[n], slope[n], layer[n]), axis=2))
    arr_pics = np.stack(pics)
    arr_pics = arr_pics.astype(np.float32)
    return arr_pics

def mergeImg(im_list, im_dict, mask=False):
    height = int(im_dict[list(im_dict)[-1]].split(',')[0].split(':')[1])
    width = int(im_dict[list(im_dict)[-1]].split(',')[1].split(':')[1])
    im_merged = np.zeros((height, width), dtype=np.float32)
    if not mask:
        ndv = 3.4028230607370965e+38
        im_merged = im_merged-ndv
    else:
        im_merged = im_merged+0.5
    for k in im_dict:
        h0 = int(im_dict[k].split(',')[0].split(':')[0])
        h1 = int(im_dict[k].split(',')[0].split(':')[1])
        w0 = int(im_dict[k].split(',')[1].split(':')[0])
        w1 = int(im_dict[k].split(',')[1].split(':')[1])
        if not mask:
            im_merged[h0:h1, w0:w1] = np.where(
                im_list[k] > im_merged[h0:h1, w0:w1], im_list[k], im_merged[h0:h1, w0:w1])
        else:
            im_merged[h0:h1, w0:w1] = np.where(abs(
                im_list[k]-0.5) > abs(im_merged[h0:h1, w0:w1]-0.5), im_list[k], im_merged[h0:h1, w0:w1])
    return im_merged

