# ----------
# Generate EM data masks
# ----------

# ----------
# Configuration
# ----------

import numpy as np
import os
import pandas as pd
import sys
from cloudvolume import CloudVolume, Bbox
from PIL import Image

# Define the group size for this piecewise export
groupsize = 100

# Get the IDs to process in this run
start_idx = int(sys.argv[1])
end_idx = start_idx + groupsize


# Define desired image chunk size in px^3
chunksize = 256

# Define center pixel coordinates given chunk size
xc_px = yc_px = zc_px = int(chunksize/2)

filelist = '../resources/image_chunks_8_8_40_filelist'

synid_list = []
with open(filelist,'r') as f:
    lines = f.readlines()
    for line in lines:
        synid_curr = np.uint32(line[line.rfind('/')+1:line.rfind('.npy')])
        synid_list.append(synid_curr)
        
# Pull the full list of synapses and remove verified and unverified synapse IDs from above from this table
all_syns = pd.read_csv('../resources/pni_microns_public_all_synapses_v185.csv')

# Key: 0: Background
# 1: Presynaptic cell
# 2: Synaptic cleft
# 3: Postsynaptic cell

# Define mip level chosen for export (mip 1, res (x,y,z) = (8,8,40))
mip = 1

# Define v185 segmentation at chosen mip level
# mip0 = (8,8,40) nm^3/vx
segvol = CloudVolume('gs://microns_public_datasets/pinky100_v185/seg', mip=(mip-1), parallel=True, progress=True) # or mip-1

# Define synaptic cleft segmentation at chosen mip level
# mip0 = (8,8,40) nm^3/vx
cleftvol = CloudVolume('gs://neuroglancer/pinky100_v0/clefts/mip1_d2_1175k', mip=(mip-1), parallel=True, progress=True)

# ----------
# Function definitions
# ----------

def gen_test_img_from_mask(nparr,targetfname):
    # Takes a numpy array, with mask values 1 through 3
    # and scales them up for easier viewing as embedded in a 256-bit
    # image.
    imgcurr = np.uint8(np.divide(nparr,np.max(nparr))*255)
    img = Image.fromarray(imgcurr)
    img.save(targetfname)
    
def pull_mask_chunks(synid,cleftcv,segcv,chunksize,targetdir,is_img_test=0):
    # Download an image chunk of size @chunksize centered on the synaptic cleft
    # with segment ID @ synid
    # Pull masks for the presynaptic cell (label 1) and postsynaptic cell (label 3)
    # from the segmentation layer @segcv, and
    # pull the mask for the synaptic cleft (label 2) from the synaptic cleft layer
    # @cleftcv
    # Save the summed mask as a .npy file in @targetdir    
    # Currently, all_syns and mip are treated like globals.
    
    # Define label maps
    presyn_label = 1
    cleft_label = 2
    postsyn_label = 3
    
    synrow = all_syns.loc[all_syns['id'] == synid]    
    
    # Get pre- and postsynaptic seg IDs for seg pulls
    preid = list(synrow['pre_root_id'])[0]
    postid = list(synrow['post_root_id'])[0]
    
    print(synid,preid,postid)
    print(type(synid),type(preid),type(postid))
    
    # Get synapse centroid at full resolution ((x,y,z) = (4,4,40) nm3/vx)
    x0 = list(synrow['ctr_pos_x_vx'])[0]
    y0 = list(synrow['ctr_pos_y_vx'])[0]
    z0 = list(synrow['ctr_pos_z_vx'])[0]

    # Define bbox for export
    xc = x0/(2**mip)
    yc = y0/(2**mip)
    
    xtl = int(xc - chunksize/2)
    xbr = int(xc + chunksize/2)
    ytl = int(yc - chunksize/2)
    ybr = int(yc + chunksize/2)
    if is_img_test:
        ztl = int(z0)
        zbr = int(z0 + 1)
        px_center = np.asarray([xc_px,yc_px])
    else:
        ztl = int(z0 - chunksize/2)
        zbr = int(z0 + chunksize/2)
        px_center = np.asarray([xc_px,yc_px,zc_px])
        
    # Get volume for pre- and postsynaptic cell masks
    succeeded = 1
    try:
        segvol_dl = segcv.download(Bbox([xtl,ytl,ztl],[xbr,ybr,zbr]))
    except:
        succeeded = 0
    # Get volume for cleft mask
    try:
        cleftvol_dl = cleftcv.download(Bbox([xtl,ytl,ztl],[xbr,ybr,zbr]))
    except:
        succeeded = 0
   
    if succeeded:
        # Generate synaptic cleft mask
        cleftvol = np.squeeze(np.asarray(cleftvol_dl))
        # Map the synaptic cleft ID to the one used in this version of the synapse
        # segmentation layer (same segments, different seg IDs)
        # Do this by identifying the cleft closest to the synapse centroid
        cleft_pxs = np.argwhere(cleftvol)
        deltas = np.asarray([np.linalg.norm(q-px_center) for q in cleft_pxs])
        closest_cleft = cleft_pxs[np.argmin(deltas)]
        mapped_cleftid = cleftvol[tuple(closest_cleft)]
        cleft_mask = np.uint8((cleftvol == mapped_cleftid))
        
        # Generate pre- and postsynaptic partner masks        
        segvol = np.squeeze(np.asarray(segvol_dl))
        presyn_mask = np.uint8((segvol == preid))
        postsyn_mask = np.uint8((segvol == postid))

        # Combine masks (adding cleft labels last so they persist)
        mask = postsyn_mask * postsyn_label
        mask[presyn_mask.nonzero()] = presyn_label
        mask[cleft_mask.nonzero()] = cleft_label

        if is_img_test:
            # Print test masks
            gen_test_img_from_mask(cleft_mask,os.path.join(targetdir,'testimg_cleft.png'))
            gen_test_img_from_mask(presyn_mask,os.path.join(targetdir,'testimg_presyn.png'))
            gen_test_img_from_mask(postsyn_mask,os.path.join(targetdir,'testimg_postsyn.png'))
            gen_test_img_from_mask(mask,os.path.join(targetdir,'test_composite.png'))
        else:
            np.save(os.path.join(targetdir,'{0}_mask.npy'.format(synid)),mask)

# ----------
# Run export
# ----------

targdir = '../resources/img_chunk_masks_8_8_40/'
for sidx in np.arange(start_idx,end_idx):
    synid = synid_list[sidx]
    pull_mask_chunks(synid,cleftvol,segvol,chunksize,targdir,is_img_test=0)


