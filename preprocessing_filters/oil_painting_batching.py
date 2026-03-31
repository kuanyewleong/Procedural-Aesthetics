import cv2
import random
import os
import glob
import numpy as np

SRC_DIR = "data/flowers-102/jpg"
DST_DIR = "data/flowers-102/jpg_oil"

os.makedirs(DST_DIR, exist_ok=True)

# Grab all jpg/JPG/jpeg/JPEG files in the source folder
img_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
    img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"No images found in: {SRC_DIR}")


#Kuwahara filter implementation
def kuwahara(x,a):
# x = image on which filter is used, (2a + 1) = size of window
  a = int(a)
  n,m = x.shape
  x_kuw = np.zeros((n,m))
  for i in range(0,n):
    for j in range(0,m):
      quad_1 = []
      quad_2 = []
      quad_3 = []
      quad_4 = []
      for dx in np.arange(-a,1):
        for dy in np.arange(-a,1):
          if 0<=i+dx<n and 0<=j+dy<m:
              quad_1.append(x[i+dx,j+dy]) 
      quad_1 = np.array(quad_1)
      m1 = np.mean(quad_1)
      v1 = np.var(quad_1)
      
      
      for dx in np.arange(0,a+1):
        for dy in np.arange(-a,1):
          if 0<=i+dx<n and 0<=j+dy<m: 
              quad_2.append(x[i+dx,j+dy])
      quad_2 = np.array(quad_2)
      m2 = np.mean(quad_2)
      v2 = np.var(quad_2)

      for dx in np.arange(-a,1):
          for dy in np.arange(0,a+1):
              if 0<=i+dx<n and 0<=j+dy<m: quad_3.append(x[i+dx,j+dy])
      quad_3 = np.array(quad_3)
      m3 = np.mean(quad_3)
      v3 = np.var(quad_3)

      for dx in np.arange(0,a+1):
          for dy in np.arange(0,a+1):
              if 0<=i+dx<n and 0<=j+dy<m: quad_4.append(x[i+dx,j+dy])
      quad_4 = np.array(quad_4)
      m4 = np.mean(quad_4)
      v4 = np.var(quad_4)
      
      variance_list = [v1,v2,v3,v4] 
      mean_list = [m1,m2,m3,m4]
      minimum = min(variance_list)
      pos = variance_list.index(minimum)
      x_kuw[i,j] = mean_list[pos] 
  return x_kuw

# batching
saved = 0
failed = 0

for in_path in img_paths:
    image = cv2.imread(in_path)
    if image is None:
        print(f"[WARN] Failed to read: {in_path}")
        failed += 1
        continue
    
    #applying kuwahara filter to the B, G and R components individually
    b_kuw = kuwahara(image[:,:,0],6)
    g_kuw = kuwahara(image[:,:,1],6)
    r_kuw = kuwahara(image[:,:,2],6)

    #merging R, G and B components
    out_img = np.dstack((b_kuw,g_kuw,r_kuw))

    # Keep the original filename, save into destination folder
    base_name = os.path.basename(in_path)
    out_path = os.path.join(DST_DIR, base_name)

    ok = cv2.imwrite(out_path, out_img)
    if not ok:
        print(f"[WARN] Failed to write: {out_path}")
        failed += 1
        continue

    saved += 1
    if saved % 100 == 0:
        print(f"Saved {saved}/{len(img_paths)}...")

print(f"Done. Saved: {saved}, Failed: {failed}, Total: {len(img_paths)}")
print(f"Output folder: {DST_DIR}")

