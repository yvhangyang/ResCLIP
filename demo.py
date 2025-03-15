import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from resclip_segmentor import ResCLIPSegmentation
import cv2
import numpy as np
import sys

sys.path.append("..")

img = Image.open('./images/demo.jpg')


name_list = ['aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
             'sofa', 'train', 'tvmonitor']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = ResCLIPSegmentation(
    clip_path='ViT-B/16',  # ViT-B/16 ViT-L/14 ViT-B/32
    name_path='./configs/my_name.txt',
    prob_thd=0.35,  # need to adjust if background is given
    temp_thd=0.10,
    pamr_steps=10,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.6],
    attn_sfr_weights=[2.1, 0.6],
    use_rcs=True,
    use_sfr=True,
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

img_np = np.array(img)

num_classes = len(name_list)
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
colors = (colors[:, :3] * 255).astype(np.uint8)

seg_pred_color = np.zeros((*seg_pred.shape, 3), dtype=np.uint8)
for i in range(num_classes):
    seg_pred_color[seg_pred == i] = colors[i]

if img_np.shape[:2] != seg_pred_color.shape[:2]:
    seg_pred_color = cv2.resize(seg_pred_color, (img_np.shape[1], img_np.shape[0]))

img_np = img_np.astype(np.uint8)
seg_pred_color = seg_pred_color.astype(np.uint8)

superimposed_img = cv2.addWeighted(img_np, 0.3, seg_pred_color, 0.7, 0.0)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(seg_pred_color)
plt.title('Segmentation Prediction')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title('Superimposed Image')
plt.axis('off')

plt.tight_layout()
# plt.savefig('./images/seg_pred/demo.jpg', bbox_inches='tight')
plt.show()
plt.close()