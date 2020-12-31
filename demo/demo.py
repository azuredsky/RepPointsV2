from mmdet.apis import init_detector, inference_detector
import mmcv, os, time
config_file = 'configs/reppoints_v2/reppoints_v2_r50_fpn_giou_mstrain_2x_coco.py'
checkpoint_file = 'ckpt/reppoints_v2_r50_fpn_giou_mstrain_2x_coco-889d053a.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

save_path = '11test_result'
local_file = os.listdir('test_images/')
for i in range(len(local_file)):
    img = 'test_images/'+local_file[i]  # or img = mmcv.imread(img), which will only load it once
    print(f'img : {img}')
    result = inference_detector(model, img)
    print(result)
    model.show_result(img, result, score_thr=0.5, thickness=1, font_scale=0.5, out_file=save_path+'/result_'+local_file[i])

# #运行demo_image.py文件即可以进行文件夹内图片的检测
#
# if 你觉得原版结果可以:
#     pass
#  else:
#     #修改下原版的mmcv.imshow_det_bboxes()画图
#     ####这里先用python -c "import mmcv;print(mmcv)"找到mmcv安装包的位置
#
#     #结果中<module 'mmcv' from '/home/deployer/anaconda3/lib/python3.7/site-packages/mmcv-0.6.2-py3.7-linux-x86_64.egg/mmcv/__init__.py'>
#     /home/deployer/anaconda3/lib/python3.7/site-packages/mmcv-0.6.2-py3.7-linux-x86_64.egg/mmcv就是mmcv的安装包，再运行如下语句进行画图修改
#     vi /home/deployer/anaconda3/lib/python3.7/site-packages/mmcv-0.6.2-py3.7-linux-x86_64.egg/mmcv/visualization/image.py
#     找到80行的imshow_det_bboxes函数，将125行后面修改为下面的代码：
#     tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     for bbox, label in zip(bboxes, labels):
#         tf = max(tl - 1, 1)
#         bbox_int = bbox.astype(np.int32)
#         left_top = (bbox_int[0], bbox_int[1])
#         right_bottom = (bbox_int[2], bbox_int[3])
#         cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=tl, lineType=cv2.LINE_AA)
#         label_text = class_names[
#             label] if class_names is not None else f'cls {label}'
#         if len(bbox) > 4:
#             label_text += f'|{bbox[-1]:.02f}'
#         t_size = cv2.getTextSize(label_text, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = left_top[0] + t_size[0], left_top[1] - t_size[1] - 3
#         cv2.rectangle(img, left_top, c2, bbox_color, -1, cv2.LINE_AA)
#         cv2.putText(img, label_text, (left_top[0], left_top[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#
#     vi mmdet/models/detectors/base.py
#     #修改这个文件夹下的292行的show_result函数的bbox_color和text_color，固定框子颜色
#     bbox_color=(234, 71, 87),
#     text_color=(141, 97, 71),
