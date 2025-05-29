import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import torch
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# from CODINO.mmdet.apis import init_detector, inference_detector, show_result_pyplot

from sam2.build_sam import build_sam2_video_predictor
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from utils.drawers import *
from utils.pathcfg import *
from utils.videowriter import *
from sam2mot.TMS import TMS
from sam2mot.CI import CI

# 模型和数据路径
detector_checkpoint = DetectorCheckpoint
detector_cfg = DetectorCfg
sam2_checkpoint = Sam2Checkpoint # SAM2 checkpoint路径
sam2_model_cfg = Sam2Cfg # SAM2模型cfg路径

track_id_to_obj_id = {}

# def trim_old_frames(inference_state, frame_id, keep_last_n=400):
#     for obj_id in inference_state["obj_ids"]:
#         obj_idx = inference_state["obj_id_to_idx"][obj_id]
#         for k in ["cond_frame_outputs", "non_cond_frame_outputs"]:
#             frame_dict = inference_state["output_dict_per_obj"][obj_idx].get(k, {})
#             keys_to_delete = [fid for fid in frame_dict if fid < frame_id - keep_last_n]
#             for fid in keys_to_delete:
#                 del frame_dict[fid]


def sam2mot(video_dir, output_path, video_save_dir):
    device = torch.device('cuda:0')
    # 设置高性能矩阵运算
    if device.type=='cuda' and torch.cuda.get_device_properties(0).major>=8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 扫描视频帧名的列表并排序
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    # 初始化各个模块
    # detector
    detector = load_model(model_config_path=DetectorCfg, model_checkpoint_path=DetectorCheckpoint, device=device)

    TEXT_PROMPT = "person"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    # SAM2
    predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=video_dir) # 初始化

    # TMS
    tms = TMS()
    # CI
    ci = CI()
    results = []
    detector_results = {}
    # Grounding DINO获取每帧上的bbox
    for frame_id, frame_name in enumerate(frame_names):
        print(f'===================frame:{frame_id}=======================')
        torch.cuda.empty_cache()
        # step 1: detector检测
        IMAGE_PATH = f'{video_dir}/{frame_name}'
        image_source, image = load_image(IMAGE_PATH)
        with torch.no_grad():
            boxes, logits, _ = predict(
                model=detector,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            detector_results[frame_id] = [boxes, logits]

    del detector

    for frame_id, frame_name in enumerate(frame_names):
        print(f'===================frame:{frame_id}=======================')
        if frame_id == 325:
            print("Danger!")
        torch.cuda.empty_cache()
        det_bboxes = detector_results[frame_id][0]
        det_logits = detector_results[frame_id][1]
        if frame_id>0: # 不是第一帧，进行SAM2 track
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for _, _, _ in predictor.propagate_in_video(inference_state, start_frame_idx=frame_id, max_frame_num_to_track=0):
                    print('-----propogating-----')

        # step 2: 检测数据和inference_state传入TMS
        new_prompts, obj_ids, removed = tms.TrajectroyManage(frame_id=frame_id, inference_state=inference_state, det_bboxes=det_bboxes, det_logits=det_logits, device=torch.device('cuda:1'))

        print(f'new_prompts:{new_prompts}\nobj_ids:{obj_ids}\nremoved:{removed}')

        # step 3: SAM2 进行Tracking
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if removed: # 有需要删除的obj_id
                for obj_id in removed:
                    predictor.remove_object(inference_state=inference_state, obj_id=obj_id)
            if len(new_prompts)>0: # 有需要添加的prompts
                for i, obj_id in enumerate(obj_ids): # 输入box prompts
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_id,
                        obj_id=obj_id,
                        box=np.array(new_prompts[i], dtype=np.float32),
                    )

                for _, _, _ in predictor.propagate_in_video(inference_state, start_frame_idx=frame_id, max_frame_num_to_track=0):
                    print('-----propogating-----')

        # step 4: tck_masks和tck_logits传入CI
        occluded = ci.remove_occlusion(inference_state=inference_state, frame_id=frame_id, device= torch.device('cuda:1'))
        print(f'occluded:{occluded}')
        
        if occluded: # 删除被遮挡的
            for obj_id in occluded:
                if obj_id not in inference_state["obj_id_to_idx"]: # 跳过无效obj_id
                    continue  # 保险：跳过无效 obj_id
                obj_idx = inference_state["obj_id_to_idx"][obj_id]
                out_dict = inference_state["output_dict_per_obj"][obj_idx]
                if frame_id in out_dict['non_cond_frame_outputs'].keys():
                    out_dict["non_cond_frame_outputs"][frame_id]["object_score_logits"] = torch.tensor([[-1.0]], device=device, dtype=torch.bfloat16)
                # if frame_id in out_dict.get("non_cond_frame_outputs", {}):
                #     out_dict["non_cond_frame_outputs"][frame_id]["object_score_logits"] = -1.0

        print(f'===========保存+绘制frame{frame_id}======================')
        tck_ids, _, tck_bboxes, _, tck_masks = tms.get_trackinfo(frame_idx=frame_id, inference_state=inference_state, device=torch.device('cuda:1'))

        frame_path = os.path.join(video_dir, frame_names[frame_id])
        image = np.array(Image.open(frame_path).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for i in range(len(tck_ids)):
            obj_id = tck_ids[i]
            bbox = tck_bboxes[i]
            mask = tck_masks[i].cpu().numpy().astype(np.uint8)
            bb_left = bbox[0]
            bb_top = bbox[3]
            bb_width = bbox[2] - bbox[0]
            bb_height = bbox[3] - bbox[1]
            res = [frame_id+1, obj_id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1]
            results.append(res)
            with open(output_path, "a") as f:
                line = ','.join(map(str, res)) + '\n'
                f.write(line)

            image = show_mask(image, mask, obj_id=obj_id)
            image = show_box(image=image, box=bbox, obj_id=obj_id)

        save_path = os.path.join(video_save_dir, frame_names[frame_id])
        cv2.imwrite(save_path, image)

    # # 写入文件
    # print(f'==========写入文件======================')
    # with open(output_path, "w") as f:
    #     for row in results:
    #         line = ','.join(map(str, row)) + '\n'
    #         f.write(line)
    # print(f'==========写入文件完成======================')

    # 制作视频
    create_video_from_images(image_folder=video_save_dir, output_video_path=video_save_dir, frame_rate=20)
    # 释放 inference_state 占用的 GPU 显存
    del inference_state
    del tms
    del ci
    torch.cuda.empty_cache()


    