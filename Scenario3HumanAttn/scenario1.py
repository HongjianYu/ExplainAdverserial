from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
import sys
import json
import os
sys.path.append("/Users/edwardyeung/Sal/MaskSearchDemo-main/Scenario1Wilds")
from topk import *
app = Flask(__name__)
CORS(app)
#INTERATION WORKS

@app.route('/api/topk_search', methods=['POST'])
def topk_search():
    data = request.json
    k = data.get('k')
    enable = data.get('ms')
    k = int(k)
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    order = data.get('order')
    reverse = True if order == 'DESC' else False

    
    query_command = f"""
    SELECT mask_id,
    CP(intersect(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) 
    / CP(union(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) as iou 
    FROM MasksDatabaseView WHERE mask_type IN (1, 2)
    GROUP BY image_id ORDER BY iou {order} LIMIT {k};
    """
    start = time.time()
    cam_size_x = 384
    cam_size_y = 384
    hist_size = 2
    bin_width = 256 // hist_size
    total_images = 11788
    examples = np.arange(1, 11788)
    available_coords = 16
    lv = 0.0
    uv = 1.0
    region = (0, 0, 384, 384)
   
    count, images = get_max_IoU_across_masks_in_memory(
        cam_size_y=384,
        cam_size_x=384,
        bin_width=bin_width,
        hist_size=2,
        examples=examples,
        lv=lv,
        uv=uv,
        in_memory_index_suffix_in=in_memory_index_suffix_in,
        in_memory_index_suffix_un=in_memory_index_suffix_un,
        region=region,
        k=k,
        region_area_threshold=0,
        ignore_zero_area_region=True,
        reverse=reverse,
        available_coords=available_coords,
        compression=None,
    )
    print("here")
    print(count)
    image_ids = [int(image_idx) for (metric, image_idx) in images]
    end = time.time()
    time_used = end - start
    print("time1: ", time_used)
    print(image_ids)
    execution_time = round(time_used, 3)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': execution_time, 'images_count': len(image_ids)})


@app.route('/api/augment', methods=['POST'])
def augment():
    data = request.json
    img_ids = data.get('image_ids')
    # Now you can use img_ids for augmentation
    print(img_ids)
    return jsonify({'image_ids': img_ids})



@app.route('/api/filter_search', methods=['POST'])
def filter_search():
    print("round")
    data = request.json
    threshold = data.get('threshold')
    enable = data.get('ms')
    v = float(threshold)
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    comparison = data.get('thresholdDirection')
    reverse = True if comparison == '<' else False

    query_command = f"""
    SELECT mask_id,
    CP(intersect(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) 
    / CP(union(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) as iou 
    FROM MasksDatabaseView WHERE iou {comparison} {threshold}, mask_type IN (1, 2)
    GROUP BY image_id;
    """
    
    # query_command = f"""
    # SELECT mask_id
    # FROM MasksDatabaseView
    # WHERE CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {comparison} {threshold};
    # """
    start = time.time()
    cam_size_x = 384
    cam_size_y = 384
    hist_size = 2
    bin_width = 256 // hist_size
    total_images = 11788
    examples = np.arange(1, 11788)
    available_coords = 16
    lv = 0.0
    uv = 1.0
    region = (0, 0, 384, 384)
   

    count, images = get_Filter_IoU_across_masks_in_memory(
        cam_size_y=384,
        cam_size_x=384,
        bin_width=bin_width,
        hist_size=2,
        examples=examples,
        lv=lv,
        uv=uv,
        in_memory_index_suffix_in=in_memory_index_suffix_in,
        in_memory_index_suffix_un=in_memory_index_suffix_un,
        region=region,
        v=v,
        region_area_threshold=0,
        ignore_zero_area_region=True,
        reverse=reverse,
        available_coords=available_coords,
        compression=None,
    )
    num = 0
    if(len(images)>50): 
        num = 50
        print("haha")
    else: 
        num = len(images)
    image_ids = [int(image_idx) for (metric,image_idx) in images[:num]]
    print("there")
    #image_ids = np.array(map(str, image_ids))
    end = time.time()
    time_used = end - start
    print("time2: ", time_used)
    execution_time = round(time_used, 3)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time' : execution_time, 'images_count': 11788 - count})

BASE_DIR = "/Users/edwardyeung/Sal/MaskSearchDemo-main/GUI/backend"

@app.route('/saliency_images/<filename>')
def topk_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'saliency_images'), filename)

@app.route('/human_att_images/<filename>')
def filter_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'human_att_images'), filename)

@app.route('/intersect_visualization/<filename>')
def intersect_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'intersect_visualization'), filename)

@app.route('/union_visualization/<filename>')
def union_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'union_visualization'), filename)



if __name__ == '__main__':
    #app.run(debug=True)
    
    in_memory_index_suffix_in = np.load(
        f"/Users/edwardyeung/Sal/intersect_index.npy"
    )
    in_memory_index_suffix_un = np.load(
        f"/Users/edwardyeung/Sal/union_index.npy"
    )
    app.run(port=8000)