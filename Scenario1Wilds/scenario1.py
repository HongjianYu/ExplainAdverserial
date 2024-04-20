from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import sys
sys.path.append("/Users/lindseywei/MaskSearch/wilds")
from topk import *
from s1_data_process import data_process
app = Flask(__name__)
CORS(app)


@app.route('/api/topk_search', methods=['POST'])


def topk_search():
    data = request.json
    k = data.get('k')
    k = int(k)
    roi = 'True' if data.get('roi') == 'object bounding box' else 'False'
    region = 'object' if roi == 'True' else 'custom'
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    lv = float(pixel_lower_bound)
    uv = float(pixel_upper_bound)
    order = data.get('order')
    reverse = False if order == 'DESC' else True
    
    query_command = f"""
    SELECT mask_id
    FROM MasksDatabaseView
    ORDER BY CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {order}
    LIMIT {k};
    """
    
    # Dummy implementation to return the query command and some mock image IDs
    hist_size = 16
    bin_width = 256 // hist_size
    cam_size_y = 448
    cam_size_x = 448
    region_area_threshold = 5000
    available_coords = 64
    id_val_data, ood_val_data, label_map, pred_map, cam_map, object_detection_map, dataset_examples, in_memory_index_suffix, image_access_order = data_process()
    print("hey")
    count, images = get_max_area_in_subregion_in_memory_version(
    "wilds",
    (id_val_data, ood_val_data),
    label_map,
    pred_map,
    cam_map,
    object_detection_map,
    bin_width,
    cam_size_y,
    cam_size_x,
    hist_size,
    dataset_examples,
    lv,
    uv,
    region,
    in_memory_index_suffix,
    image_access_order,
    early_stoppable=False,
    k=k,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    reverse=reverse,
    visualize=True,
    available_coords=available_coords,
    compression=None,
    )

    image_ids = list(range(0, k))
    return jsonify({'query_command': query_command, 'image_ids': image_ids})

@app.route('/api/filter_search', methods=['POST'])
def filter_search():
    data = request.json
    threshold = data.get('threshold')
    roi = 'True' if data.get('roi') == 'object bounding box' else 'False'
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    comparison = data.get('thresholdDirection')
    
    query_command = f"""
    SELECT mask_id
    FROM MasksDatabaseView
    WHERE CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {comparison} {threshold};
    """
    
    # Dummy implementation to return the query command and some mock image IDs
    image_ids = list(range(1, 10))  # Mock 6 images for the filter results
    return jsonify({'query_command': query_command, 'image_ids': image_ids})

@app.route('/topk_results/<filename>')
def topk_image(filename):
    return send_from_directory('topk_results', filename)

@app.route('/filter_results/<filename>')
def filter_image(filename):
    return send_from_directory('filter_results', filename)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=8000)
