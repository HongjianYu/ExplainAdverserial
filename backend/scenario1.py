from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/topk_search', methods=['POST'])
def topk_search():
    data = request.json
    k = data.get('k')
    roi = 'True' if data.get('roi') == 'object bounding box' else 'False'
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    order = data.get('order')
    
    query_command = f"""
    SELECT mask_id
    FROM MasksDatabaseView
    ORDER BY CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {order}
    LIMIT {k};
    """
    
    # Dummy implementation to return the query command and some mock image IDs
    image_ids = list(range(1, int(k) + 1))
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
    app.run(debug=True)
