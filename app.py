from flask import Flask, render_template, request, url_for
import os
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'  # Updated path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form.get('model')
    uploaded_file = request.files['file']
    
    print("\nSelected Model: " +str(selected_model))
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    # Open the image using PIL
    image = Image.open(file_path)

    if selected_model == 'model2':
        # Instantiate the feature extractor specific to the model checkpoint
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # Process the image using the feature extractor
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])

        # Render the prediction result with the uploaded image
        return render_template('result.html', model=model.config.id2label[predicted_class_idx])#, image_path=url_for('static', filename='uploads/' + uploaded_file.filename))
    else:
        # Load a pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Run inference on the uploaded image
        results = model(image, verbose=False) 

        # Save YOLO result as image
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'yolo_result.png')
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save(result_image_path)  

        # Render the result page with the YOLO result image
        return render_template('result.html', model="Yolo Model", image_path=url_for('static', filename='uploads/yolo_result.png'))


if __name__ == '__main__':
    app.run(debug=True)
