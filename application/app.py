import io
import os
import cv2
import uuid
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from flask import Flask, request, render_template, send_file, url_for, jsonify, redirect

app = Flask(__name__)

# Configuration des dossiers
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(app.root_path, 'static', 'results')
app.config['ANNOTATION_FOLDER'] = os.path.join(app.root_path, 'static', 'annotations')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATION_FOLDER'], exist_ok=True)

# Chargement du modèle YOLO
model_path = os.path.join(app.root_path, 'models', 'best.pt')
model = YOLO(model_path)

# Initialiser un dataframe pour stocker les prédictions
data_columns = ["Image ID", "Component", "Area", "Void %", "Max Void %"]
predictions_df = pd.DataFrame(columns=data_columns)

# Chargement du modèle SAM
sam_checkpoint = os.path.join(app.root_path, "models", "sam_vit_h_4b8939.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predictions_df
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        image_id = str(uuid.uuid4())
        original_filename = file.filename
        # Enregistrer le fichier téléchargé
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)
        
        # Exécuter la prédiction
        results = model.predict(
            source=original_path, 
            save=True, 
            conf=0.05,
            project=app.config['RESULT_FOLDER'],
            name='saves',
            exist_ok=True
        )
        
        # L'image annotée est enregistrée dans 'static/results/saves'
        annotated_image_path = os.path.join(app.config['RESULT_FOLDER'], 'saves', original_filename)
        print(f"Annotated image path: {annotated_image_path}")
        
        # Extraire les informations pour le tableau
        result_data = results[0].boxes.data  # Accéder aux données des boîtes de détection
        component_count = len(result_data)  # Nombre de composants détectés
        
        img = Image.open(original_path)
        img_area = img.size[0] * img.size[1]
        
        if component_count > 0:
            result_data = result_data.cpu().numpy()  # Convertir le tenseur en tableau numpy
            x1 = result_data[:, 0]
            y1 = result_data[:, 1]
            x2 = result_data[:, 2]
            y2 = result_data[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            total_area = areas.sum()
            void_percentage = (total_area / img_area) * 100
            max_void_percentage = (areas.max() / img_area) * 100

            # Arrondir à deux chiffres après la virgule
            total_area = round(total_area, 2)
            void_percentage = round(void_percentage, 2)
            max_void_percentage = round(max_void_percentage, 2)
        else:
            # Aucun composant détecté
            total_area = 0
            void_percentage = 0
            max_void_percentage = 0
        
        # Ajouter une nouvelle ligne au DataFrame
        new_row = pd.DataFrame([{
            "Image ID": image_id,
            "Component": component_count,
            "Area": total_area,
            "Void %": void_percentage,
            "Max Void %": max_void_percentage
        }])
        
        predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
        
        # Sauvegarder les prédictions en CSV
        csv_path = os.path.join(app.config['RESULT_FOLDER'], 'predictions.csv')
        predictions_df.to_csv(csv_path, index=False)
        
        # Rendre la page des résultats
        return render_template('results.html',
                               original_image=url_for('static', filename=f'uploads/{original_filename}'),
                               annotated_image=url_for('static', filename=f'results/saves/{original_filename}'),
                               table_data=predictions_df.to_dict(orient="records"))

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return "No file part"
        files = request.files.getlist('files[]')
        image_paths = []
        for file in files:
            if file and file.filename != '':
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_paths.append(url_for('static', filename=f"uploads/{filename}"))
        return render_template('annotate.html', image_paths=image_paths)
    else:
        return render_template('annotate.html')

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    image_url = data['image_url']
    points = data['points']
    labels = data['labels']
    # Supprimer le '/' initial si présent
    if image_url.startswith('/'):
        image_url = image_url[1:]
    image_path = os.path.join(app.root_path, image_url)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    input_points = np.array(points)
    input_labels = np.array(labels)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    mask = masks[0]
    mask_image = (mask * 255).astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_image)
    masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    result_filename = f"annotated_{uuid.uuid4().hex}.png"
    result_path = os.path.join(app.config['ANNOTATION_FOLDER'], result_filename)
    cv2.imwrite(result_path, masked_image_bgr)
    
    result_url = url_for('static', filename=f"annotations/{result_filename}")
    return jsonify({'result_url': result_url})

@app.route('/finetune', methods=['POST'])
def finetune():
    data = request.get_json()
    image_url = data['image_url']
    # Supprimer le '/' initial si présent
    if image_url.startswith('/'):
        image_url = image_url[1:]
    image_path = os.path.join(app.root_path, image_url)
    # Ici, implémentez la logique de fine-tuning
    try:
        # Code de fine-tuning à implémenter
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(app.config['RESULT_FOLDER'], 'predictions.csv')
    return send_file(csv_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port = 5400, host = "0.0.0.0")