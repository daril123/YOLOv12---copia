# Standard library imports
import os
import sys
import json
import base64
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import ollama

class LabelExtractor:
    """
    Processor for extracting and analyzing etiquetas (labels) using OCR
    """
    
    def __init__(self):
        """
        Initialize the label extractor with OCR capabilities
        """
        self.name = "etiqueta"
        # Initialize Ollama model
        self.model_name = "granite3.2-vision"
        print(f"Label extractor initialized with OCR model: {self.model_name}")
        
    def extract_json(self, texto):
        """
        Extract the first valid JSON block from 'texto' and parse it.
        Returns a dict with the JSON content.
        Raises ValueError if no valid JSON is found.
        """
        # Find position of first and last braces
        inicio = texto.find('{')
        fin = texto.rfind('}')
        if inicio == -1 or fin == -1 or inicio > fin:
            raise ValueError("No valid JSON block found")
        # Extract the JSON substring
        json_str = texto[inicio:fin + 1]
        # Parse with json.loads()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON: {e}")

    def extract_label_content(self, img_etiqueta):
        """
        Extract content from a label image using OCR
        
        Args:
            img_etiqueta: Label image
            
        Returns:
            Dictionary with extracted fields
        """
        prompt = """
        Act as an OCR specialist with expertise in reading industrial labels. 
        Analyze the provided image and:
        1. Recognize all text in the image with high precision.
        2. IMPORTANT: Characters that appear spread apart but are on the same line should be treated as a SINGLE WORD or CODE, not as separate words.
        3. Extract exactly three pieces of information:
        a. "code": The ENTIRE first line of text (treat all characters as part of one code even if spaces appear between them)
        b. "quality": The ENTIRE second line (again, treat all characters as one value)
        c. "line": The ENTIRE third/last line (all characters as one value)
        FORMAT THE OUTPUT AS JSON:
        {
        "code": "entire first line code",
        "quality": "entire second line value",
        "line": "entire third line value"
        }
        Return ONLY valid JSON, nothing else. Do not split codes that appear on the same line even if characters are spaced apart.
        """
        
        try:
            # Check if input is valid
            if img_etiqueta is None or img_etiqueta.size == 0 or img_etiqueta.shape[0] < 10 or img_etiqueta.shape[1] < 10:
                print("Error: Imagen de etiqueta inválida o demasiado pequeña para OCR")
                return {
                    'code': 'INVALID_IMAGE',
                    'quality': 'INVALID_IMAGE',
                    'line': 'INVALID_IMAGE',
                    'error': 'Invalid or too small image'
                }
                
            # Convert to grayscale if not already
            if len(img_etiqueta.shape) > 2:
                img_etiqueta = cv2.cvtColor(img_etiqueta, cv2.COLOR_BGR2GRAY)
                
            # Resize if too small for good OCR
            if img_etiqueta.shape[0] < 50 or img_etiqueta.shape[1] < 50:
                factor = max(50 / img_etiqueta.shape[0], 50 / img_etiqueta.shape[1])
                new_size = (int(img_etiqueta.shape[1] * factor), int(img_etiqueta.shape[0] * factor))
                img_etiqueta = cv2.resize(img_etiqueta, new_size, interpolation=cv2.INTER_CUBIC)
                
            # Enhance contrast for better OCR
            img_etiqueta = cv2.equalizeHist(img_etiqueta)
            
            # Encode image for API
            _, img_encoded = cv2.imencode('.jpg', img_etiqueta)
            encoded = base64.b64encode(img_encoded).decode("utf-8")
            
            # Call Ollama API
            res = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [encoded],
                        'options': {
                            'temperature': 0,   # Keep consistency
                        }
                    }
                ]
            )
            
            # Extract JSON result
            output = self.extract_json(res['message']['content'])
            
            # Remove spaces from values
            output = {k: v.replace(" ", "") for k, v in output.items()}
            
            return output
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            # Return default values if failed
            return {
                'code': 'OCR_ERROR',
                'quality': 'OCR_ERROR',
                'line': 'OCR_ERROR',
                'error': str(e)
            }
    
    def generate_report(self, image_name, labels_data, output_dir):
        """
        Generate a report of the OCR extracted label data
        
        Args:
            image_name: Name of the original image
            labels_data: List of dictionaries with the data of the labels
            output_dir: Directory where to save the report
        
        Returns:
            report_paths: Paths to the generated report files
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame with the data
        df = pd.DataFrame(labels_data)
        
        # Report format
        report_path = os.path.join(output_dir, f"{image_name}_etiqueta_report.csv")
        
        # Save as CSV
        df.to_csv(report_path, index=False)
        
        # Also generate a text version for easy visualization
        text_report_path = os.path.join(output_dir, f"{image_name}_etiqueta_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(f"ETIQUETA OCR REPORT - {image_name}\n")
            f.write("="*50 + "\n\n")
            
            for i, label in enumerate(labels_data):
                f.write(f"LABEL #{i+1}\n")
                f.write(f"  Code: {label.get('code', 'UNKNOWN')}\n")
                f.write(f"  Quality: {label.get('quality', 'UNKNOWN')}\n")
                f.write(f"  Line: {label.get('line', 'UNKNOWN')}\n")
                f.write(f"  Confidence: {label.get('conf', 0):.2f}\n\n")
        
        print(f"Report generated at: {report_path}")
        print(f"Text report generated at: {text_report_path}")
        
        return report_path, text_report_path
    
    def process(self, etiqueta_detections, image, corners=None, zone_masks=None, image_name=None, output_dir=None):
        """
        Process all detected labels
        
        Args:
            etiqueta_detections: List of label detections from vertex detector
            image: Original image
            corners: Corners of the palanquilla (optional)
            zone_masks: Zone masks (optional)
            image_name: Image name (without extension)
            output_dir: Output directory for saving reports
            
        Returns:
            processed_data: Dictionary with processing results
        """
        results = []
        visualizations = {}
        
        # Process each detected label
        for i, detection in enumerate(etiqueta_detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection.get('conf', 0)
            
            # Extract the label region
            label_image = image[y1:y2, x1:x2].copy()
            
            # Process the label with OCR
            ocr_data = self.extract_label_content(label_image)
            
            # Combine data
            label_data = {
                'id': i+1,
                'code': ocr_data.get('code', 'UNKNOWN'),
                'quality': ocr_data.get('quality', 'UNKNOWN'),
                'line': ocr_data.get('line', 'UNKNOWN'),
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            }
            
            # Create visualization
            viz_img = label_image.copy()
            # Add text overlay on the visualization
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(viz_img, f"Code: {ocr_data.get('code', 'UNKNOWN')}", (10, 20), font, 0.5, (0, 0, 255), 2)
            cv2.putText(viz_img, f"Quality: {ocr_data.get('quality', 'UNKNOWN')}", (10, 40), font, 0.5, (0, 0, 255), 2)
            cv2.putText(viz_img, f"Line: {ocr_data.get('line', 'UNKNOWN')}", (10, 60), font, 0.5, (0, 0, 255), 2)
            
            # Save visualization
            visualization_key = f"etiqueta_{i+1}"
            visualizations[visualization_key] = viz_img
            
            # Also save the original label image for reference
            visualization_key_orig = f"etiqueta_orig_{i+1}"
            visualizations[visualization_key_orig] = label_image
            
            results.append(label_data)
        
        # If there are results and we have image name and output directory, generate a report
        report_paths = None
        if results and image_name and output_dir:
            # Create directory for this type - ensure it's named "etiqueta"
            etiqueta_dir = os.path.join(output_dir, image_name, "etiqueta")
            os.makedirs(etiqueta_dir, exist_ok=True)
            
            # Generate the report in the specific folder
            report_paths = self.generate_report(image_name, results, etiqueta_dir)
        
        return {
            'processed_data': results,
            'visualizations': visualizations,
            'report_paths': report_paths
        }