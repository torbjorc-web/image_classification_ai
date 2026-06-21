AI Image Classification Dashboard
A Streamlit app that lets users upload images and get instant AI-powered classification results using Hugging Face Transformers.
The dashboard includes:
Single image classification.
Classification history.
Analytics and visualizations.
CSV export of all results.
Features
Upload JPG, JPEG, or PNG images.
Classify images with state-of-the-art Hugging Face models.
Compare predictions from different model architectures.
View classification history in collapsible sections.
Track top predictions and confidence scores.
Explore analytics with charts and summary metrics.
Download all classification results as CSV.
Clear stored results and start fresh.
Supported Models
Google ViT Base Patch16 224: google/vit-base-patch16-224
Microsoft ResNet 50: microsoft/resnet-50
These models are trained for image classification and return top-k predictions with confidence scores.
Project Structure
.
├── image_app.py
├── image_app_v2.py
├── solution.py
└── README.md
Requirements
Install the required Python packages:
pip install streamlit pandas numpy matplotlib seaborn pillow torch transformers
If you are running on a machine with GPU support, make sure your PyTorch installation matches your CUDA setup.
Running the App
Start the Streamlit app with:
streamlit run image_app.py
Or run the improved version:
streamlit run image_app_v2.py
How to Use
Open the app in your browser.
Choose a model from the sidebar.
Click Load AI Model.
Upload an image in the first tab.
Click Classify Image.
Review predictions, history, and analytics.
Download results as CSV if needed.
Tabs
Single Image Classification
Upload an image and view the model’s top predictions with confidence scores.
Classification History
Browse previously classified images in expandable sections.
Analytics
View summary metrics, class frequency charts, confidence distribution, and a detailed results table.
Notes
Images are converted to RGB before classification to ensure compatibility with the model.
Session state is used to preserve results during the current browser session.
Model loading uses caching so the model is not reloaded unnecessarily.
Troubleshooting
Model load fails
Check your internet connection.
Make sure transformers and torch are installed correctly.
Try the fallback model if the primary model is unavailable.
Image does not classify
Use a supported image format: JPG, JPEG, or PNG.
Try a different image with clearer content.
Make sure the model has been loaded first.
Learning Goals
This project demonstrates:
Streamlit app structure.
Session state management.
Hugging Face image classification pipelines.
Data visualization with Matplotlib and Seaborn.
Exporting results for downstream analysis.
License
This project is intended for educational use.# image_classification_ai