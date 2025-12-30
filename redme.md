### training

mtcf-net  --->mtcf-net.py

mtf-net    --->mtf-net.py



### package

&#x23; --- Core Scientific Stack ---&#x20;

numpy==1.26.4&#x20;

pandas==1.2.4&#x20;

scipy==1.13.1&#x20;

h5py==3.13.0&#x20;

scikit-learn==1.6.1&#x20;

matplotlib==3.9.4&#x20;

seaborn==0.13.2&#x20;

tqdm==4.67.1&#x20;

&#x23; --- Deep Learning (PyTorch) ---&#x20;

&#x23; Note: The specific versions below are nightly builds with CUDA 12.8 support.&#x20;

&#x23; For public usage, you might recommend stable versions, e.g., torch>=2.0.0&#x20;

torch==2.7.0.dev20250304+cu128&#x20;

torchvision==0.22.0.dev20250305+cu128&#x20;

&#x23; --- Utilities & Data Processing ---&#x20;

imbalanced-learn==0.12.4&#x20;

joblib==1.4.2&#x20;

openpyxl==3.1.5&#x20;

pillow==11.1.0&#x20;

pywavelets==1.5.0



### Installation Install dependencies:&#x20;

&#x60;``bash pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

