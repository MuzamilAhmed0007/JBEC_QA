This repository contains the complete implementation of **JBEC-QA**

## 📦 Installation

### ⚙️ 1. Clone the Repository
git clone https://github.com/your-username/jbec-qa.git
cd jbec-qa 

### 2. Create Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # for Linux/macOS
venv\\Scripts\\activate   # for Windows
### 3. Install Requirements

pip install -r requirements.txt
### Ensure you also download required NLTK models:

import nltk
\n nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
