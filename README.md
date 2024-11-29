# Badge Scanner - Conference Badge Information Extractor

A Streamlit application that uses GPT-4o-mini to extract information from conference badge photos.

## Features
- Upload multiple badge photos at once
- Configurable fields for extraction (name, company, etc.)
- Automatic image optimization before processing
- Export results to CSV
- Batch processing with progress tracking

## Demo
Try the live application at: [https://badge-scanner.streamlit.app](https://badge-scanner.streamlit.app)

## Requirements
- OpenAI API key with GPT-4o-mini access
- Python 3.9 or later

## Local Development Setup
1. Clone the repository
```bash
git clone [your-repo-url]
cd badge-scanner
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run badge_extractor.py
```

## Usage
- Get your OpenAI API key from https://platform.openai.com/api-keys
- Enter your API key in the sidebar
- Configure extraction fields (default: first name, last name, company)
- Upload badge photos (supported formats: JPG, JPEG, PNG)
- Click "Extract Information"
- Download results as CSV

## Cost Estimation
GPT-4o-mini API costs approximately $0.003825 per image (as of November 2024)

## Limitations
- Maximum batch size: Unlimited (but processed sequentially)
- Maximum image size: Automatically optimized to 800x800px
- Supported file types: JPG, JPEG, PNG

Built with Streamlit
Powered by OpenAI's GPT-4o-mini 
