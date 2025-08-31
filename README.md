# Visual Product Matcher

A web app for finding visually similar products based on uploaded images.

## Features
- Image upload via file or URL
- View uploaded image
- Search and display similar products with metadata (name, category)
- Filter results by similarity score
- Product database: 100 items from DummyJSON API
- Loading states and error handling
- Mobile-responsive

## Setup and Run Locally
1. Clone the repo: `git clone https://github.com/yourusername/visual-product-matcher.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Deployment (e.g., on Render.com - Free Tier)
1. Create a free account at render.com.
2. New > Web Service > Build from GitHub repo.
3. Runtime: Python
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port $PORT --server.enableCORS false`
6. Deploy – gets a free URL like https://your-app.onrender.com

## Approach
See the brief write-up below.