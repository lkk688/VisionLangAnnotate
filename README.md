# VisionLangAnnotate
vision and language model, open vocabulary output, deep learning inference, annotation, python backend, fine-tuning and continuous improvement

## Repository Setup
```bash
git clone https://github.com/lkk688/VisionLangAnnotate.git
cd VisionLangAnnotate
% conda env list
% conda activate mypy311
pip freeze > requirements.txt
pip install -r requirements.txt
#Install in Development Mode
#pip install -e .
pip install flit
flit install --symlink
#test import models: >>> import VisionLangAnnotateModels
```

Create Conda virtual environment and install cuda
```bash
conda create --name py312 python=3.12
conda activate py312
conda info --envs #check existing conda environment
% conda env list
$ conda install cuda -c nvidia/label/cuda-12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install mkdocs mkdocs-material
```

```bash
#test backend
uvicorn src.main:app --reload
#Verify at: http://localhost:8000
```

```bash
#Set Up React Frontend
cd ../frontend
brew install node
npm create vite@latest . --template react 
#choose react, JavaScript+SWC(Speedy Web Compiler) a Rust-based alternative to Babel.
npm install
#run the frontend
npm run dev
```

```bash
npm install @vitejs/plugin-react --save-dev
npm install react-router-dom
```

Documents
```bash
pip install mkdocs mkdocs-material
docs % mkdocs new .
docs % ls
docs                    getting-started.md      mkdocs.yml
#Run locally:
mkdocs serve --dev-addr localhost:8001  #Docs will be at: http://localhost:8001, default port is 8000
#find the process using port 8000
lsof -i :8000
#kill -9 <PID>
```

Git setup
```bash
git add .
git commit -m "Initial setup: FastAPI + React + Docs"
git push origin main
```

Dockerize: backend/Dockerfile
Backend: Deploy FastAPI (Render, Railway).
Frontend: Deploy React (Vercel, Netlify).

## File Structure

annotation-tool/
├── backend/
│   ├── src/
│   │   ├── main.py         # FastAPI app
│   │   └── api/            # API routes
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # React app
│   │   └── components/
│   ├── package.json
│   └── vite.config.js
├── docs/
│   ├── getting-started.md
│   └── api-reference.md
├── pyproject.toml          # Python packaging
├── README.md
├── .gitignore
└── LICENSE