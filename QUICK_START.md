# 🚀 Quick Start: Deploy to Hugging Face Spaces

## TL;DR - Fast Deployment

```bash
# 1. Install Git LFS (if not already installed)
sudo apt-get install git-lfs  # Linux
# brew install git-lfs        # Mac
git lfs install

# 2. Navigate to project
cd /home/calviem/CSSE416/Homework/Project

# 3. Initialize git (if not done)
git init

# 4. Add Hugging Face Space as remote (create Space first on huggingface.co)
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# 5. Add, commit, and push
git add .
git commit -m "Deploy ECG Classification App"
git push space main
```

## 📝 Before You Start

1. **Create a Hugging Face account:** [huggingface.co/join](https://huggingface.co/join)

2. **Create a new Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose a name (e.g., `ecg-heartbeat-classification`)
   - Select **Docker** as SDK
   - Make it Public (free tier)
   - Click "Create Space"

3. **Install Git LFS** (for large model files)

## ✅ Files Ready for Deployment

Your project now includes:
- ✅ `Dockerfile` - Container configuration
- ✅ `requirements.txt` - Python dependencies (with gunicorn)
- ✅ `README.md` - With Hugging Face metadata
- ✅ `.gitattributes` - Git LFS configuration for models
- ✅ `.dockerignore` - Optimized Docker builds
- ✅ `app.py` - Updated for port 7860 (HF Spaces default)

## 🌐 After Deployment

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Direct app URL:
```
https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space
```

## 🐛 Common Issues

**"Git LFS not installed"**
```bash
sudo apt-get install git-lfs
git lfs install
```

**"Authentication failed"**
- Use a Hugging Face token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Configure git credentials:
```bash
git config --global credential.helper store
# Then push again and enter your username and token
```

**"Build failed"**
- Check logs in your Space's "Logs" tab
- Most common: missing dependencies or wrong port

## 📖 Full Guide

For detailed instructions, troubleshooting, and customization options, see [DEPLOYMENT.md](DEPLOYMENT.md)

## 💡 Quick Test Locally

Test with Docker before deploying:

```bash
# Build Docker image
docker build -t ecg-app .

# Run container
docker run -p 7860:7860 ecg-app

# Visit: http://localhost:7860
```

## 🔄 Update Your Deployment

Made changes? Just push again:

```bash
git add .
git commit -m "Update: description of changes"
git push space main
```

The Space will automatically rebuild! 🎉

---

**Need help?** See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

