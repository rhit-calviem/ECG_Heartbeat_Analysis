# 🚀 Deployment Guide: Hugging Face Spaces

This guide will walk you through deploying your ECG Heartbeat Classification app to Hugging Face Spaces for free!

## 📋 Prerequisites

Before you begin, make sure you have:
- ✅ A Hugging Face account (create one at [huggingface.co](https://huggingface.co/join))
- ✅ Git installed on your machine
- ✅ Git LFS (Large File Storage) installed

### Installing Git LFS

If you don't have Git LFS installed:

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install git-lfs
git lfs install
```

**Mac:**
```bash
brew install git-lfs
git lfs install
```

**Windows:**
Download from [git-lfs.github.com](https://git-lfs.github.com/)

## 🎯 Step-by-Step Deployment

### Step 1: Create a New Space on Hugging Face

1. Go to [huggingface.co](https://huggingface.co) and log in
2. Click on your profile picture → **"New Space"**
3. Fill in the details:
   - **Name:** `ecg-heartbeat-classification` (or your preferred name)
   - **License:** Choose appropriate license (e.g., MIT)
   - **Select SDK:** Choose **Docker**
   - **Visibility:** Public (for free tier)
4. Click **"Create Space"**

### Step 2: Set Up Git Repository

After creating your Space, you'll see instructions to push your code. Open a terminal in your project directory:

```bash
# Navigate to your project directory
cd /home/calviem/CSSE416/Homework/Project

# Initialize git if not already done
git init

# Add the Hugging Face Space as remote
# Replace YOUR_USERNAME and YOUR_SPACE_NAME with your actual values
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Or if you prefer SSH:
# git remote add space git@hf.co:spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

### Step 3: Configure Git LFS for Model Files

Your model files are large (560MB total), so we'll use Git LFS:

```bash
# Initialize Git LFS (should already be done via .gitattributes)
git lfs install

# Track your model files (already configured in .gitattributes)
# This tells Git LFS to handle .keras and .pth files
git lfs track "*.keras"
git lfs track "*.pth"
```

### Step 4: Add and Commit Files

```bash
# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: ECG Heartbeat Classification App"
```

### Step 5: Push to Hugging Face Spaces

```bash
# Push to main branch
git push space main

# If you're on a different branch (e.g., master), use:
# git push space master:main
```

**Note:** The first push will take some time because it's uploading ~560MB of model files via Git LFS.

### Step 6: Wait for Build

1. Go to your Space page on Hugging Face: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. You'll see a build log showing Docker building your container
3. This usually takes 5-10 minutes for the first build
4. Once complete, your app will be live! 🎉

## 🌐 Accessing Your Deployed App

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Or the direct app URL:
```
https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space
```

## 🔧 Configuration Files

Your deployment uses these files:

### 1. **README.md**
Contains metadata in YAML frontmatter:
```yaml
---
title: ECG Heartbeat Classification
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---
```

### 2. **Dockerfile**
Defines how to build and run your app in a container.

### 3. **.gitattributes**
Tells Git LFS which files to track (model files).

### 4. **requirements.txt**
Lists all Python dependencies.

## 🎨 Customizing Your Space

### Change the Title or Emoji

Edit the YAML frontmatter in `README.md`:
```yaml
---
title: Your Custom Title
emoji: 🫀
colorFrom: blue
colorTo: green
---
```

### Change the Port

If needed, edit both:
1. `README.md` (line with `app_port: 7860`)
2. `Dockerfile` (EXPOSE line and CMD line)

## 🐛 Troubleshooting

### Build Failed

1. Check the build logs in your Space
2. Common issues:
   - Missing dependencies in `requirements.txt`
   - Incorrect Dockerfile syntax
   - Port mismatch between Dockerfile and README

### Models Not Loading

1. Ensure Git LFS is properly configured
2. Check that model files were uploaded (they should show file size on HF)
3. Look for "Stored with Git LFS" badge on model files

### App Timeout

If your app times out:
1. The free tier has 16GB RAM and 2 vCPU
2. Consider reducing the number of models loaded at once
3. Or upgrade to a paid tier for more resources

### Git LFS Quota Exceeded

Hugging Face offers generous LFS storage, but if you hit limits:
1. You can request a quota increase
2. Or optimize your model files (pruning, quantization)

## 🔄 Updating Your Deployment

To update your deployed app:

```bash
# Make your changes to the code
# Then commit and push

git add .
git commit -m "Description of your changes"
git push space main
```

The Space will automatically rebuild and redeploy!

## 📊 Monitoring

- **Build logs:** Check the "Logs" tab in your Space
- **Runtime logs:** View them in the "Logs" section while app is running
- **Analytics:** Hugging Face provides basic usage analytics

## 💡 Pro Tips

1. **Use a .dockerignore file** to exclude unnecessary files from the Docker build:
   ```
   .git
   __pycache__
   *.pyc
   .env
   Notebooks/
   ```

2. **Optimize Docker layer caching** by copying `requirements.txt` before other files

3. **Pin your dependencies** in requirements.txt for reproducible builds

4. **Test locally with Docker** before pushing:
   ```bash
   docker build -t ecg-app .
   docker run -p 7860:7860 ecg-app
   ```

5. **Enable persistent storage** if you need to save user uploads or logs (configure in Space settings)

## 🆘 Getting Help

- **Hugging Face Forums:** [discuss.huggingface.co](https://discuss.huggingface.co)
- **Discord:** [hf.co/join/discord](https://hf.co/join/discord)
- **Documentation:** [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)

## 📝 Alternative: Deploy Without Docker

If you prefer not to use Docker, you can use the **Gradio SDK** instead:

1. Change `sdk: docker` to `sdk: gradio` in README.md
2. Remove the Dockerfile
3. Create a Gradio interface wrapper for your Flask app
4. Rename your main file to `app.py`

However, Docker gives you more control and is recommended for this Flask application.

## ✅ Checklist

Before deploying, make sure:
- [ ] Git LFS is installed and initialized
- [ ] .gitattributes is configured for model files
- [ ] README.md has correct YAML frontmatter
- [ ] Dockerfile is present
- [ ] requirements.txt is complete
- [ ] All model files are in the `models/` directory
- [ ] Data directory structure is correct
- [ ] You've updated the README with your actual Space URL

---

🎉 **Congratulations!** Your ECG Heartbeat Classification app is now deployed and accessible from anywhere in the world!

