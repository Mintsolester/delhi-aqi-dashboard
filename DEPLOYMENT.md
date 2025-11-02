# 🚀 Deployment Guide for Delhi AQI Dashboard

## Step-by-Step Deployment to Streamlit Cloud

### 1. Prepare for Git

```bash
cd /home/aryan/pollution_project

# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Delhi Environmental Health Dashboard"
```

### 2. Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `delhi-aqi-dashboard`
3. Description: "AI-Powered LSTM Forecasting & Air Quality Monitoring"
4. Keep it PUBLIC (required for free Streamlit hosting)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### 3. Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/delhi-aqi-dashboard.git

# Push code
git branch -M main
git push -u origin main
```

### 4. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Click "New app"

2. **Connect Repository:**
   - Authorize GitHub access
   - Select repository: `YOUR_USERNAME/delhi-aqi-dashboard`
   - Branch: `main`
   - Main file path: `app.py`

3. **Add Secrets:**
   Click "Advanced settings" → "Secrets"
   
   Paste this (get your own API keys first):
   ```toml
   [api_keys]
   openaq_key = ""
   openweathermap_key = "YOUR_OWM_KEY_HERE"
   ```

4. **Deploy:**
   - Click "Deploy!"
   - Wait 5-10 minutes for first deployment
   - Your app will be live at: `https://YOUR_USERNAME-delhi-aqi-dashboard-app-xxxxx.streamlit.app/`

### 5. Get Free API Keys

#### OpenWeatherMap (Required - FREE)
1. Go to: https://openweathermap.org/api
2. Sign up for free account
3. Navigate to: API Keys
4. Copy your API key
5. Add to Streamlit secrets: `openweathermap_key = "YOUR_KEY"`

#### OpenAQ (Optional - FREE)
1. Go to: https://docs.openaq.org/
2. API v3 requires authentication
3. For now, app works with synthetic data

### 6. Model Files Issue

**Problem:** Model files (`.keras`, `.pkl`) are too large for Git (>100MB limit)

**Solutions:**

**Option A: Use Git LFS (Git Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.keras"
git lfs track "*.pkl"
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add models/
git commit -m "Add model files with LFS"
git push
```

**Option B: Host Models Separately (Recommended for production)**
```bash
# Upload models to cloud storage
# - Google Cloud Storage
# - AWS S3
# - Hugging Face Hub
# - GitHub Releases

# Modify app.py to download models on startup
```

**Option C: Retrain on Deployment (Not recommended - slow)**
- Remove models from git
- Let Streamlit rebuild models on first run
- Will take 10-15 minutes on deployment

### 7. Test Your Deployed App

Visit your app URL and test:
- [ ] Live Air Quality Map loads
- [ ] LSTM Forecast tab works (if models deployed)
- [ ] Location Search functions
- [ ] Health Advisory generates
- [ ] Alert subscriptions save data
- [ ] Data export downloads CSV

### 8. Custom Domain (Optional)

1. In Streamlit Cloud dashboard:
   - Settings → General → Custom domain
   - Add: `delhi-aqi.yourdomain.com`

2. In your domain registrar:
   - Add CNAME record pointing to Streamlit domain

### 9. Monitor & Maintain

**View Logs:**
- Streamlit dashboard → Manage app → Logs

**Update App:**
```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push

# Streamlit auto-redeploys on push
```

**Manage Secrets:**
- Dashboard → Settings → Secrets
- Update API keys without redeploying

### 10. Troubleshooting

**Error: "ModuleNotFoundError"**
- Check `requirements.txt` has all dependencies
- Ensure package names are correct

**Error: "Out of memory"**
- Streamlit Cloud has 1GB RAM limit
- Optimize data loading (already done with lazy loading)
- Remove heavy model files if needed

**Error: "Model not found"**
- Deploy models using Git LFS
- Or remove LSTM forecast feature temporarily

**App is slow:**
- Enable caching (already done with `@st.cache_data`)
- Reduce data size
- Optimize geospatial data

### 11. Free Tier Limits

**Streamlit Community Cloud:**
- ✅ Unlimited public apps
- ✅ 1GB RAM per app
- ✅ Auto-sleep after inactivity
- ✅ Auto-redeploy on git push
- ❌ No GPU support (LSTM training requires local setup)

**Alternative: Deploy with GPU**
- Use AWS EC2 with GPU
- Use Google Cloud Run
- Use Hugging Face Spaces (free GPU)

---

## Quick Deploy Commands

```bash
# One-time setup
cd /home/aryan/pollution_project
git init
git add .
git commit -m "Initial commit"

# Push to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/delhi-aqi-dashboard.git
git push -u origin main

# Then deploy on: https://share.streamlit.io/
```

---

## Post-Deployment Checklist

- [ ] App loads successfully
- [ ] All tabs are functional
- [ ] API keys are configured in secrets
- [ ] Data subscriptions work
- [ ] Share URL with users
- [ ] Monitor logs for errors
- [ ] Set up analytics (optional)

---

**🎉 Your app is now live! Share it with the world!**

App URL: `https://YOUR_APP_NAME.streamlit.app/`

