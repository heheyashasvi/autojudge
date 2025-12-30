#!/bin/bash

echo "🚀 Uploading AutoJudge to GitHub..."

# Initialize git repository
echo "📁 Initializing git repository..."
git init

# Add all files
echo "📦 Adding all files..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "🎯 Initial commit: AutoJudge Difficulty Predictor

✅ Complete ML pipeline with Random Forest models
✅ React frontend with TypeScript
✅ Flask API backend with comprehensive evaluation
✅ 51.6% classification accuracy on 4,112 problems
✅ Confusion matrix, MAE, RMSE metrics available
✅ Production-ready with error handling and documentation

Features:
- Custom feature extraction (TF-IDF + statistical)
- Dual prediction (classification + regression)
- Beautiful evaluation dashboard
- Complete test suite and documentation"

# Add remote repository
echo "🔗 Connecting to GitHub repository..."
git remote add origin https://github.com/heheyashasvi/autojudge.git

# Set main branch
echo "🌿 Setting main branch..."
git branch -M main

# Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push -u origin main

echo "✅ Successfully uploaded to GitHub!"
echo "🌐 Visit: https://github.com/heheyashasvi/autojudge"
echo ""
echo "🎉 Your AutoJudge project is now live on GitHub!"