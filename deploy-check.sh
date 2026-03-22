#!/bin/bash
# 🚀 Quick Deploy Script for Railway

echo "🔍 Checking deployment readiness..."

# Check if required files exist
if [ -f "main.py" ]; then
    echo "✅ main.py found"
else
    echo "❌ main.py missing"
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
else
    echo "❌ requirements.txt missing"
    exit 1
fi

if [ -f "concrete_models.pkl" ]; then
    echo "✅ concrete_models.pkl found"
else
    echo "❌ concrete_models.pkl missing"
    exit 1
fi

if [ -f "imgone.png" ] && [ -f "imgtwo.png" ]; then
    echo "✅ Images found"
else
    echo "❌ Images missing"
    exit 1
fi

echo ""
echo "🎉 All files ready for deployment!"
echo ""
echo "📋 Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Ready for deployment'"
echo "3. git push origin main"
echo "4. Go to railway.app and deploy from GitHub"
echo ""
echo "🌐 Your app will be live in ~2 minutes!"
