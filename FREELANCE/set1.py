    # 🔥 Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

# 🧠 Step 1: Create Fake Job Data
data = {
    'title': [
        'Build a website', 
        'Design a logo', 
        'React app development', 
        'SEO optimization', 
        'Backend API setup'
    ],
    'skills': [
        'HTML CSS JS', 
        'Photoshop Illustrator', 
        'React Node', 
        'SEO Content', 
        'Node Express MongoDB'
    ],
    'timeline_days': [5, 2, 7, 4, 6],
    'price': [3000, 1500, 6000, 2500, 5000]
}

df = pd.DataFrame(data)

# 👀 Show the data
df

# 🧪 Step 2: Convert skills (text) to numbers
vectorizer = CountVectorizer()
skill_features = vectorizer.fit_transform(df['skills'])

# Combine skills and timeline into one feature set
from scipy.sparse import hstack
X = hstack([skill_features, df[['timeline_days']]])

# 🎯 Step 3: Train the model
model = LinearRegression()
model.fit(X, df['price'])

# 🔮 Step 4: Predict Price for a New Job
new_skills = ['React MongoDB']  # change this for testing
new_timeline = [5]

new_skill_features = vectorizer.transform(new_skills)
new_X = hstack([new_skill_features, [[new_timeline[0]]]])

predicted_price = model.predict(new_X)
print(f"\n💸 Predicted Price: ₹{round(predicted_price[0])}")
