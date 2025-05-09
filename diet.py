import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Step 1: Simulate dummy training data
def generate_dummy_data(n=500):
    np.random.seed(42)
    data = {
        "age": np.random.randint(18, 60, n),
        "height": np.random.randint(150, 200, n),
        "weight": np.random.randint(50, 120, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "diet_type": np.random.choice(["Vegan", "Carnivorous"], n),
    }

    df = pd.DataFrame(data)
    
    # Simplified formula to simulate macros
    df["calories"] = df["weight"] * 25 + np.random.randint(-200, 200, n)
    df["protein"] = df["weight"] * 1.8
    df["carbs"] = df["weight"] * 2.5
    df["fats"] = df["weight"] * 0.8

    return df

# Step 2: Preprocessing and model training
def train_model(df):
    X = df[["age", "height", "weight", "gender", "diet_type"]].copy()
    y = df[["calories", "protein", "carbs", "fats"]]

    # Encode categorical features
    le_gender = LabelEncoder()
    le_diet = LabelEncoder()
    X["gender"] = le_gender.fit_transform(X["gender"])
    X["diet_type"] = le_diet.fit_transform(X["diet_type"])

    model = RandomForestRegressor()
    model.fit(X, y)

    return model, le_gender, le_diet

# Step 3: Get user input
def get_user_input():
    age = int(input("Age: "))
    height = int(input("Height (cm): "))
    weight = int(input("Weight (kg): "))
    gender = input("Gender (Male/Female): ")
    diet_type = input("Diet Type (Vegan/Carnivorous): ")
    return age, height, weight, gender, diet_type

# Step 4: Predict macros
def predict_macros(model, le_gender, le_diet, age, height, weight, gender, diet_type):
    gender_enc = le_gender.transform([gender])[0]
    diet_enc = le_diet.transform([diet_type])[0]

    input_data = np.array([[age, height, weight, gender_enc, diet_enc]])
    prediction = model.predict(input_data)[0]
    return {
        "Calories": round(prediction[0], 2),
        "Protein (g)": round(prediction[1], 2),
        "Carbs (g)": round(prediction[2], 2),
        "Fats (g)": round(prediction[3], 2)
    }

# Step 5: Sample diet chart
def generate_diet_chart(macros, diet_type):
    print("\n--- Diet Chart ---")
    print(f"Calories: {macros['Calories']} kcal")
    print(f"Protein: {macros['Protein (g)']} g")
    print(f"Carbs: {macros['Carbs (g)']} g")
    print(f"Fats: {macros['Fats (g)']} g")

    print("\nSample Meal Plan:")
    if diet_type.lower() == "vegan":
        print("- Breakfast: Oats with soy milk and banana")
        print("- Lunch: Lentil curry with brown rice")
        print("- Dinner: Tofu stir-fry with quinoa")
        print("- Snacks: Nuts, fruits, protein shake (vegan)")
    else:
        print("- Breakfast: Eggs, toast, peanut butter")
        print("- Lunch: Grilled chicken with rice and vegetables")
        print("- Dinner: Fish or meat with sweet potatoes")
        print("- Snacks: Greek yogurt, boiled eggs, protein shake")

# Main function
def main():
    df = generate_dummy_data()
    model, le_gender, le_diet = train_model(df)

    age, height, weight, gender, diet_type = get_user_input()
    macros = predict_macros(model, le_gender, le_diet, age, height, weight, gender, diet_type)
    generate_diet_chart(macros, diet_type)

if __name__ == "__main__":
    main()
