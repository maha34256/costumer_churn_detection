import pickle

# Load trained model
loaded_model = pickle.load(open("churn_model.pkl", "rb"))

print("===== CUSTOMER CHURN PREDICTION =====")

credit_score = int(input("Enter Credit Score: "))

print("\nGeography:")
print("0 = France")
print("1 = Germany")
print("2 = Spain")
geography = int(input("Enter Geography: "))

print("\nGender:")
print("0 = Female")
print("1 = Male")
gender = int(input("Enter Gender: "))

age = int(input("Enter Age: "))
tenure = int(input("Enter Tenure: "))
balance = float(input("Enter Balance: "))
num_products = int(input("Enter Number of Products: "))
has_card = int(input("Has Credit Card? (1=Yes, 0=No): "))
active_member = int(input("Is Active Member? (1=Yes, 0=No): "))
salary = float(input("Enter Estimated Salary: "))

customer_data = [[
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    active_member,
    salary
]]

prediction = loaded_model.predict(customer_data)

print("\n===== PREDICTION RESULT =====")

if prediction[0] == 1:
    print("⚠ Customer is likely to CHURN")
else:
    print("✅ Customer is likely to STAY")