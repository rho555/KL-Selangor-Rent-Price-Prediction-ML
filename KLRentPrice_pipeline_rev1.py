# Required imports
import pandas as pd
import re
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Load the dataset
data = pd.read_csv(r"C:\Users\...\mudah-apartment-kl-selangor.csv")

# Function to clean numeric columns (removes non-numerical characters and whitespace)
def clean_numeric_column(value_str):
    """
    Cleans a string by extracting only the numerical values.
    Removes any non-numeric characters and any spaces between numbers.
    Returns the cleaned numeric string, or the original value if it's not a string.
    """
    if isinstance(value_str, str):
        # Remove all non-numeric characters except spaces
        value_str = re.sub(r'[^\d\s]', '', value_str).strip()
        # Remove any whitespace between numbers
        value_str = re.sub(r'\s+', '', value_str)
        return value_str
    return value_str  # Return original value for non-string types (e.g., NaN)

# Clean and convert 'monthly_rent' and 'size' columns
data['monthly_rent'] = data['monthly_rent'].apply(clean_numeric_column)
data['size'] = data['size'].apply(clean_numeric_column)

# Convert cleaned columns to numeric (convert to integers and handle errors)
data['monthly_rent'] = pd.to_numeric(data['monthly_rent'], errors='coerce').astype('Int64')
data['size'] = pd.to_numeric(data['size'], errors='coerce').astype('Int64')

# Dropping unnecessary columns.
columns_to_drop = ['prop_name', 'completion_year', 'property_type', 'furnished', 
                   'facilities', 'additional_facilities', 'region', 'ads_id']

data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Handle missing values in 'parking' column by replacing NaN with 0
data['parking'] = data['parking'].fillna(0)

# Drop rows with any remaining missing values
data = data.dropna()

# Ensure 'rooms' is a string to filter out non-numeric values
data['rooms'] = data['rooms'].astype(str)

# Filter rows where 'rooms' is a digit and convert 'rooms' to an integer
data = data[data['rooms'].str.isdigit()]
data['rooms'] = data['rooms'].astype(int)

# Remove outliers based on logical conditions for 'monthly_rent' and 'size'
data = data[(data['monthly_rent'] >= 200) & (data['monthly_rent'] <= 20000) & 
            (data['size'] >= 200) & (data['size'] <= 20000)]

# Additional outlier removal: exclude properties with size > 6000 sqft but rent < 10,000
data = data[~((data['size'] > 6000) & (data['monthly_rent'] < 10000))]

#### Label Encoding for 'location' column ####
# Ensure that the 'location' column exists
if 'location' in data.columns:
    le = LabelEncoder()
    data['location_encoded'] = le.fit_transform(data['location'])
else:
    raise ValueError("The 'location' column is missing from the dataset.")

#### Train-Test Split ####
# Set 'monthly_rent' as the target variable
X = data.drop(columns=['monthly_rent', 'location'])  # We drop 'location' since we encoded it
y = data['monthly_rent']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### XGBoost Model Training ####
# Define the XGBoost regressor model with a squared error objective
regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the model using the training data
regressor.fit(X_train, y_train)

#### Model Evaluation ####
# Predict the test set results
y_pred = regressor.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print evaluation metrics
print(f'\nModel Performance:\nMean Absolute Error (MAE): {mae:.2f}\nRoot Mean Square Error (RMSE): {rmse:.2f}')

#### Save the Model and the Label Encoder ####
# Ensure the model and encoder are saved in a specific directory
save_dir = r'C:\Users\..'

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the XGBoost model
joblib.dump(regressor, os.path.join(save_dir, 'KLSelangorRent.pkl'))

# Save the LabelEncoder for 'location' column (you will need this for future predictions)
joblib.dump(le, os.path.join(save_dir, 'location_encoder.pkl'))

print(f"\nModel and LabelEncoder saved in {save_dir}")
