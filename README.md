# order-cancellation-prediction

Applied machine learning project to predict order cancellations using Logistic Regression with SHAP explainability.

Includes:
- Imbalanced classification handling
- Model comparison
- SHAP interpretation
- Simple prediction script

-------------------------------------------------------------------------------------------
FOR USERS TO USE IT

STEP 1: Get the Code
  1- Download or clone the GitHub repository containing:
  prediction_model.py
  orders_cleaned.csv
  2- Make sure both files are in the same folder.

STEP 2: Prepare the Input Data (orders_cleaned.csv)
  The CSV must contain these columns:
  
  customer_collects
  is_a_repeat_order
  lead_time
  n_customer_notes
  n_items_above_quantity_10
  n_listed_addresses
  n_listed_payment_methods
  n_previous_cancelled_orders
  n_previous_completed_orders
  n_small_items
  payment_type → values: credit or debit
  total_price
  status → values: Canceled or Not_Canceled

STEP 3: Install Required Libraries

STEP 4: Run the Prediction Script: prediction_model.py

STEP 5: Understand the Output

EXPECTED OUTPUT ----------------------
--- Predicted Order Cancellations (Highest Risk First) ---
 cancellation_risk_score predicted_status
 0.91                    Canceled
 0.87                    Canceled
 0.62                    Canceled
 0.48                    Not_Canceled
