import numpy as np
import pandas as pd
import tensorflow as tf

# Check available devices
print(tf.config.list_physical_devices('CPU'))
print("TensorFlow version:", tf.__version__)

# Load the dataset
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")  # Training data
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")  # Evaluation data

# Separate target column
y_train = dftrain.pop("survived")  # Target for training
y_eval = dfeval.pop("survived")  # Target for evaluation

# Define categorical and numeric columns
categorical_columns = ["sex", "n_siblings_spouses", "class", "deck", "embark_town", "alone", 'parch']
numeric_columns = ["age", "fare"]

# Preprocessing for numeric columns
numeric_inputs = {name: tf.keras.layers.Input(name=name, shape=(1,), dtype=tf.float32) for name in numeric_columns}

# Convert all categorical columns to strings in the dataset
for col in categorical_columns:
    dftrain[col] = dftrain[col].astype(str)
    dfeval[col] = dfeval[col].astype(str)

# Define the categorical inputs and corresponding layers
categorical_inputs = {}
categorical_layers = []  # Initialize categorical_layers here

for feature_name in categorical_columns:
    unique_values = dftrain[feature_name].unique().astype(str)
    categorical_inputs[feature_name] = tf.keras.layers.Input(name=feature_name, dtype=tf.string, shape=(1,))

    # Define the StringLookup layer with the unique values as string
    lookup = tf.keras.layers.StringLookup(vocabulary=unique_values.tolist(), mask_token=None)
    encoding = tf.keras.layers.CategoryEncoding(num_tokens=len(unique_values) + 1, output_mode="binary")
    categorical_layers.append(tf.keras.Sequential([lookup, encoding]))

# Combine inputs for numeric and categorical features
inputs = {**numeric_inputs, **categorical_inputs}

# Apply categorical layers and combine them with numeric inputs
x = []
for key in inputs:
    if key in categorical_columns:
        index = categorical_columns.index(key)
        x.append(categorical_layers[index](inputs[key]))  # Apply corresponding categorical layer
    else:
        x.append(inputs[key])  # Directly use numeric inputs

# Combine all features
x = tf.keras.layers.concatenate(x)

# Define the output layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create and compile the model
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Convert the datasets into TensorFlow datasets
def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
    dataframe = dict(dataframe)
    dataset = tf.data.Dataset.from_tensor_slices((dataframe, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    return dataset

# Prepare the datasets
batch_size = 32
train_dataset = df_to_dataset(dftrain, y_train, shuffle=True, batch_size=batch_size)
eval_dataset = df_to_dataset(dfeval, y_eval, shuffle=False, batch_size=batch_size)

# Train the model
model.fit(train_dataset, epochs=20)

# Evaluate the model
result = model.evaluate(eval_dataset)

print("Accuracy:", result[1])

# Predict on the evaluation dataset
predictions = list(model.predict(eval_dataset))

# Get the features for the first entry from the original dataframe
first_entry_features = dftrain.iloc[1][categorical_columns + numeric_columns]

# Get the first prediction (probability for the first entry)
first_prediction = predictions[1][0]  # This is the predicted probability for the first entry

# Print the features along with the predicted probability
print("First entry features:")
for feature_name, feature_value in first_entry_features.items():
    print(f"{feature_name}: {feature_value}")

# Log the predicted probability for the first entry
print(f"Predicted probability for the second entry: {first_prediction * 100}%")
print('Actual value for the second entry:', y_eval.iloc[4])