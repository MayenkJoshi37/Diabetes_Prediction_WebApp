# Importing necessary libraries
import joblib
import pandas as pd
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization, Activation, Dropout
# Loading dataset
df = pd.read_csv('App\Final_Dataset.xlsx')

# Loading scaling parameters for dataset
scaler = joblib.load('App\scaler.pkl')

# Separating features (X) and target variable (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

# Scaling features using loaded scaling parameters
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining the neural network model
def Model():
    model = Sequential([
        Dense(64, input_dim=8, kernel_regularizer=l2(0.001)),  # Defining input layer with 64 neurons, input dimension 8 (features), and L2 regularization
        BatchNormalization(),                                  # Applying batch normalization
        Activation('relu'),                                    # Applying ReLU activation function
        Dropout(0.4),                                          # Applying dropout with a rate of 0.4
        
        Dense(32, kernel_regularizer=l2(0.001)),               # Defining hidden layer with 32 neurons and L2 regularization
        BatchNormalization(),                                  # Applying batch normalization
        Activation('relu'),                                    # Applying ReLU activation function
        Dropout(0.4),                                          # Applying dropout with a rate of 0.4

        Dense(1, activation='sigmoid')                         # Defining output layer with sigmoid activation function for binary classification
    ])

    # Compiling the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Creating the model
model = Model()

# Training the model with best epoch, batch size, verbose and validation split values
history = model.fit(X_train_scaled, y_train , epochs=40, batch_size=512, verbose=2, validation_split=0.2)

model.summary()

# Saving the trained model
# Uncomment this line if you want to save the model again
# model.save(f'Model_3_0.001_Bn_relu_.4_0.00001_40_1024_LaReDrLrEpBs_1.h5')
