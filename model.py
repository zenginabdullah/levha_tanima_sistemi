from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential

def build_model(input_shape, num_classes,
                kernel_size = (3, 3), 
                pool_size = (2, 2), 
                dropout_rate = 0.3,
                padding = 'same'):  # Padding parametresi burada tanımlandı
    
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = kernel_size, padding = padding, 
                     input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, kernel_size = kernel_size, padding = padding))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, kernel_size = kernel_size, padding = padding))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())  
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
    
    return model

model = build_model((64,64,3),43)
model.summary()


def train_model(model, 
                x_train, y_train, 
                x_val, y_val, 
                batch_size = 16, 
                epochs = 10):
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(x_val, y_val))
    return model, history   