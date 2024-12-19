from prep_data import set_train_data
from model import build_model, train_model
from visualize import visualize_history
import os


data_dir = 'archive'
train_data_dir = os.path.join(data_dir, 'train')

num_classes = 43

x_train, y_train, x_val, y_val = set_train_data(
    train_data_dir, num_classes)

input_shape = x_train.shape[1:]

model = build_model(input_shape, num_classes)
model, history = train_model(model, x_train, y_train, x_val, y_val)

visualize_history(history)

model_save_path = 'trafik_levhasi_modeli.h5'
model.save(model_save_path)
print(f"Model '{model_save_path}' dosyasına kaydedildi.")