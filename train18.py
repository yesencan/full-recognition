import tensorflow.keras as keras
from resnet_152 import resnet152_model
from resnet_50 import resnet50_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from classification_models.tfkeras import Classifiers

img_width, img_height = 224, 224
num_channels = 3
train_data = 'data/train'
valid_data = 'data/valid'
num_classes = 203
num_train_samples = 6618
num_valid_samples = 1671
verbose = 1
batch_size = 16
num_epochs = 100000
patience = 50

if __name__ == '__main__':
    # build a classifier model
    ResNet18, preprocess_input = Classifiers.get('resnet18')

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    valid_data_gen = ImageDataGenerator()
    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_accuracy', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_accuracy', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical')
    
    base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    # fine tune the model
    model.fit(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=valid_generator,
        validation_steps=num_valid_samples / batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose)
    # model.fit(train_generator, batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=verbose, validation_data=valid_generator)