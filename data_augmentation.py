from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import uuid

datagen = image.ImageDataGenerator(
    rotation_range=40,  # (degree) randomly rotate image
    width_shift_range=0.2,  # (fraction of total width) randomly translate image horizontally
    height_shift_range=0.2,  # same above
    shear_range=0.2,  # shearing
    zoom_range=0.2,  # zooming inside image
    horizontal_flip=True,  # flip image horizontally
    fill_mode='nearest'  # filling in newly created pixels
)

# data_dir = "images"
# fnames = [fname for fname in os.listdir(data_dir)]
# print(len(fnames))
# img_path = fnames[0]
img_path = "images/test_1.jpg"
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    f = str(uuid.uuid4())
    plt.savefig(f)
    i += 1
    if i == 5:
        break


plt.show()
