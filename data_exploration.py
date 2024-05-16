import numpy as np
import matplotlib.pyplot as plt


# import os
# folder = "./data"
# filenames = sorted(os.listdir(folder))

# data = []

# for i, filename in enumerate(filenames):
#     if filename.endswith('.npy'):

#         image = np.load(os.path.join(folder, filename))
#         data.append(image)
# np.save("./data/imgs", np.array(data))


data = np.load("./imgs.npy")

print(data[0].shape)


def plot_imgs(dataset, i, cols):
    fig, axes = plt.subplots(2, cols, sharex='all', sharey='all', figsize=(14,9))
    plt.axis('off')

    axes = axes.flatten()

    for j in range(0, len(axes)):
        print(dataset[i+j].transpose(2,1,0).shape)
        axes[j].imshow(dataset[i+j].transpose(2,1,0))
    
    plt.tight_layout()
    plt.show()

    
cols = 4

# for i in range(len(data)):
#     plot_imgs(data, i, cols)