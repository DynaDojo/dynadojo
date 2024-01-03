You can download the source image `here <https://github.com/FlyingWorkshop/dynadojo/blob/main/graphics/dino.png>`_.

.. code:: python
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    def grayscale(png_path, basewidth=30):
        image = Image.open(png_path)
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        grayscale_image = image.convert("L")
        grey = np.array(grayscale_image)
        return grey

    dino = grayscale("../graphics/dino.png")
    print(dino.shape)
    latent_dim = dino.shape[0] ** 2
    plt.imshow(dino, cmap='gray')
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

.. image:: ../_images/bw_dino.png

.. code:: python

    from dynadojo.systems.heat import HeatEquation
    from dynadojo.wrappers import SystemChecker
    from dynadojo.utils.heat import plot

    # Create the init conds from the image
    x0 = np.expand_dims(dino.flatten()[::-1], axis=0)
    x0 *= 2

    n = 1
    timesteps = 20
    embed_dim = latent_dim
    system = SystemChecker(HeatEquation(latent_dim, embed_dim, noise_scale=0))
    x = system.make_data(x0, timesteps=timesteps)
    plot(x[0], timesteps, system._system.dt, savefile="../graphics/hot_dino.gif")

.. image:: ../_images/hot_dino.gif
