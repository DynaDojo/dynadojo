# DynaDojo


## Example

```
import dynadojo as dd
import numpy as np

latent_dim = 2
embed_dim = 50
n = 5000
timesteps = 50
challenge = dd.systems.LDSSystem(latent_dim, embed_dim)
x0 = challenge.make_init_conds(n)
y0 = challenge.make_init_conds(30, in_dist=False)
x = challenge.make_data(x0, control=np.zeros((n, timesteps, embed_dim)), timesteps=timesteps)
y = challenge.make_data(y0, control=np.zeros((n, timesteps, embed_dim)), timesteps=timesteps)
dd.utils.lds.plot([x, y], target_dim=3, labels=["in", "out"], max_lines=30)
```
