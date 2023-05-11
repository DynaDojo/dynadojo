from dynascale.utils.lds import plot
from dynascale.challenges.lds import LDSChallenge
from dynascale.baselines.new_koopman import Koopman


def main():
    latent_dim = 4
    embed_dim = 5
    n = 5000
    timesteps = 50
    challenge = LDSChallenge(latent_dim=latent_dim, embed_dim=embed_dim)
    x0 = challenge.make_init_conds(n)
    y0 = challenge.make_init_conds(n, in_dist=False)
    x = challenge.make_data(x0, timesteps=timesteps)
    y = challenge.make_data(y0, timesteps=timesteps)

    model = Koopman(latent_dim=latent_dim, embed_dim=embed_dim, timesteps=timesteps)
    model.fit(x)


if __name__ == "__main__":
    main()
