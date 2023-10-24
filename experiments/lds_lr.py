import os
import pandas as pd
from dynadojo.challenges import FixedError, FixedComplexity, FixedTrainSize
from dynadojo.baselines import LinearRegression
from dynadojo.systems import LDS

s = "lds"
m = "lr"

def run_fc(
        reps = 5, 
        in_dist=True, 
        output_dir="experiments/lds_lr", 
        seed=0
        ):
    l = 5
    control = False
    add_more = False
    challenge = FixedComplexity(
        N=[10, 100, 1000],
        l=l,
        t=50,
        reps=reps,
        test_examples=50,
        test_timesteps=50,
        system_cls=LDS
        )

    data = challenge.evaluate(
        model_cls=LinearRegression, 
        id=f"{m} ({l}, {in_dist})", 
        noisy=True, 
        in_dist=in_dist
    )

    file = f"{output_dir}/fc/{s}_{m}_{l=}_{in_dist=}.csv"
    assert os.path.exists(file)
    data = pd.read_csv(file)

    df = data #.head(4*5 *2)
df=df.dropna(subset=['loss','loss_out_dist'])
display(df['n'].value_counts())
df = df.reset_index().melt(['n', 'rep'], ["loss", "loss_out_dist"], var_name='dist',  value_name='Loss')

print(file)
g = plot_metric(df, "n", "Loss", hue="dist", xlabel=r'$n$', ylabel=r'$loss$', log=True, errorbar=("pi", 50))
g._legend.set_title("Dist")
# replace labels
new_labels = ['in', 'out']
for t, nl in zip(g._legend.texts, new_labels):
    t.set_text(nl)
sns.move_legend(g, "upper right",
    bbox_to_anchor=(0.8, .95), ncol=1, title=None, frameon=False,)


# if control:
#     file = f"{path}/{s}/{m}_fc_{l=}_{in_dist=}_CONTROL.csv"
#     assert os.path.exists(file)
#     data = pd.read_csv(file)

#     df = data #.head(4*5 *2)
#     df=df.dropna(subset=['loss','loss_out_dist'])
#     display(df['n'].value_counts())
#     df = df.reset_index().melt(['n', 'rep'], ["loss", "loss_out_dist"], var_name='dist',  value_name='Loss')

#     print(file)
#     g = plot_metric(df, "n", "Loss", hue="dist", xlabel=r'$n$', ylabel=r'$loss$', log=True, errorbar=("pi", 50))
#     g._legend.set_title("Dist")
#     # replace labels
#     new_labels = ['in', 'out']
#     for t, nl in zip(g._legend.texts, new_labels):
#         t.set_text(nl)

# g.figure.savefig(f"{path}/{s}/{m}_fc_{l=}_{in_dist=}.pdf", bbox_inches='tight')