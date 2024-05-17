system_dict = {
    "lds" : ("dynadojo.systems.lds", "LDSystem"),
    "lorenz": ("dynadojo.systems.lorenz", "LorenzSystem"),
    "lv_p": ("dynadojo.systems.lv", "PreyPredatorSystem"),
    "lv_c": ("dynadojo.systems.lv", "CompetitiveLVSystem"),
    "epi_1": ("dynadojo.systems.epidemic", "SIRSystem"),
    "epi_2": ("dynadojo.systems.epidemic", "SISSystem"),
    "epi_3": ("dynadojo.systems.epidemic", "SEISSystem"),
    "nbody": ("dynadojo.systems.santi", "NBodySystem"),
    "kura": ("dynadojo.systems.kuramoto", "KuramotoSystem"),
    "fbsnn_1": ("dynadojo.systems.fbsnn_pde", "BSBSystem"),
    "fbsnn_2": ("dynadojo.systems.fbsnn_pde", "HJBSystem"),
    "ctln": ("dynadojo.systems.ctln", "CTLNSystem"),
    "heat": ("dynadojo.systems.heat", "HeatEquation"),

}
algo_dict = {
    "lr" : ("dynadojo.baselines.lr", "LinearRegression"),
    "dnn" : ("dynadojo.baselines.dnn", "DNN"),
    "sindy": ("dynadojo.baselines.sindy", "SINDy"),
    "gru_rnn": ("dynadojo.baselines.gru_rnn", "GRU_RNN"),
}