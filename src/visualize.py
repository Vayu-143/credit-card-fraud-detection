import matplotlib.pyplot as plt

def plot_feature_importance(model):
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(range(len(importance)), importance)
    ax.set_title("Feature Importance")

    return fig