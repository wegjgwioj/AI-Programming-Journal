def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    import joblib
    return joblib.load(filepath)

def plot_results(predictions, actuals, title='Predictions vs Actuals'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predictions', color='blue')
    plt.plot(actuals, label='Actuals', color='orange')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()