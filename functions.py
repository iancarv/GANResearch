import data
from gan import GAN, Config
from utils import test_model_metrics, test_model_from_results

def train_gan(config=None, dataset=None):
    if dataset is None:
        dataset = data.load_tmi_data()

    X_train, y_train, X_test, y_test = dataset
    if config is None:
        config = Config(nb_epochs=15, channels=3, num_classes=2)

    gan = GAN(config)
    train_history, test_history = gan.train(X_train, y_train, X_test, y_test)
    pickle.dump({'train': train_history, 'test': test_history},
                open('output/acgan-history.pkl', 'wb'))

def basic_gan():
    config = Config(nb_epochs=15, channels=3, num_classes=2)
    gan = GAN(config)
    gan.generator.load_weights('output/params_generator_epoch_014.hdf5')
    gan.discriminator.load_weights('output/params_discriminator_epoch_014.hdf5')
    return gan

def run_model_metrics(gan=None):
    if gan is None:
        gan = basic_gan()

    aveP, avePred, all_tests, all_scores, all_preds, results = test_model_metrics(gan, 'data/out')
    outfile = open('output/metrics.pkl','wb')
    pickle.dump({
        'aveP': aveP, 
        'avePred': avePred,
        'all_tests': all_tests,
        'all_scores': all_scores,
        'all_preds': all_preds,
        'results': results
    },outfile)
    outfile.close()

    return results

def run_model_results(results, thresh_nms=0.3):
    test_model_from_results('data/out', results, thresh_nms=thresh_nms)

def run_validation_data(gan=None):
    if gan is None:
        gan = basic_gan()

    X_train, y_train, X_test, y_test = data.load_tmi_data()
    gan.predict(X_test, y_test)
    X_test, y_test = data.load_nuclei_data()
    gan.predict(X_test, y_test)
