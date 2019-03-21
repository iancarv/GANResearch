from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
from tensorflow import set_random_seed
import numpy as np
from loss import modified_binary_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score
import random
import data
from matplotlib import pyplot as plt
from utils import test_model_metrics, test_model_from_results

K.set_image_dim_ordering('th')

from collections import namedtuple

class Config(object):
    def __init__(self, **kwargs):
        self.source = dict(kwargs)
        self.seed = kwargs.get('seed', 1331)
        self.nb_epochs = kwargs.get('nb_epochs', 50)
        self.batch_size = kwargs.get('batch_size', 100)
        self.latent_size = kwargs.get('latent_size', 100)
        self.adam_lr = kwargs.get('adam_lr', 0.0002)
        self.adam_beta_1 = kwargs.get('adam_beta_1', 0.5)
        self.disc_loss = kwargs.get('disc_loss', [modified_binary_crossentropy, 'sparse_categorical_crossentropy'])
        self.disc_opt = kwargs.get('disc_opt')
        self.gen_loss = kwargs.get('gen_loss', 'binary_crossentropy')
        self.gen_opt = kwargs.get('gen_opt', Adam(lr=self.adam_lr, beta_1=self.adam_beta_1))
        self.combined_opt = kwargs.get('combined_opt', 'RMSprop')
        self.combined_loss = kwargs.get('combined_opt', [modified_binary_crossentropy, 'sparse_categorical_crossentropy'])
        
        if self.disc_opt is None or self.disc_opt == "SGD":
            self.sgd_clipvalue = kwargs.get('sgd_clipvalue', 0.01)
            self.disc_opt = SGD(clipvalue=self.sgd_clipvalue)
        elif self.disc_opt == "Adam":
            self.disc_opt = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1)

        self.img_rows = kwargs.get('img_rows', 28)
        self.img_cols = kwargs.get('img_cols', 28)
        self.channels =kwargs.get('channels',  1)
        self.num_classes =kwargs.get('num_classes',  10)


class GAN(object):
    def __init__(self, config):
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        set_random_seed(seed)
        self.config = config

        self.img_rows = config.img_rows
        self.img_cols = config.img_cols
        self.channels = config.channels
        self.num_classes = config.num_classes

        # batch and latent size taken from the paper
        self. nb_epochs = config.nb_epochs
        self.batch_size = config.batch_size
        self.latent_size = config.latent_size

        # Adam parameters suggested in https://arxiv.org/abs/1511.06434
        adam_lr = config.adam_lr
        adam_beta_1 = config.adam_beta_1

        sgd_clipvalue = config.sgd_clipvalue

        # build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer=config.disc_opt,
            loss=config.disc_loss
        )

        # build the generator
        self.generator = self.build_generator(self.latent_size)
        self.generator.compile(optimizer=config.gen_opt,
                          loss=config.gen_loss)

        latent = Input(shape=(self.latent_size, ))
        image_class = Input(shape=(1,), dtype='int32')

        # get a fake image
        fake = self.generator([latent, image_class])

        # we only want to be able to train generation for the combined model
        self.discriminator.trainable = False
        fake, aux = self.discriminator(fake)
        self.combined = Model(input=[latent, image_class], output=[fake, aux])

        self.combined.compile(
            optimizer=config.combined_opt,
            loss=config.combined_loss
        )

        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)

    def predict_proba(self, X_test):
        return self.discriminator.predict(X_test)

    def build_generator(self, latent_size):
        # we will map a pair of (z, L), where z is a latent vector and L is a
        # label drawn from P_c, to image space (..., 1, 28, 28)
        cnn = Sequential()

        cnn.add(Dense(1024, input_dim=latent_size))
        cnn.add(LeakyReLU())
        cnn.add(Dense(128 * 7 * 7))
        cnn.add(LeakyReLU())
        cnn.add(Reshape((128, 7, 7)))

        # upsample to (..., 14, 14)
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                              init='glorot_uniform'))
        cnn.add(LeakyReLU())

        # upsample to (..., 28, 28)
        cnn.add(UpSampling2D(size=(2, 2)))
        cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                              init='glorot_uniform'))
        cnn.add(LeakyReLU())

        # take a channel axis reduction
        cnn.add(Convolution2D(self.channels, 2, 2, border_mode='same',
                              activation='tanh', init='glorot_uniform'))

        # this is the z space commonly refered to in GAN papers
        latent = Input(shape=(latent_size, ))

        # this will be our label
        image_class = Input(shape=(1,), dtype='int32')

        cls = Flatten()(Embedding(self.num_classes, latent_size,
                                  init='glorot_uniform')(image_class))

        # hadamard product between z-space and a class conditional embedding
        multiply_layer = Multiply()
        h = multiply_layer([latent, cls])

        fake_image = cnn(h)

        return Model(input=[latent, image_class], output=fake_image)

    def build_discriminator(self):
        # build a relatively standard conv net, with LeakyReLUs as suggested in
        # the reference paper
        cnn = Sequential()
        img_shape = (self.channels, self.img_rows, self.img_cols)
        #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
        cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                              input_shape=img_shape))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        image = Input(shape=img_shape)

        features = cnn(image)

        # first output (name=generation) is whether or not the discriminator
        # thinks the image that is being shown is fake, and the second output
        # (name=auxiliary) is the class that the discriminator thinks the image
        # belongs to.
        fake = Dense(1, activation='linear', name='generation')(features)
        aux = Dense(self.num_classes, activation='softmax', name='auxiliary')(features)

        return Model(input=image, output=[fake, aux])

    def predict(self, X_test, y_test, display=True):

        # Generating a predictions from the discriminator over the testing dataset
        y_pred = self.discriminator.predict(X_test)
#         y_score = 
        print(y_pred[1])
        # Formating predictions to remove the one_hot_encoding format
        y_scores = y_pred[1]
        row_sums = y_scores.sum(axis=1)
        y_score = y_scores[:,1]
        y_pred = np.argmax(y_scores, axis=1)
        
        if display:
            print ('\nOverall accuracy: %f%% \n' % (accuracy_score(y_test, y_pred) * 100))
            print ('\nAveP Score: %f%% \n' % (average_precision_score(y_test, y_score) * 100))
            print ('\nAveP Preds: %f%% \n' % (average_precision_score(y_test, y_pred) * 100))
            print('Max confidence', np.max(y_score))
            print('Min confidence', np.min(y_score))

            # Calculating and ploting a Classification Report
            class_names = ['Non-nunclei', 'Nuclei']
            print('Classification report:\n %s\n'
                % (classification_report(y_test, y_pred, target_names=class_names)))

            # Calculating and ploting Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print('Confusion matrix:\n%s' % cm)
        
        return y_pred, y_scores


    def train(self, X_train, y_train, X_test, y_test):
        img_shape = (self.channels, self.img_rows, self.img_cols)
        nb_train, nb_test = X_train.shape[0], X_test.shape[0]

        nb_epochs = self.nb_epochs
        batch_size = self.batch_size

        for epoch in range(nb_epochs): 
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

            nb_batches = int(X_train.shape[0] / batch_size)
            progress_bar = Progbar(target=nb_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for index in range(nb_batches):
                if len(epoch_gen_loss) + len(epoch_disc_loss) > 1:
                    progress_bar.update(index, values=[('disc_loss',np.mean(np.array(epoch_disc_loss),axis=0)[0]), ('gen_loss', np.mean(np.array(epoch_gen_loss),axis=0)[0])])
                else:
                    progress_bar.update(index)
                # generate a new batch of noise
                #noise = np.random.uniform(-1, 1, (batch_size, latent_size))
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_size))

                # get a batch of real images
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]

                # sample some labels from p_c
                sampled_labels = np.random.randint(0, config.num_classes, batch_size)

                # generate a batch of fake images, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (batch_size, 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_images = self.generator.predict(
                    [noise, sampled_labels.reshape((-1, 1))], verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = np.array([-1] * batch_size + [1] * batch_size)
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(self.discriminator.train_on_batch(X, [y, aux_y]))

                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator
                #noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
                noise = np.random.normal(0, 1, (2 * self.batch_size, self.latent_size))
                sampled_labels = np.random.randint(0, config.num_classes, 2 * self.batch_size)

                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = -np.ones(2 * batch_size)

                epoch_gen_loss.append(self.combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

            print('\nTesting for epoch {}:'.format(epoch + 1))
            self.predict(X_test, y_test)
            # evaluate the testing loss here

            # generate a new batch of noise
            #noise = np.random.uniform(-1, 1, (nb_test, latent_size))
            noise = np.random.normal(0, 1, (nb_test, self.latent_size))

            # sample some labels from p_c and generate images from them
            sampled_labels = np.random.randint(0, config.num_classes, nb_test)
            generated_images = self.generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=False)

            X = np.concatenate((X_test, generated_images))
            y = np.array([1] * nb_test + [0] * nb_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            discriminator_test_loss = self.discriminator.evaluate(
                X, [y, aux_y], verbose=False)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            # make new noise
            #noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
            noise = np.random.normal(0, 1, (2 * nb_test, self.latent_size))
            sampled_labels = np.random.randint(0, config.num_classes, 2 * nb_test)

            trick = np.ones(2 * nb_test)

            generator_test_loss = self.combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels], verbose=False)

            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            self.train_history['generator'].append(generator_train_loss)
            self.train_history['discriminator'].append(discriminator_train_loss)

            self.test_history['generator'].append(generator_test_loss)
            self.test_history['discriminator'].append(discriminator_test_loss)

            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                'component', *self.discriminator.metrics_names))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
            print(ROW_FMT.format('generator (train)',
                                 *self.train_history['generator'][-1]))
            print(ROW_FMT.format('generator (test)',
                                 *self.test_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                                 *self.train_history['discriminator'][-1]))
            print(ROW_FMT.format('discriminator (test)',
                                 *self.test_history['discriminator'][-1]))
            # save weights every epoch
            self.generator.save_weights(
                'output/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
            self.discriminator.save_weights(
                'output/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

            # generate some digits to display
            #noise = np.random.uniform(-1, 1, (100, latent_size))
            noise = np.random.normal(-1, 1, (10 * config.num_classes, self.latent_size))

            sampled_labels = np.array([
                [i] * 10 for i in range(config.num_classes)
            ]).reshape(-1, 1)

            # get a batch to display
            generated_images = self.generator.predict(
                [noise, sampled_labels], verbose=0)

            generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
            generated_images = np.transpose(generated_images, (0,3,2,1))

            r = self.config.num_classes
            c = 10
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(generated_images[cnt, :,:])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig('output/plot_epoch_{0:03d}_generated.png'.format(epoch))
            plt.close()
        return self.train_history, self.test_history

if __name__ == '__main__':
    # epoch = 0
    # X_train, y_train, X_test, y_test = data.load_tmi_data()
    config = Config(nb_epochs=15, channels=3, num_classes=2)
    gan = GAN(config)
    # train_history, test_history = gan.train(X_train, y_train, X_test, y_test)
    # pickle.dump({'train': train_history, 'test': test_history},
    #             open('output/acgan-history.pkl', 'wb'))

    # aveP, avePred, all_tests, all_scores, all_preds, results = test_model_metrics(gan, 'data/out')
    # outfile = open('output/metrics.pkl','wb')
    # pickle.dump({
    #     'aveP': aveP, 
    #     'avePred': avePred,
    #     'all_tests': all_tests,
    #     'all_scores': all_scores,
    #     'all_preds': all_preds,
    #     'results': results
    # },outfile)
    # outfile.close()

    gan.generator.load_weights('output/params_generator_epoch_014.hdf5')
    gan.discriminator.load_weights('output/params_discriminator_epoch_014.hdf5')
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
    
    highest = 0
    hi_thresh = 0
    decreased = True
    path = 'output/metrics.pkl'
    pickle_in = open(path,"rb")
    full_result = pickle.load(pickle_in)
    results = full_result.get('results', full_result)
    aveP, avePred, all_tests, all_scores, all_preds, results = test_model_from_results('data/out', results, 0.3, True)
        
    print('Highest AveP', aveP)
    print('Highest AveP', avePred)
