from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers import BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import  Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
import datetime
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import keras.backend as K


def build_generator( ):
    # Image input
    gf = 64
    channels = 1

    def up_dim(vecs):
        img_A = vecs
        img_A = K.expand_dims(img_A, axis=1)
        img_A = K.expand_dims(img_A, axis=3)
        return img_A

    def down_dim(vec):
        img_A = vec
        img_A = K.reshape(img_A, [-1, 4096])
        return img_A

    def conv2d(layer_input, filters, f_size=(1, 2), bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=(2, 1), dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=(vgg_feature_shape,))
    d0_ = Lambda(up_dim)(d0)

    # Downsampling
    d1 = conv2d(d0_, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)
    d8 = conv2d(d7, gf * 8)

    # Upsampling
    u0 = deconv2d(d8, d7, gf * 8)
    u1 = deconv2d(u0, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)
    # u7 = deconv2d(u6, d1, gf)

    u8 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=(2, 1), strides=1, padding='valid', activation='tanh')(u8)
    #output_img = Flatten(name='flatten')(output_img)
    # output_img = tf.contrib.layers.flatten(output_img)
    output_img = Lambda(down_dim)(output_img)

    model = Model(d0, output_img)
    return model

def build_discriminator():
    img_shape = (1, 4096, 1)
    def d_layer(layer_input, filters, f_size=(1, 4), bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def up_dim(vecs):
        img_A, img_B = vecs
        img_A = K.expand_dims(img_A, axis=1)
        img_A = K.expand_dims(img_A, axis=3)
        img_B = K.expand_dims(img_B, axis=1)
        img_B = K.expand_dims(img_B, axis=3)
        combined_vgg_features = Concatenate(axis=1)([img_A, img_B])
        return combined_vgg_features

    img_A = Input(shape=(vgg_feature_shape,))
    img_B = Input(shape=(vgg_feature_shape,))

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Lambda(up_dim)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, f_size=(2, 1), bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)
    d5 = d_layer(d4, df * 16)

    # validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
    validity = tf.layers.conv2d(
            inputs=d5,
            filters=1,
            kernel_size=4,
            strides=(1, 1),
        padding='same',
            activation=tf.nn.relu,
        )


    model = Model([img_A, img_B], validity)
    return model

def validate( grd_descriptor, sat_descriptor):
    '''
    For final result verification
    '''
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    return accuracy

def l2_norm(vects):
    '''
    l2_normalize the features of A triple_model
    '''
    return K.l2_normalize(vects, axis=1)  # (,256)

def l2_norm_output_shape(shapes):
    shape1 = shapes
    return shape1

def build_single_triple(D_C_share_output_shape, code_length):
    input_data = Input(shape=(D_C_share_output_shape,))
    x = Dense(1024, name='dense1')(input_data)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Dense(2048, name='dense2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Dense(code_length, activation='relu')(x)
    prob_output = Lambda(l2_norm, output_shape=l2_norm_output_shape)(x)
    triple_model = Model(inputs=[input_data], outputs=[prob_output])
    return triple_model

def siams_distance(vects):
    sat_global, grd_global = vects
    dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
    pos_dist = tf.diag_part(dist_array)  #
    pair_n = triple_batch_size * (triple_batch_size - 1.0)
    # ground to satellite
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n
    # satellite to ground
    triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
    loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n
    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

def siams_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[0])

def siams_loss(y_true, y_pred):
    return K.sum(y_pred)



if __name__ == '__main__':
    epochs = 1
    batch_size = 64
    triple_epochs = 5
    triple_batch_size = 100
    # Input shape
    vgg_feature_shape = 4096
    pix_gan_input_shape = 4096
    code_length = 1024
    loss_weight = 10.0
    df = 64
    # 用于l2 归一化
    norm1 = Normalizer(norm='l2')
    # Guild and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='mse',  optimizer=Adam(0.0002, 0.5),  metrics=['accuracy'])
    # -------------------------
    # Construct Computational
    #   Graph of Generator
    # -------------------------
    # Guild the generator
    generator = build_generator()
    # Input images and their conditioning images
    vgg_feature_S = Input(shape=(vgg_feature_shape,))
    vgg_feature_G = Input(shape=(vgg_feature_shape,))

    # Gy conditioning on G generate a fake version of S
    fake_S = generator(vgg_feature_G)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # Discriminators determines validity of translated  / condition pairs
    valid = discriminator([fake_S, vgg_feature_G])
    combined = Model(inputs=[vgg_feature_S, vgg_feature_G], outputs=[valid, fake_S])
    combined.compile(loss=['mse', 'mae'], loss_weights=[1, 1],  optimizer=Adam(0.0002, 0.5))

    data = np.load('data_vgg/train_sat_vgg_feature.npz')
    train_sat = data['train_sat_vgg_feature']
    data = np.load('data_vgg/train_grd_vgg_feature.npz')
    train_grd = data['train_grd_vgg_feature']
    data = np.load('data_vgg/test_sat_vgg_feature.npz')
    test_sat = data['test_sat_vgg_feature']
    data = np.load('data_vgg/test_grd_vgg_feature.npz')
    test_grd = data['test_grd_vgg_feature']

    start_time = datetime.datetime.now()

    # Sdversarial loss ground truths
    half_batch_size = int(batch_size / 2)
    valid = np.ones((batch_size,)+(1,128,1))
    fake = np.zeros((batch_size,)+(1,128,1))
    nb_batches = int(train_grd.shape[0] / batch_size)
    for epoch in range(epochs):
        index = 0
        while index < nb_batches:
            vgg_features_G = train_grd[index * batch_size:(index + 1) * batch_size]
            vgg_features_S = train_sat[index * batch_size:(index + 1) * batch_size]

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Condition on G and generate a translated version
            fake_S = generator.predict(vgg_features_G)
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch([vgg_features_S, vgg_features_G], valid)
            d_loss_fake = discriminator.train_on_batch([fake_S, vgg_features_G], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # -----------------
            #  Train Generator
            # -----------------
            # Train the generators
            g_loss = combined.train_on_batch([vgg_features_S, vgg_features_G], [valid, vgg_features_S])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("[Epoch %d/%d] [Gatch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                  index, nb_batches,
                                                                                                  d_loss[0],
                                                                                                  100 * d_loss[1],
                                                                                                  g_loss[0],
                                                                                                  elapsed_time))
            index += 1
        ###############################################################################################################
        predict_sat = generator.predict(test_grd)
        l2_predict_sat = norm1.fit_transform(predict_sat)
        l2_test_sat = norm1.fit_transform(test_sat)
        print('compute accuracy')
        te_acc = validate(l2_predict_sat, l2_test_sat)
        with open('no_triple_pix2_gan_acc.txt', 'a') as file:
            file.write(str(epoch) + ' : ' + str(te_acc) + '\r\n')
        print('%d:*pix2_gan Accuracy on test set: %0.8f%%' % (epoch, 100 * te_acc))
        model_dir = 'generator_Model/' + str(epoch) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        generator.save(model_dir + 'generator.h5')
        discriminator.save(model_dir + 'discriminator.h5')

    # trian triple network
    single_triple_Model = build_single_triple(vgg_feature_shape, code_length=code_length)

    sat_tensor = Input(shape=(pix_gan_input_shape,))
    grd_tensor = Input(shape=(pix_gan_input_shape,))
    sat_output = single_triple_Model(sat_tensor)
    grd_output = single_triple_Model(grd_tensor)

    output_siams = Lambda(siams_distance, output_shape=siams_dist_output_shape)([sat_output, grd_output])
    triple_Model = Model(inputs=[sat_tensor, grd_tensor], outputs=[output_siams])
    triple_Model.compile(loss=siams_loss, loss_weights=[1], optimizer=Adam(lr=0.0001))

    train_grd_gan_feature = generator.predict(train_grd)
    train_sat_gan_feature = train_sat
    test_grd_gan_feature = generator.predict(test_grd)
    test_sat_gan_feature = test_sat

    # triple network leanring starts here
    for triple_epoch in range(triple_epochs):
        print('triple network Epoch {} of {}'.format(triple_epoch + 1, triple_epochs))
        nb_batches = int(train_grd_gan_feature.shape[0] / triple_batch_size)
        print(train_grd_gan_feature.shape[0])
        epoch_triple_loss = []
        index = 0

        while index < nb_batches:
            grd_data_train = train_grd_gan_feature[index * triple_batch_size:(index + 1) * triple_batch_size]
            sat_data_train = train_sat_gan_feature[index * triple_batch_size:(index + 1) * triple_batch_size]
            triple_label = np.array([0] * grd_data_train.shape[0])

            triple_loss = triple_Model.train_on_batch([sat_data_train, grd_data_train], triple_label)
            epoch_triple_loss.append(triple_loss)
            index += 1

        print('\n[Loss_triple: {:.8f}'.format(np.mean(epoch_triple_loss)))
        sat_output1 = single_triple_Model.predict(test_sat_gan_feature)
        grd_output1 = single_triple_Model.predict(test_grd_gan_feature)
        # test accuracy
        print('compute accuracy')
        te_acc = validate(grd_output1, sat_output1)
        with open('pixgan_triple_accuracy.txt', 'a') as file:
            file.write(str(triple_epoch) + ' : ' + str(te_acc) + '\r\n')
        print('%d:* pixgan_triple Accuracy on test set: %0.8f%%' % (triple_epoch, 100 * te_acc))
        model_dir = 'tri_Model/' + str(triple_epoch) + '/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        triple_Model.save(model_dir + 'model.h5')
        single_triple_Model.save(model_dir + 'single_triple_Model.h5')
