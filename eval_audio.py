import numpy as np
import soundfile as sf
import torch

import utils

device = 'cpu'

def get_data(dataset):
    consts = utils.Constants(dataset)
    test_spec = np.load('data/%s/%s.npz' % (dataset, 'test'))['spec']
    test_Y = np.abs(test_spec[:,0:1])
    test_B = np.abs(test_spec[:,1:2])
    print ('dataset shape', test_spec.shape)
    return  test_Y, test_B, test_spec, consts

dataset = 'speech'
test_Y, test_B, test_complex_spectrogram, consts = get_data(dataset)

model = torch.load('model-9',map_location='cpu').to(device)


num_iterations = 10
num_epochs = 25 # Every iteration this number of epochs of training
batchsize = 64

num_test_batches = test_Y.shape[0] // batchsize
test_Y = test_Y[:num_test_batches * batchsize]
test_B = test_B[:num_test_batches * batchsize]

test_complex_spectrogram = test_complex_spectrogram[:num_test_batches * batchsize]
B_test_ground_truth_audio = utils.spec2aud(test_complex_spectrogram[:, 1].transpose((1, 0, 2)).reshape((257, -1)), consts)
X_test_ground_truth_audio = utils.spec2aud(test_complex_spectrogram[:, 2].transpose((1, 0, 2)).reshape((257, -1)), consts)
Y_test_ground_truth_audio = utils.spec2aud(test_complex_spectrogram[:, 0].transpose((1, 0, 2)).reshape((257, -1)), consts)

sf.write("B_test.wav", B_test_ground_truth_audio, consts.SR)
sf.write("X_test.wav", X_test_ground_truth_audio, consts.SR ) # consts.SR
sf.write("Y_test.wav", Y_test_ground_truth_audio, consts.SR)

noize = sf.read("data/speech/noises/5.wav")
clear_speech = sf.read("data/speech/clear_speech/5.wav")

spec_abs, spec = utils.aud2spec((Y_test_ground_truth_audio,1), consts)

y_spec_abs, y_spec = utils.create_merge_speech_noize(clear_speech, noize, consts)

test_mask = []
test_batch_Y = spec_abs
test_batch_B = test_B
test_batch_Y = torch.from_numpy(test_batch_Y).to(device).float()
test_batch_B = torch.from_numpy(test_batch_B).to(device).float()
batch_mask = model(test_batch_Y)
test_mask.append(batch_mask.cpu().data.numpy())

test_mask = np.concatenate(test_mask, axis=0).squeeze()
B_test_prediction_spectrogram = spec[:, 0] * test_mask
X_test_prediction_spectrogram = spec[:, 0] * (1 - test_mask)

B_pred_audio = utils.spec2aud(B_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)
X_pred_audio = utils.spec2aud(X_test_prediction_spectrogram.transpose((1, 0, 2)).reshape((257, -1)), consts)

sf.write("B_test_pred.wav", B_pred_audio, consts.SR)
sf.write("X_test_pred.wav", X_pred_audio, consts.SR)