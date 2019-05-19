from keras.models import Model
from keras.layers import *



nfeatures = 6
timesteps = 16

#function to split the input in multiple outputs
def splitter(x):
    return [x[:,:,i:i+1] for i in range(nfeatures)]


#model's input tensor
inputs = Input((timesteps,nfeatures))

#splitting in 128 parallel tensors - 128 x (batch,15,1)
multipleFeatures = Lambda(splitter)(inputs)

# # original
# #applying one individual convolution on each parallel branch
# multipleFeatures = [
#    Conv1D(padding = 'valid',filters = 1,strides = 5, kernel_size = 5)(feat)
#    for feat in multipleFeatures ]

# modified
#applying one individual convolution on each parallel branch
multipleFeatures_new = []
for feat in multipleFeatures:
    multipleFeatures_new.append(Conv1D(padding = 'valid',filters = 1,strides = 5, kernel_size = 5)(feat))

#joining the branches into (batch, 3, 128)
joinedOutputs = Concatenate()(multipleFeatures_new)
joinedOutputs = Activation('relu')(joinedOutputs)

outputs = MaxPooling1D()(joinedOutputs)
outputs = Lambda(lambda x: K.squeeze(x,axis=1))(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(inputs, outputs)
pass
