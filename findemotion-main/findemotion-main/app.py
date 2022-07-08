import streamlit as st 
import soundfile as sf
import io
import keras
import numpy as np
import librosa


st.title('Speech Emotion Recognition')

uploaded_file=st.file_uploader('Choose Audio file',type=['wav','ogg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    #print(uploaded_file.name)
    st.audio(bytes_data, format='audio/ogg')
    #scipy.io.wavfile.write(uploaded_file, 16000, bytes_data)
    data, sampling_rate = sf.read(io.BytesIO(bytes_data))
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    loaded_model = keras.models.load_model('SER_model.h5')
    predict_x=loaded_model.predict(x)
    classes_x=np.argmax(predict_x,axis=1)
    label_conversion = {'0': 'Neutral',
                            '1': 'Calm',
                            '2': 'Happy',
                            '3': 'Sad',
                            '4': 'Angry',
                            '5': 'Fearful',
                            '6': 'Disgust',
                            '7': 'Surprised'}

    for key, value in label_conversion.items():
        if int(key) == classes_x:
            label = value 
    st.header('Predicted Emotion:',)
    st.subheader(label)
