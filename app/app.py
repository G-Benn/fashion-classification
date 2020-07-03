import streamlit as st 
from PIL import Image
import functions as f
from io import StringIO
import sys

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    
st.title("Classify apparel images using multiple Neural Nets!")
st.write("Models were trained on the Fashion-MNIST dataset.")


               
st.write("Loading models...")               
torch_model, tf_model_1, tf_model_2 = f.load_models()
# Do some munging to get the tf summaries to write correctly
old_stdout = sys.stdout
result = StringIO()
sys.stdout = result
tf_model_1.summary()
sys.stdout = old_stdout
summary_tf_1 = result.getvalue()

result = StringIO()
sys.stdout = result
tf_model_2.summary()
sys.stdout = old_stdout
summary_tf_2 = result.getvalue()

if st.checkbox('Show model info?'):
    st.write("Torch:")
    st.text(torch_model)
    st.write("-------------------------")
    st.write("Tensorflow mimicing PyTorch:")
    st.text(summary_tf_1)
    st.write("-------------------------")
    st.write("Tensorflow best model:")
    st.text(summary_tf_2)
    st.text("""
         'kernel_size': 3,\n
         'max_pool_size': 2,\n
         'pool_strides': 1,\n
         'filters': (6, 12),\n
         'dense_size': (120, 60),\n
         'learning_rate': 0.0005,\n
         'dropout_d': 0.2,\n
         'batch_size': 100,\n
         'epochs': 100,\n
         'shuffle': True,\n
         'callbacks': [[<tensorflow.python.keras.callbacks.EarlyStopping at 0x1760900ce10>]]""")
    
    st.write("==========================================================================================================")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    torch_prediction, tf_prediction_1, tf_prediction_2 = f.make_predictions(image, torch_model, tf_model_1, tf_model_2)
    
    torch_label, tf_label_1, tf_label_2 = f.get_label_from_predictions(torch_prediction), f.get_label_from_predictions(tf_prediction_1), f.get_label_from_predictions(tf_prediction_2)

    st.markdown(f"PyTorch thinks... **{torch_label}**")
    st.markdown(f"Tensorflow mimicing PyTorch thinks... **{tf_label_1}**")
    st.markdown(f"Tensorflow thinks... **{tf_label_2}**")
    st.write("------------")
    st.write("Creating plots....")
    
    st.write(f.plot_comparisons(f.process_image(image)[0], torch_prediction, tf_prediction_1, tf_prediction_2))
    
