import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np 
from googletrans import Translator


# Create Translator object
translator = Translator()

def translator_text(text,target_language):
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text


language_options = {
   
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Hindi": "hi",
    "Japanese": "ja",
}
list_language=[]

list_language=list(language_options.keys())
list_language.insert(0,"Select a Language")
language = st.selectbox("", list_language)
if language=="Select a Language":
    target_language='en'
else:
     target_language=language

# Define classes and corresponding descriptions for each plant
plants_info = {
    translator_text('Potato',target_language) : {
        'model_path': 'checkpoints.keras',
        'classes': [ 'Early Blight' ,'Late Blight' ,'Healthy' ],
        'descriptions': {
            
            'Early Blight' : 'Early blight, caused by the fungus Alternaria solani, \
            typically appears as dry, brown lesions on the lower leaves of the plant. \
            It can lead to reduced yield if not controlled.',
            'Late Blight' : 'Late blight, caused by the fungus Phytophthora infestans, \
            manifests as dark, water-soaked lesions on the leaves and stems of the plant. \
            It spreads rapidly and can cause significant damage to potato crops.',
            'Healthy' : 'The plant appears to be healthy with no signs of disease.'
        },
        'medicine_links': {
           
            'Early Blight' : 'https://example.com/early_blight_medicine',
            'Late Blight' : 'https://example.com/late_blight_medicine',
             'Healthy' : None
        }
    },
   
    translator_text('Grape',target_language) : {
        'model_path': 'grape_model.keras',
        'classes': ['Black Rot','Esca (Black Measles)','Grape Leaf blight (Isariopsis Leaf Spot)','Healthy'],
        'descriptions': {
            'Black Rot':'Grape black rot is a fungal disease caused by an ascomycetous fungus, Guignardia bidwellii, that attacks grape vines during hot and humid weather. “Grape black rot originated in eastern North America, but now occurs in portions of Europe, South America, and Asia. It can cause complete crop loss in warm, humid climates, but is virtually unknown in regions with arid summers.” The name comes from the black fringe that borders growing brown patches on the leaves. The disease also attacks other parts of the plant, “all green parts of the vine: the shoots, leaf and fruit stems, tendrils, and fruit. The most damaging effect is to the fruit',
             'Esca (Black Measles)':'Grape esca, also known as black measles, is a fungal disease caused by several different species, including Phaeomoniella chlamydospora and Phaeoacremonium spp. It is characterized by dark, necrotic spots on leaves and canes, often surrounded by a yellow halo. Symptoms may also include internal discoloration of wood and berries. Esca can result in reduced vine vigor and crop quality.',
             'Grape Leaf blight (Isariopsis Leaf Spot)':' Grape leaf blight, caused by the fungus Isariopsis viticola, manifests as small, circular lesions on grape leaves. These lesions may initially appear water-soaked and later turn tan or gray with a reddish border. Severe infections can lead to defoliation and reduce fruit quality.',
             'Healthy' : 'A grape classified as "healthy" shows no visible signs of disease. Leaves are green and intact, without any lesions, spots, or discoloration. The canes and fruit appear normal, with no signs of damage or decay.',

        },
        'medicine_links': {
             'Healthy' : None,
             'Black Rot':'https://example.com/black_rot_medicine',
             'Esca (Black Measles)':'https://example.com/esca',
             'Grape Leaf blight (Isariopsis Leaf Spot)':'https://example.com/grape_leaf_blight'
        }
    },

    translator_text('Pepper Bell',target_language): {
        'model_path': 'pepper_bell_model.keras',
        'classes': ['Bacterial Spot' ,'Healthy'],
        'descriptions': {
            'Healthy': 'The plant appears to be healthy with no signs of disease.',
            'Bacterial Spot': 'Bacterial spot, caused by Xanthomonas campestris, \
            appears as small, dark spots on the leaves, which may develop into larger lesions. \
            It can cause defoliation and yield loss if not managed properly.'
        },
        'medicine_links': {
            'Healthy': None,
            'Bacterial Spot': 'https://example.com/bacterial_spot_medicine'
        }
    },
    translator_text('Apple',target_language): {
        'model_path': 'apple_model.keras',
        'classes': ['Apple Scab', 'Black Rot', 'Cedar Apple Rust',  'Healthy' ],
        'descriptions': {
             'Healthy' : 'The plant appears to be healthy with no signs of disease.',
            'Apple Scab': 'Apple scab, caused by the fungus Venturia inaequalis, \
            appears as olive-green to black spots on the leaves, which may also affect \
            the fruit and twigs. It can cause defoliation and yield loss if not managed properly.',
            'Cedar Apple Rust': 'Cedar apple rust, caused by the fungus Gymnosporangium juniperi-virginianae, \
            manifests as yellow-orange spots on the leaves, which can lead to premature leaf drop \
            and reduced fruit quality.',
            'Black Rot': 'Black rot, caused by the fungus Botryosphaeria obtusa, \
            results in dark, sunken lesions on the fruit and leaves. It can cause fruit rot \
            and significant yield loss.'
        },
        'medicine_links': {
             'Healthy' : None,
            'Apple Scab': 'https://example.com/apple_scab_medicine',
            'Cedar Apple Rust': 'https://example.com/cedar_apple_rust_medicine',
            'Black Rot': 'https://example.com/black_rot_medicine'
        }
    }

}

# Function to load a model for a given plant
def load_model(plant):
    model_path = plants_info[plant]['model_path']
    return tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array,0)# Create a batch
    return img_array

# Function to make predictions for a given plant
def predict_disease(image, plant):
    model = load_model(plant)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    classes = plants_info[plant]['classes']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return prediction,predicted_class, confidence, plants_info[plant]['descriptions'][predicted_class]

# Streamlit app
st.title(translator_text('Plant Disease Detection',target_language))
plant_type = st.selectbox(translator_text('Select the type of plant:',target_language), list(plants_info.keys()))
uploaded_file = st.file_uploader(translator_text("Choose an image...",target_language), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption=translator_text('Uploaded Image',target_language), use_column_width=True)

    # Make prediction based on selected plant type
    st.subheader(f'{plant_type} {translator_text("Disease Detection", target_language)}')

    prediction, predicted_class, confidence, description = predict_disease(image, plant_type)
    st.write(translator_text('Predicted Class:',target_language), translator_text(predicted_class,target_language))
    st.write(translator_text('Confidence:',target_language),confidence)
    st.write(translator_text('Description:',target_language), translator_text(description,target_language))

    # Display medicine link if available
    medicine_link = plants_info[plant_type]['medicine_links'][predicted_class]
    if medicine_link:
       markdown_link = f"[{predicted_class} Medicine]({medicine_link})"
        # Concatenate the text and the Markdown link
       combined_text = f"{translator_text('Medicine Link:', target_language)} {markdown_link}"

        # Display the combined text and Markdown link in the same line
       st.write(combined_text, unsafe_allow_html=True)
    else:
        st.write(translator_text("Congratulation! Your Plant is healthy.",target_language))
